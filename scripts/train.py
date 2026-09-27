"""Stage 3: train the performer model.

Run:    .venv/bin/python scripts/train.py configs/base.json
Tweak:  ... --set max_steps=200 --set batch_size=64 --set model.n_layer=12
Resume: ... --resume   (continues from <out_dir>/last.pt)
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchestramaker.batches import BatchSampler, load_pieces
from orchestramaker.model import GPT, GPTConfig
from orchestramaker.tokenizer import Tokenizer


def load_config(path, overrides):
    cfg = json.loads(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        target = cfg
        for p in parents:
            target = target[p]
        try:
            target[leaf] = json.loads(value)
        except json.JSONDecodeError:
            target[leaf] = value  # plain string, e.g. out_dir=checkpoints/big
    return cfg


def init_weights(model, checkpoint, device):
    """Start from a trained model. A longer context gets the learned position embeddings stretched
    (linear interpolation: position i of the new context borrows from position i * old / new)."""
    state = torch.load(checkpoint, map_location=device, weights_only=False)["model"]
    old = state.get("pos_emb.weight")
    new = model.pos_emb.weight if model.pos_emb is not None else None
    if old is not None and new is not None and old.shape[0] != new.shape[0]:
        state["pos_emb.weight"] = torch.nn.functional.interpolate(
            old.T[None], size=new.shape[0], mode="linear", align_corners=True)[0].T
        print(f"stretched position embeddings {old.shape[0]} -> {new.shape[0]}")
    model.load_state_dict(state)
    print(f"initialized from {checkpoint}")


def lr_at(step, cfg):
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    progress = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate(model, sampler, cfg, device):
    """Validation loss per source on the same windows every time: per token, and in bits per note
    (comparable across tokenizers that spend a different number of tokens on a note)."""
    model.eval()
    losses, bits = {}, {}
    is_note_end = torch.as_tensor(sampler.tok.is_note_end, device=device)
    batches = cfg["eval_batches"] * cfg.get("grad_accum", 1)
    for source in sampler.sources:
        sampler.rng = np.random.default_rng(1234)
        total, nats, tokens, notes = 0.0, 0.0, 0, 0
        for _ in range(batches):
            x, y = sampler.batch(cfg["batch_size"] // cfg.get("grad_accum", 1), source=source, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(x, y)[1].item()
            n = int((y != sampler.tok.pad).sum())
            total, nats, tokens, notes = total + loss, nats + loss * n, tokens + n, notes + int(is_note_end[y].sum())
        losses[source], bits[source] = total / batches, nats / max(notes, 1) / math.log(2)
    model.train()
    return losses, bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. max_steps=200")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)

    data = ROOT / cfg.get("data", "data/tokens")
    tokenizer = Tokenizer.load(data / "vocab.json")
    pieces = load_pieces(data)
    common = dict(tokenizer=tokenizer, source_weights=cfg["source_weights"], block_size=cfg["model"]["block_size"])
    train = BatchSampler(pieces, split="train", max_transpose=cfg["max_transpose"], seed=cfg["seed"], **common)
    val = BatchSampler(pieces, split="validation", **common)

    model_cfg = GPTConfig(vocab_size=len(tokenizer), **cfg["model"])
    model = GPT(model_cfg).to(device)
    model.grad_checkpoint = cfg.get("grad_checkpoint", False)
    if cfg.get("init_from") and not (args.resume and (out / "last.pt").exists()):
        init_weights(model, ROOT / cfg["init_from"], device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=(0.9, 0.95), fused=device == "cuda")
    step, best = 0, float("inf")
    if args.resume and (out / "last.pt").exists():
        ckpt = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        step, best = ckpt["step"], ckpt["best_val"]
        print(f"resumed from step {step}")
    fwd = torch.compile(model) if cfg["compile"] else model
    print(f"model: {model.num_params() / 1e6:.1f}M params on {device}"
          f" ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})")
    (out / "config.json").write_text(json.dumps(cfg, indent=1))

    def save(name, val_losses, with_optimizer):
        ckpt = {"model": model.state_dict(), "model_config": model_cfg.to_dict(),
                "vocab": tokenizer.config(),
                "config": cfg, "step": step, "best_val": best, "val": val_losses}
        if with_optimizer:
            ckpt["optimizer"] = opt.state_dict()
        torch.save(ckpt, out / f"{name}.tmp")
        os.replace(out / f"{name}.tmp", out / name)  # atomic: watchers never read a half-written file

    accum = cfg.get("grad_accum", 1)  # batch_size is split into `accum` micro-batches that fit in memory
    tokens_per_step = cfg["batch_size"] * cfg["model"]["block_size"]
    t0, log_t0 = time.time(), time.time()
    while step < cfg["max_steps"]:
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            x, y = train.batch(cfg["batch_size"] // accum, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = fwd(x, y)
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        step += 1

        if step % 50 == 0:
            torch.cuda.synchronize() if device == "cuda" else None
            rate = 50 * tokens_per_step / (time.time() - log_t0)
            eta = (cfg["max_steps"] - step) * tokens_per_step / rate / 60
            print(f"step {step:6d}  loss {loss.item():.3f}  lr {lr_at(step, cfg):.1e}  "
                  f"{rate / 1e3:.0f}k tok/s  eta {eta:.0f} min", flush=True)
            log_t0 = time.time()
        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            losses, bits = evaluate(model, val, cfg, device)
            mean = sum(losses.values()) / len(losses)
            print(f"  val {'  '.join(f'{k} {v:.3f}' for k, v in losses.items())}  (mean {mean:.3f})", flush=True)
            print(f"  bits/note {'  '.join(f'{k} {v:.2f}' for k, v in bits.items())}", flush=True)
            if mean < best:
                best = mean
                save("best.pt", losses, with_optimizer=False)  # small: for listening / downloading
            save("last.pt", losses, with_optimizer=True)   # for --resume
            log_t0 = time.time()
    print(f"done in {(time.time() - t0) / 60:.1f} min, best mean val loss {best:.3f}")


if __name__ == "__main__":
    main()
