"""Stage 4: let the trained model play.

    .venv/bin/python scripts/generate.py --style jazz_bebop --instrument alto_sax
    .venv/bin/python scripts/generate.py --style mozart --instrument piano --prime 8
    .venv/bin/python scripts/generate.py --all          # one take for a set of pairings

--prime N starts from the first N seconds of a real validation piece and lets the model continue.
Each take is saved as WAV + JSON with two versions: exactly what the model played, and the same
after make_playable. Browse them with scripts/serve.py.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchestramaker.instruments import INSTRUMENTS
from orchestramaker.model import GPT, GPTConfig
from orchestramaker.takes import save_take
from orchestramaker.tokenizer import Tokenizer

PAIRINGS = [("mozart", "piano"), ("bach", "piano"), ("chopin", "piano"),
            ("jazz_bebop", "alto_sax"), ("jazz_hardbop", "tenor_sax"), ("jazz_cool", "trumpet"),
            ("jazz_swing", "acoustic_bass"), ("jazz_postbop", "jazz_guitar")]


def load(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_config"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"loaded {checkpoint.name}: step {ckpt['step']}, val {ckpt['val']}")
    info = {"checkpoint": str(checkpoint.relative_to(ROOT)), "step": ckpt["step"],
            "val_loss": {k: round(v, 3) for k, v in ckpt["val"].items()}}
    return model, Tokenizer(**ckpt["vocab"]), info


def prime_tokens(tokenizer, style, instrument, seconds):
    pieces = torch.load(ROOT / "data/tokens/pieces.pt", weights_only=False)["pieces"]
    piece = next(p for p in pieces if p["split"] == "validation" and p["style"] == style
                 and p["instrument"] == instrument)
    _, _, notes = tokenizer.decode(piece["tokens"])
    print(f"  priming with {seconds}s of: {piece['title']}")
    return tokenizer.encode([n for n in notes if n.start < seconds], style, instrument)[:-1], piece["title"]


def play(model, tokenizer, info, style, instrument, args, device, out_dir):
    meta = {"style": style, "instrument": instrument, **info,
            "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed}
    if args.prime:
        ids, meta["prime_title"] = prime_tokens(tokenizer, style, instrument, args.prime)
        meta["prime_seconds"] = args.prime
    else:
        ids = [tokenizer.bos, tokenizer.index[f"STYLE_{style}"], tokenizer.index[f"INST_{instrument}"]]
    idx = torch.tensor([ids], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model.generate(idx, args.tokens, temperature=args.temperature, top_p=args.top_p, eos=tokenizer.eos)
    _, _, notes = tokenizer.decode(out[0].tolist())
    notes = [n for n in notes if n.start < args.seconds]
    inst = INSTRUMENTS[instrument]
    raw_problems = inst.check(notes)
    fixed = inst.make_playable(notes)
    name = f"{style}_{instrument}" + ("_primed" if args.prime else "")
    save_take(out_dir, name, [("as generated", [(inst, notes)]), ("made playable", [(inst, fixed)])], meta)
    length = max((n.end for n in notes), default=0)
    print(f"  {name}: {len(notes)} notes, {length:.0f}s, {len(raw_problems)} playability problems as generated"
          f"{' e.g. ' + raw_problems[0] if raw_problems else ''}; after fix: {len(inst.check(fixed))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/base/best.pt")
    ap.add_argument("--style")
    ap.add_argument("--instrument")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--prime", type=float, default=0, help="seconds of a real piece to continue from")
    ap.add_argument("--tokens", type=int, default=1500)
    ap.add_argument("--seconds", type=float, default=45)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/stage4")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, info = load(ROOT / args.checkpoint, device)
    out_dir = ROOT / args.out
    for style, instrument in PAIRINGS if args.all else [(args.style, args.instrument)]:
        play(model, tokenizer, info, style, instrument, args, device, out_dir)
    print(f"WAV files in {out_dir}")


if __name__ == "__main__":
    main()
