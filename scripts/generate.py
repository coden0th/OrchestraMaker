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

from orchestramaker.batches import load_pieces
from orchestramaker.instruments import INSTRUMENTS
from orchestramaker.metrics import pick_take, reference_windows
from orchestramaker.model import GPT, GPTConfig
from orchestramaker.takes import save_take
from orchestramaker.tokenizer import Tokenizer

# --all plays the pairings the checkpoint's vocabulary knows.
PAIRINGS = [("mozart", "piano"), ("bach", "piano"), ("chopin", "piano"), ("beethoven", "piano"),
            ("debussy", "piano"), ("liszt", "piano"), ("jazz_art_tatum", "piano"), ("jazz_brad_mehldau", "piano"),
            ("jazz_keith_jarrett", "piano"), ("jazz_oscar_peterson", "piano"), ("jazz_piano", "piano"),
            ("jazz_bebop", "alto_sax"), ("jazz_hardbop", "tenor_sax"), ("jazz_cool", "trumpet"),
            ("jazz_swing", "acoustic_bass"), ("jazz_postbop", "jazz_guitar")]


def load(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_config"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"loaded {checkpoint.name}: step {ckpt['step']}, val {ckpt['val']}")
    info = {"checkpoint": str(checkpoint.relative_to(ROOT)), "step": ckpt["step"],
            "val_loss": {k: round(v, 3) for k, v in ckpt["val"].items()},
            "data": ckpt["config"].get("data", "data/tokens")}
    return model, Tokenizer(**ckpt["vocab"]), info


def prime_tokens(tokenizer, data, style, instrument, seconds):
    piece = next(p for p in load_pieces(ROOT / data) if p["split"] == "validation" and p["style"] == style
                 and p["instrument"] == instrument)
    _, _, notes = tokenizer.decode(piece["tokens"])
    print(f"  priming with {seconds}s of: {piece['title']}")
    return tokenizer.encode([n for n in notes if n.start < seconds], style, instrument)[:-1], piece["title"]


def play(model, tokenizer, info, style, instrument, args, device, out_dir):
    meta = {"style": style, "instrument": instrument, **info,
            "temperature": args.temperature, "top_p": args.top_p, "min_p": getattr(args, "min_p", 0.0),
            "seed": args.seed,
            "memory": getattr(args, "memory", 0)}
    if args.prime:
        ids, meta["prime_title"] = prime_tokens(tokenizer, info["data"], style, instrument, args.prime)
        meta["prime_seconds"] = args.prime
    else:
        ids = [tokenizer.bos, tokenizer.index[f"STYLE_{style}"], tokenizer.index[f"INST_{instrument}"]]
    candidates = getattr(args, "candidates", 1)
    idx = torch.tensor([ids], device=device).repeat(candidates, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out, coherence = model.generate(idx, args.tokens, temperature=tokenizer.temperatures(args.temperature, getattr(args, 'pitch_temperature', None)), top_p=args.top_p,
                                        eos=tokenizer.eos, memory=getattr(args, "memory", 0),
                                        min_p=getattr(args, "min_p", 0.0), return_logprob=True)
    takes = [[n for n in tokenizer.decode(row.tolist())[2] if n.start < args.seconds] for row in out]
    notes = takes[0]
    if candidates > 1:  # keep a coherent take whose statistics look like real music of this style
        ref = reference_windows(load_pieces(ROOT / info["data"]), tokenizer, style, args.seconds, 40,
                                from_start=not args.prime, instrument=instrument)
        if ref:
            best, meta["choice"] = pick_take(takes, ref, coherence)
            notes = takes[best]
    inst = INSTRUMENTS[instrument]
    raw_problems = inst.check(notes)
    fixed = inst.make_playable(notes)
    name = f"{style}_{instrument}" + ("_primed" if args.prime else "")
    save_take(out_dir, name, [("as generated", [(inst, notes)]), ("made playable", [(inst, fixed)])], meta)
    length = max((n.end for n in notes), default=0)
    print(f"  {name}: {len(notes)} notes, {length:.0f}s, {len(raw_problems)} playability problems as generated"
          f"{' e.g. ' + raw_problems[0] if raw_problems else ''}; after fix: {len(inst.check(fixed))}"
          f"{'; picked: ' + meta['choice'] if 'choice' in meta else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/v1_piano/best.pt")
    ap.add_argument("--style")
    ap.add_argument("--instrument")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--prime", type=float, default=0, help="seconds of a real piece to continue from")
    ap.add_argument("--tokens", type=int, default=2400)
    ap.add_argument("--seconds", type=float, default=45)
    # Measured with evaluate_samples.py: top_p < 1 and temperature 1.0 made takes sparse and loopy.
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--min_p", type=float, default=0.0, help="note: made Chopin takes go silent")
    ap.add_argument("--pitch_temperature", type=float, help="separate temperature for which notes to play")
    ap.add_argument("--candidates", type=int, default=8, help="generate N, keep the best (see pick_take)")
    ap.add_argument("--memory", type=int, default=0,
                    help="keep the first N tokens (the opening) in context for long pieces, e.g. 400")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/stage4")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, info = load(ROOT / args.checkpoint, device)
    out_dir = ROOT / args.out
    known = [(s, i) for s, i in PAIRINGS if s in tokenizer.styles and i in tokenizer.instruments]
    for style, instrument in known if args.all else [(args.style, args.instrument)]:
        play(model, tokenizer, info, style, instrument, args, device, out_dir)
    print(f"WAV files in {out_dir}")


if __name__ == "__main__":
    main()
