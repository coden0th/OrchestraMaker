"""How good are the model's takes, compared with real music of the same style?

Generates --count takes, measures them (orchestramaker/metrics.py) next to random windows of real
validation performances, and sorts each take into: repetitive (repeats itself more than 90% of
real windows), sparse (fewer notes or pitches than 90% of real windows), or typical.

Run: .venv/bin/python scripts/evaluate_samples.py --style chopin --count 16
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import ROOT, load  # noqa: E402

from orchestramaker.batches import load_pieces  # noqa: E402
from orchestramaker.instruments import INSTRUMENTS  # noqa: E402
from orchestramaker.metrics import KEYS, classify, describe, reference_windows  # noqa: E402
from orchestramaker.takes import save_take  # noqa: E402

def generate_batch(model, tokenizer, style, instrument, args, device):
    idx = torch.tensor([[tokenizer.bos, tokenizer.index[f"STYLE_{style}"], tokenizer.index[f"INST_{instrument}"]]],
                       device=device).repeat(args.count, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model.generate(idx, args.tokens, temperature=args.temperature, top_p=args.top_p,
                             )
    return [[n for n in tokenizer.decode(row.tolist())[2] if n.start < args.seconds] for row in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/v1_piano/best.pt")
    ap.add_argument("--style", default="chopin")
    ap.add_argument("--instrument", default="piano")
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=45)
    ap.add_argument("--tokens", type=int, default=2400)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--ref", choices=["openings", "random"], default="openings",
                    help="compare with the first seconds of real pieces, or random windows")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, info = load(ROOT / args.checkpoint, device)
    ref = reference_windows(load_pieces(ROOT / info["data"]), tokenizer, args.style, args.seconds, 60,
                            from_start=args.ref == "openings")
    takes = generate_batch(model, tokenizer, args.style, args.instrument, args, device)

    out_dir = ROOT / "outputs/eval" / f"{args.style}_{args.label}"
    rows, classes = [], []
    for i, notes in enumerate(takes):
        stats = describe(notes)
        if not stats:
            classes.append("empty")
            continue
        cls = classify(stats, ref)
        classes.append(cls)
        rows.append(stats)
        meta = {"title": f"{args.style} #{i + 1} ({cls})", "style": args.style, "instrument": args.instrument,
                **info, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()},
                "temperature": args.temperature, "top_p": args.top_p}
        save_take(out_dir, f"{i + 1:02d}_{cls}", [("as generated", [(INSTRUMENTS[args.instrument], notes)])], meta)

    print(f"\n{args.style}, {args.label}: {len(takes)} takes of {args.seconds:.0f}s vs {len(ref)} real {args.ref}")
    print(f"{'':18s} {'model median':>12s} {'real median':>12s} {'real 10-90%':>16s}")
    for k in KEYS:
        real = [r[k] for r in ref]
        lo, hi = np.percentile(real, [10, 90])
        print(f"{k:18s} {np.median([r[k] for r in rows]):12.2f} {np.median(real):12.2f} {lo:8.2f} – {hi:5.2f}")
    counts = {c: classes.count(c) for c in ("typical", "sparse", "repetitive", "empty") if c in classes}
    print("takes:", ", ".join(f"{c} {n}" for c, n in counts.items()), f"-> {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
