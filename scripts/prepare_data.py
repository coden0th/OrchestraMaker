"""Stage 2: download the datasets, tokenize them and write data/tokens/.

Also decodes a few pieces back from tokens and renders them, so the pipeline can be checked by ear.
Run: .venv/bin/python scripts/prepare_data.py
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchestramaker.datasets import load_maestro, load_wjazzd
from orchestramaker.instruments import INSTRUMENTS
from orchestramaker.render import render
from orchestramaker.tokenizer import Tokenizer

OUT = ROOT / "data/tokens"
LISTEN = ROOT / "outputs/stage2"


def main():
    print("loading Weimar Jazz Database...")
    pieces = load_wjazzd()
    print("loading MAESTRO...")
    pieces += load_maestro()

    tokenizer = Tokenizer(sorted({p.style for p in pieces}), sorted({p.instrument for p in pieces}))
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer.save(OUT / "vocab.json")

    records, stats = [], defaultdict(lambda: np.zeros(4))
    out_of_range = defaultdict(lambda: [0, 0])
    for p in pieces:
        ids = np.array(tokenizer.encode(p.notes, p.style, p.instrument), dtype=np.int16)
        records.append({"source": p.source, "title": p.title, "style": p.style,
                        "instrument": p.instrument, "split": p.split, "tokens": ids})
        seconds = max(n.end for n in p.notes)
        stats[(p.style, p.instrument)] += [1, len(p.notes), len(ids), seconds / 3600]
        inst = INSTRUMENTS[p.instrument]
        out_of_range[p.instrument][0] += sum(not inst.low <= n.pitch <= inst.high for n in p.notes)
        out_of_range[p.instrument][1] += len(p.notes)
    torch.save({"vocab": OUT / "vocab.json", "pieces": records}, OUT / "pieces.pt")

    print(f"\nvocab: {len(tokenizer)} tokens")
    print(f"{'style':22s} {'instrument':14s} {'pieces':>6s} {'notes':>9s} {'tokens':>10s} {'hours':>6s}")
    for (style, inst), (n, notes, toks, hours) in sorted(stats.items()):
        print(f"{style:22s} {inst:14s} {n:6.0f} {notes:9.0f} {toks:10.0f} {hours:6.1f}")
    for split in ("train", "validation", "test"):
        print(f"{split:10s} {sum(len(r['tokens']) for r in records if r['split'] == split):>11,d} tokens")
    print("\nnotes outside the standard instrument range (real players, kept as-is):")
    for inst, (bad, total) in sorted(out_of_range.items()):
        if bad:
            print(f"  {inst:14s} {bad:6d} / {total:8d} ({100 * bad / total:.1f}%)")

    # Round trip: tokens -> notes -> audio, first 30 seconds.
    LISTEN.mkdir(parents=True, exist_ok=True)
    picks = [lambda r: r["style"] == "mozart", lambda r: r["instrument"] == "tenor_sax",
             lambda r: r["instrument"] == "acoustic_bass"]
    for pick in picks:
        r = next(r for r in records if pick(r))
        style, inst, notes = tokenizer.decode(r["tokens"])
        notes = [n for n in notes if n.start < 30]
        path = render([(INSTRUMENTS[inst], notes)], LISTEN / f"roundtrip_{style}_{inst}.wav")
        print(f"rendered {path.name}: {r['title']}")


if __name__ == "__main__":
    main()
