"""Fingerprint of popular repertoire: 8-note pitch sequences that occur in at least --min-performances
different performances. A take made mostly of these is quoting a well-known piece (K.545 and friends),
so take selection can skip it. Only a sample of the training data is needed: a piece popular enough to be
memorized has many performances in any sizeable slice of the data.

Run: .venv/bin/python scripts/build_fingerprint.py --data data/tokens_abl_compact data/tokens_v15
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from check_memorization import sequence_hashes  # noqa: E402

from orchestramaker.batches import load_pieces  # noqa: E402
from orchestramaker.tokenizer import Tokenizer  # noqa: E402

OUT = ROOT / "data/fingerprint_popular_8.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/tokens_abl_compact", "data/tokens_v15"])
    ap.add_argument("--min-performances", type=int, default=5)
    args = ap.parse_args()

    per_piece, seen = [], set()
    for data in args.data:
        tok = Tokenizer.load(ROOT / data / "vocab.json")
        for p in load_pieces(ROOT / data):
            key = (p["source"], p["title"], len(p["tokens"]))
            if key in seen:  # the same performance tokenized in two datasets
                continue
            seen.add(key)
            pitches = tok.pitch_of[np.asarray(p["tokens"])]
            per_piece.append(np.unique(sequence_hashes(pitches[pitches >= 0], 8)))
    hashes, counts = np.unique(np.concatenate(per_piece), return_counts=True)
    popular = hashes[counts >= args.min_performances]
    np.save(OUT, popular)
    print(f"{len(per_piece)} performances, {len(hashes):,} distinct 8-note sequences, "
          f"{len(popular):,} in >= {args.min_performances} performances -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
