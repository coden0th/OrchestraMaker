"""Is a generated take original, or copied from the training data?

Compares the take's N-note pitch sequences (notes ordered by onset, then pitch - the token order)
with every training piece of the same style. Reports the share found anywhere, and the pieces
sharing the most. A few percent is normal (scales, arpeggios); a high share for one piece means
the model is replaying it.

Run: .venv/bin/python scripts/check_memorization.py outputs/progress/v1_piano/step_16000/chopin_piano.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchestramaker.batches import load_pieces  # noqa: E402
from orchestramaker.tokenizer import Tokenizer  # noqa: E402


def sequence_hashes(pitches, n: int) -> np.ndarray:
    p = np.asarray(pitches, dtype=np.int64)
    if len(p) < n:
        return np.empty(0, dtype=np.int64)
    h = np.zeros(len(p) - n + 1, dtype=np.int64)
    for i in range(n):
        h = h * 131 + p[i:len(p) - n + 1 + i]
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take", type=Path)
    ap.add_argument("--data", help="token directory (default: the one the take's checkpoint was trained on)")
    ap.add_argument("--style", help="default: the take's style")
    ap.add_argument("-n", type=int, default=8)
    args = ap.parse_args()

    take = json.loads(args.take.read_text())
    style = args.style or take["meta"]["style"]
    data = ROOT / (args.data or take["meta"].get("data", "data/tokens"))
    notes = take["versions"][0]["tracks"][0]["notes"]
    generated = np.unique(sequence_hashes([n[0] for n in sorted(notes, key=lambda n: (n[1], n[0]))], args.n))

    tokenizer = Tokenizer.load(data / "vocab.json")
    found, results = np.zeros(len(generated), dtype=bool), []
    for piece in load_pieces(data):
        if piece["style"] != style or piece["split"] != "train":
            continue
        ids = np.asarray(piece["tokens"]).astype(np.int64)
        pitches = tokenizer.pitch_of[ids]
        hit = np.isin(generated, sequence_hashes(pitches[pitches >= 0], args.n))
        found |= hit
        results.append((hit.mean(), piece["title"]))
    results.sort(reverse=True)
    print(f"{args.take.name}: {len(generated)} distinct {args.n}-note sequences, "
          f"checked against {len(results)} '{style}' training pieces in {data.relative_to(ROOT)}")
    print(f"found anywhere: {found.mean():.1%}")
    for share, title in results[:5]:
        print(f"  {share:5.1%}  {title}")


if __name__ == "__main__":
    main()
