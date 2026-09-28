"""Did the critic pick takes the listener likes more? Compares blind ratings of the A/B pool pairs."""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "outputs/v1_5/ab_pool.json"
RATINGS = ROOT / "outputs/ratings.json"


def main():
    ratings = json.loads(RATINGS.read_text()) if RATINGS.exists() else {}
    pairs = {}
    for url in json.loads(POOL.read_text())["takes"]:
        meta = json.loads((ROOT / url.lstrip("/")).read_text())["meta"]
        if url in ratings:
            pairs.setdefault(meta["ab_style"], {})[meta["ab_method"]] = ratings[url]["rating"]
    done = {s: p for s, p in pairs.items() if len(p) == 2}
    if not done:
        sys.exit("no complete pairs rated yet")
    wins = sum(p["critic"] > p["coherence"] for p in done.values())
    losses = sum(p["critic"] < p["coherence"] for p in done.values())
    for style, p in sorted(done.items()):
        print(f"  {style:16s} coherence {p['coherence']}  critic {p['critic']}")
    mean = lambda m: sum(p[m] for p in done.values()) / len(done)
    print(f"{len(done)} pairs: mean rating coherence {mean('coherence'):.2f}, critic {mean('critic'):.2f}; "
          f"critic better {wins}, worse {losses}, tied {len(done) - wins - losses}")
    n = wins + losses  # two-sided sign test on the untied pairs
    if n:
        p = min(1.0, 2 * sum(math.comb(n, k) for k in range(max(wins, losses), n + 1)) / 2 ** n)
        print(f"sign test p = {p:.2f} ({'a real difference' if p < 0.05 else 'could still be chance with this few pairs'})")


if __name__ == "__main__":
    main()
