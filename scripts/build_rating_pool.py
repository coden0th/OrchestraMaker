"""Build the blind-rating pool for teaching a critic the listener's taste (1-5 ratings in the web UI).

The pool mixes, so the critic sees the whole range from chaotic to excellent:
- existing v1.5 takes (evaluation sets, early and late checkpoints, picked takes),
- new raw v1.5 takes across many styles and temperatures (no candidate selection),
- excerpts of real performances (the "real" end of the scale; the rater doesn't know which is which).

Writes outputs/v1_5/rating_pool.json; serve it with:
  .venv/bin/python scripts/serve.py --port 8001 --title v1.5 --pool outputs/v1_5/rating_pool.json --takes ...
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import ROOT, load  # noqa: E402

from orchestramaker.batches import load_pieces  # noqa: E402
from orchestramaker.instruments import PIANO, Note  # noqa: E402
from orchestramaker.takes import save_take  # noqa: E402
from orchestramaker.tokenizer import TIME_STEP  # noqa: E402

OUT = ROOT / "outputs/v1_5/pool"
MANIFEST = ROOT / "outputs/v1_5/rating_pool.json"
STYLES = ["mozart", "chopin", "beethoven", "bach", "debussy", "liszt", "schubert", "rachmaninoff", "satie",
          "haydn", "ragtime", "blues", "jazz_piano", "jazz_art_tatum", "jazz_oscar_peterson", "jazz_keith_jarrett"]
TEMPERATURES = [0.9, 1.0, 1.1, 1.2, 1.3]


def existing_takes(rng):
    """A sample of the v1.5 takes already made, so earlier experiments get rated too."""
    urls = []
    for group in sorted((ROOT / "outputs/eval").glob("*v1_5*")):
        urls += rng.sample(sorted(group.glob("*.json")), 8)
    for step in ("01000", "03000", "07000", "13000", "19000", "25000"):
        urls += sorted((ROOT / f"outputs/progress/v1_5/step_{step}").glob("*.json"))
    urls += sorted((ROOT / "outputs/v1_5_final").glob("*.json")) + sorted((ROOT / "outputs/v1_5_filtered").rglob("*.json"))
    return urls


def new_takes(seconds):
    model, tok, info = load(ROOT / "checkpoints/v1_5/best.pt", "cuda")
    paths = []
    for t in TEMPERATURES:
        torch.manual_seed(int(t * 100))
        idx = torch.tensor([[tok.bos, tok.index[f"STYLE_{s}"], tok.index["INST_piano"]] for s in STYLES], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rows = model.generate(idx, 1800, temperature=tok.temperatures(t), top_p=1.0, eos=tok.eos,
                                  time_steps=tok.time_steps(), min_steps=seconds / TIME_STEP)
        for style, row in zip(STYLES, rows):
            notes = [n for n in tok.decode(row.tolist())[2] if n.start < seconds]
            meta = {"title": f"{style}, temperature {t}", "style": style, "instrument": "piano", **info,
                    "temperature": t, "source": "generated"}
            paths.append(save_take(OUT, f"gen_{style}_t{t}", [("as generated", [(PIANO, notes)])], meta))
        print(f"  temperature {t}: {len(STYLES)} takes", flush=True)
    return paths


def real_excerpts(count, seconds, rng):
    from orchestramaker.tokenizer import Tokenizer
    tok = Tokenizer.load(ROOT / "data/tokens_v15_pod/vocab.json")
    pieces = [p for p in load_pieces(ROOT / "data/tokens_v15") if p["split"] == "validation" and len(p["tokens"]) > 4000]
    paths = []
    for p in rng.sample(pieces, count):
        _, _, notes = tok.decode(np.asarray(p["tokens"]))
        t0 = rng.uniform(min(n.start for n in notes), max(n.start for n in notes) - seconds)
        excerpt = [Note(n.pitch, n.start - t0, n.duration, n.velocity) for n in notes if t0 <= n.start < t0 + seconds]
        meta = {"title": f"Real: {p['title']}", "style": p["style"], "instrument": "piano", "source": "real"}
        paths.append(save_take(OUT, f"real_{len(paths):02d}", [("real performance", [(PIANO, excerpt)])], meta))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--real", type=int, default=25)
    args = ap.parse_args()
    rng = random.Random(0)
    urls = existing_takes(rng)
    print(f"{len(urls)} existing v1.5 takes")
    urls += new_takes(args.seconds)
    urls += real_excerpts(args.real, args.seconds, rng)
    takes = ["/" + str(Path(u).relative_to(ROOT)) for u in urls]
    MANIFEST.write_text(json.dumps({"takes": takes}, indent=1))
    print(f"{len(takes)} takes in {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
