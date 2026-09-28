"""A/B test of the critic: from the same 8 candidates, the take picked by coherence vs by the critic.

The listener rates both blind (serve.py --pool outputs/v1_5/ab_pool.json); scripts/ab_report.py then
compares the ratings per method. Pairs where both methods pick the same take are skipped.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import ROOT, critic, fingerprint, load  # noqa: E402

from orchestramaker.batches import load_pieces  # noqa: E402
from orchestramaker.instruments import PIANO  # noqa: E402
from orchestramaker.metrics import pick_take, reference_windows  # noqa: E402
from orchestramaker.takes import save_take  # noqa: E402
from orchestramaker.tokenizer import TIME_STEP  # noqa: E402

OUT = ROOT / "outputs/v1_5/ab"
MANIFEST = ROOT / "outputs/v1_5/ab_pool.json"
STYLES = ["mozart", "chopin", "beethoven", "bach", "debussy", "schubert", "liszt", "satie",
          "jazz_piano", "jazz_art_tatum", "ragtime", "blues"]
SECONDS, CANDIDATES = 30, 8


def main():
    model, tok, info = load(ROOT / "checkpoints/v1_5/best.pt", "cuda")
    pieces = load_pieces(ROOT / info["data"])
    urls, same = [], 0
    for n, style in enumerate(STYLES):
        torch.manual_seed(1000 + n)
        temperature = 1.0 if style == "mozart" else 1.1
        idx = torch.tensor([[tok.bos, tok.index[f"STYLE_{style}"], tok.index["INST_piano"]]], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rows, coherence = model.generate(idx.repeat(CANDIDATES, 1), 1800, temperature=tok.temperatures(temperature),
                                             top_p=1.0, eos=tok.eos, return_logprob=True,
                                             time_steps=tok.time_steps(), min_steps=SECONDS / TIME_STEP)
        takes = [[x for x in tok.decode(r.tolist())[2] if x.start < SECONDS] for r in rows]
        ref = reference_windows(pieces, tok, style, SECONDS, 40, from_start=True)
        a, why_a = pick_take(takes, ref, coherence, fingerprint())
        b, why_b = pick_take(takes, ref, lambda i: critic().score(PIANO, takes[i]), fingerprint(), label="critic")
        if a == b:
            same += 1
            print(f"  {style}: both pick the same take")
            continue
        for method, i, why in (("coherence", a, why_a), ("critic", b, why_b)):
            meta = {"title": f"{style}: picked by {method}", "style": style, "instrument": "piano", **info,
                    "temperature": temperature, "ab_method": method, "ab_style": style, "choice": why}
            urls.append(save_take(OUT, f"{style}_{method}", [("as generated", [(PIANO, takes[i])])], meta))
        print(f"  {style}: coherence -> #{a + 1}, critic -> #{b + 1}", flush=True)
    MANIFEST.write_text(json.dumps({"takes": ["/" + str(u.relative_to(ROOT)) for u in urls]}, indent=1))
    print(f"{len(urls)} takes ({len(urls) // 2} pairs, {same} styles where both agreed) -> {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
