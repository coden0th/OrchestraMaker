"""Stage 4: let the trained model play.

    .venv/bin/python scripts/generate.py --style jazz_bebop --instrument alto_sax
    .venv/bin/python scripts/generate.py --style mozart --instrument piano --prime 8
    .venv/bin/python scripts/generate.py --all          # one take for a set of pairings

--prime N starts from the first N seconds of a real validation piece and lets the model continue.
Each take is saved as WAV + JSON with two versions: exactly what the model played, and the same
after make_playable. Browse them with scripts/serve.py.
"""

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchestramaker.batches import load_pieces
from orchestramaker.instruments import INSTRUMENTS
from orchestramaker.metrics import harmony_profile, pick_take, reference_windows
from orchestramaker.model import GPT, GPTConfig
from orchestramaker.takes import save_take
from orchestramaker.tokenizer import TIME_STEP, Tokenizer

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


@lru_cache(maxsize=1)
def critic():
    """The listener's critic (scripts/train_critic.py): loads MERT, so only when --critic asks for it."""
    from orchestramaker.critic import Critic
    return Critic()


@lru_cache(maxsize=1)
def fingerprint():
    """Popular-repertoire fingerprint (scripts/build_fingerprint.py), if it has been built."""
    path = ROOT / "data/fingerprint_popular_8.npy"
    return np.load(path) if path.exists() else None


def style_limits(data: str, tokenizer, style: str, instrument: str) -> dict:
    """Real music's level for the brakes, per style: 90th-percentile note density of 10 s windows and the
    median worst-window dissonance of 30 s openings. Computed once, cached in <data>/style_limits.json."""
    cache = ROOT / data / "style_limits.json"
    limits = json.loads(cache.read_text()) if cache.exists() else {}
    key = f"{style}/{instrument}"
    if key not in limits:
        pieces = load_pieces(ROOT / data)
        dens = [w["notes_per_s"] for w in reference_windows(pieces, tokenizer, style, 10, 60, instrument=instrument)]
        rng = np.random.default_rng(0)
        chosen = [p for p in pieces if p["style"] == style and p["split"] == "validation"
                  and p["instrument"] == instrument]
        diss = []
        for i in rng.permutation(len(chosen))[:40]:
            ns = tokenizer.decode(np.asarray(chosen[i]["tokens"]))[2]
            t0 = min(n.start for n in ns)
            diss.append(harmony_profile([n for n in ns if n.start - t0 < 30])["worst_dissonance"])
        limits[key] = {"max_density": float(np.percentile(dens, 90)) if dens else None,
                       "max_dissonance": float(np.median(diss)) if diss else None}
        cache.write_text(json.dumps(limits, indent=1))
    return limits[key]


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
    brakes = {}
    if getattr(args, "brakes", True):  # measured: dissonance back to real music's level, critic +0.2 stars
        limits = style_limits(info["data"], tokenizer, style, instrument)
        meta["brakes"] = limits
        brakes = {"note_starts": tokenizer.pitch_of >= 0, "max_density": limits["max_density"],
                  "pitch_of": tokenizer.pitch_of, "dur_steps": tokenizer.dur_steps(),
                  "clash_penalty": 1.0, "max_dissonance": limits["max_dissonance"]}
    candidates = getattr(args, "candidates", 1)
    idx = torch.tensor([ids], device=device).repeat(candidates, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out, coherence = model.generate(idx, args.tokens, temperature=tokenizer.temperatures(args.temperature, getattr(args, 'pitch_temperature', None)), top_p=args.top_p,
                                        eos=tokenizer.eos, memory=getattr(args, "memory", 0),
                                        min_p=getattr(args, "min_p", 0.0), return_logprob=True,
                                        time_steps=tokenizer.time_steps(), min_steps=args.seconds / TIME_STEP,
                                        **brakes)
    takes = [[n for n in tokenizer.decode(row.tolist())[2] if n.start < args.seconds] for row in out]
    notes = takes[0]
    if candidates > 1:  # keep a coherent take whose statistics look like real music of this style
        ref = reference_windows(load_pieces(ROOT / info["data"]), tokenizer, style, args.seconds, 40,
                                from_start=not args.prime, instrument=instrument)
        if ref:
            if getattr(args, "critic", False):  # listen to each surviving candidate, keep the predicted favourite
                inst = INSTRUMENTS[instrument]
                best, meta["choice"] = pick_take(takes, ref, lambda i: critic().score(inst, takes[i]),
                                                 fingerprint(), label="critic")
            else:
                best, meta["choice"] = pick_take(takes, ref, coherence, fingerprint())
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
    ap.add_argument("--critic", action="store_true", help="pick by the listener's critic instead of coherence")
    ap.add_argument("--no-brakes", dest="brakes", action="store_false",
                    help="play freely: no density brake, no brake on runaway dissonance")
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
