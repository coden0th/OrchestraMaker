"""Our v1 vs EleutherAI's Aria (0.7B parameters, 60k hours of piano, Apache-2.0) on the same prompts.

Both continue the first --prompt seconds of the same held-out performances. For every take we report
note statistics and how much of it copies the real continuation of that piece (Aria's README warns it
may reproduce popular classical works). Takes go to outputs/aria_vs_v1/<prompt>/ for the web UI.

Setup: Aria source in data/external/aria (pip install --no-deps -e), weights in data/external/aria-weights.
Run:   .venv/bin/python scripts/compare_aria.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

import mido
import numpy as np
import symusic
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_memorization import sequence_hashes  # noqa: E402
from generate import ROOT, fingerprint, load  # noqa: E402

from orchestramaker.batches import load_pieces  # noqa: E402
from orchestramaker.datasets import score_notes  # noqa: E402
from orchestramaker.instruments import PIANO, Note  # noqa: E402
from orchestramaker.metrics import describe, pick_take, reference_windows  # noqa: E402
from orchestramaker.takes import save_take  # noqa: E402
from orchestramaker.tokenizer import TIME_STEP  # noqa: E402

ARIA_WEIGHTS = ROOT / "data/external/aria-weights/aria-medium-gen.safetensors"
STYLES = ["mozart", "chopin", "debussy", "jazz_piano"]


def write_midi(notes: list[Note], path: Path):
    """Notes in seconds -> MIDI at 120 bpm (1 s = 960 ticks)."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    events = sorted([(round(n.start * 960), 1, n.pitch, n.velocity) for n in notes] +
                    [(round(n.end * 960), 0, n.pitch, 0) for n in notes])
    now = 0
    for tick, on, pitch, vel in events:
        track.append(mido.Message("note_on" if on else "note_off", note=pitch, velocity=vel if on else 0,
                                  time=tick - now))
        now = tick
    mid.save(path)


def copy_share(take: list[Note], original: list[Note], n: int = 8) -> float:
    """Share of the take's n-note pitch sequences that also occur in the real continuation."""
    seq = lambda ns: [x.pitch for x in sorted(ns, key=lambda x: (x.start, x.pitch))]
    mine = np.unique(sequence_hashes(seq(take), n))
    return float(np.isin(mine, sequence_hashes(seq(original), n)).mean()) if len(mine) else 0.0


def aria_continue(prompt_mid: Path, seconds: float, variations: int, out_dir: Path) -> list[list[Note]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.mid"):
        old.unlink()
    subprocess.run([str(ROOT / ".venv/bin/aria"), "generate", "--backend", "torch_cuda",
                    "--checkpoint_path", str(ARIA_WEIGHTS), "--prompt_midi_path", str(prompt_mid),
                    "--prompt_duration", str(seconds), "--variations", str(variations),
                    "--temp", "0.98", "--min_p", "0.035", "--length", "2048", "--save_dir", str(out_dir)],
                   check=True, capture_output=True)
    return [score_notes(symusic.Score(str(f), ttype="second")) for f in sorted(out_dir.glob("*.mid"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=float, default=10)
    ap.add_argument("--variations", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=70)
    ap.add_argument("--checkpoint", default="checkpoints/v1_piano/best.pt")
    ap.add_argument("--name", default="v1", help="label for our model in take names and the output folder")
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--min_p", type=float, default=0.0)
    args = ap.parse_args()

    device = "cuda"
    model, tok, info = load(ROOT / args.checkpoint, device)
    pieces = load_pieces(ROOT / info["data"])
    rng = np.random.default_rng(3)
    print(f"{'prompt':34s} {'model':6s} {'notes/s':>7s} {'pitches':>7s} {'repeat':>6s} {'copies original':>15s}")
    for style in STYLES:
        candidates = [p for p in pieces if p["style"] == style and p["split"] == "validation"
                      and len(p["tokens"]) > 6000]
        piece = candidates[rng.integers(len(candidates))]
        _, _, notes = tok.decode(np.asarray(piece["tokens"]))
        t0 = min(n.start for n in notes)
        notes = [Note(n.pitch, n.start - t0, n.duration, n.velocity) for n in notes]
        prompt = [n for n in notes if n.start < args.prompt]
        original = [n for n in notes if args.prompt <= n.start < args.seconds]
        out = ROOT / f"outputs/aria_vs_{args.name.replace('.', '_')}" / style
        out.mkdir(parents=True, exist_ok=True)
        write_midi(prompt, out / "prompt.mid")
        title = piece["title"].replace("_", " ")
        common = {"style": style, "instrument": "piano", "prime_seconds": args.prompt, "prime_title": title}
        save_take(out, "0_original", [("real performance", [(PIANO, [n for n in notes if n.start < args.seconds])])],
                  {**common, "title": f"Original: {title}"})

        # v1: best of 8 per variation, like generate.py
        ids = tok.encode(prompt, style, "piano")[:-1]
        ref = reference_windows(pieces, tok, style, 45, 30, from_start=True)
        v1_takes = []
        for v in range(args.variations):
            torch.manual_seed(v)
            idx = torch.tensor([ids], device=device).repeat(8, 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rows, coherence = model.generate(idx, 2600, temperature=tok.temperatures(args.temperature), top_p=1.0,
                                                 min_p=args.min_p,
                                                 eos=tok.eos, return_logprob=True, time_steps=tok.time_steps(),
                                                 min_steps=args.seconds / TIME_STEP)
            takes = [[n for n in tok.decode(r.tolist())[2] if n.start < args.seconds] for r in rows]
            best, _ = pick_take(takes, ref, coherence, fingerprint())
            v1_takes.append(takes[best])
        torch.cuda.empty_cache()

        aria_takes = [[n for n in t if n.start < args.seconds]
                      for t in aria_continue(out / "prompt.mid", args.prompt, args.variations, out / "aria_midi")]
        for name, takes in ((args.name, v1_takes), ("aria", aria_takes)):
            for v, take in enumerate(takes):
                cont = [n for n in take if n.start >= args.prompt]
                stats, copied = describe(cont), copy_share(cont, original)
                save_take(out, f"{name}_{v + 1}", [("continuation", [(PIANO, take)])],
                          {**common, "title": f"{name} #{v + 1} continuing {title}",
                           "copies_original": round(copied, 3), **{k: round(x, 3) for k, x in stats.items()}})
                print(f"{title[:34]:34s} {name:6s} {stats.get('notes_per_s', 0):7.1f} "
                      f"{stats.get('distinct_pitches', 0):7.0f} {stats.get('repetition', 0):6.2f} {copied:15.1%}",
                      flush=True)
        real = describe(original)
        print(f"{title[:34]:34s} {'real':6s} {real['notes_per_s']:7.1f} {real['distinct_pitches']:7.0f} "
              f"{real['repetition']:6.2f} {'':>15s}")


if __name__ == "__main__":
    main()
