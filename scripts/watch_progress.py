"""Listen to the model while it learns.

Whenever a run saves a new best checkpoint (at most once every --every steps), generate a few takes
into outputs/progress/<run>/step_XXXXX/. Stops after the run's final checkpoint.

Run: .venv/bin/python scripts/watch_progress.py --run checkpoints/base
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import ROOT, load, play  # noqa: E402

PAIRINGS = [("mozart", "piano"), ("jazz_bebop", "alto_sax"), ("jazz_swing", "acoustic_bass")]


def checkpoint_step(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)["step"]
    except Exception:
        return None  # missing, or being written right now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="checkpoints/base")
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--poll", type=float, default=20)
    args = ap.parse_args()

    run = ROOT / args.run
    max_steps = json.loads((run / "config.json").read_text())["max_steps"]
    gen_args = argparse.Namespace(prime=0, tokens=1200, seconds=args.seconds, temperature=1.0, top_p=0.95, seed=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    done_step, seen_mtime = -args.every, None
    print(f"watching {run} (every {args.every} steps, until step {max_steps})", flush=True)
    while True:
        best, last = run / "best.pt", run / "last.pt"
        finished = checkpoint_step(last) == max_steps if last.exists() else False
        mtime = best.stat().st_mtime if best.exists() else None
        if mtime and mtime != seen_mtime:
            step = checkpoint_step(best)
            if step is not None and (step - done_step >= args.every or (finished and step != done_step)):
                model, tokenizer, info = load(best, device)
                out_dir = ROOT / "outputs/progress" / run.name / f"step_{step:05d}"
                torch.manual_seed(0)
                for style, instrument in PAIRINGS:
                    play(model, tokenizer, info, style, instrument, gen_args, device, out_dir)
                del model
                torch.cuda.empty_cache()
                done_step = step
            if step is not None:
                seen_mtime = mtime
        if finished and seen_mtime == mtime:
            print("run finished", flush=True)
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
