"""Mirror a training run on a remote GPU pod to this machine, so the web UI can follow it.

Every --every seconds: copy train.log + config.json. Whenever the remote best.pt is at least
--ckpt-every steps newer than the local copy, download it too (then watch_progress.py can play it).
Stops once the remote run has finished and its final best.pt is here.

Run: .venv/bin/python scripts/sync_pod.py --host root@195.26.233.9 --port 46940 --run v1_piano
"""

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", default="22")
    ap.add_argument("--key", default=str(Path.home() / ".ssh/id_ed25519_runpod"))
    ap.add_argument("--remote", default="/root/OrchestraMaker")
    ap.add_argument("--run", default="v1_piano")
    ap.add_argument("--every", type=float, default=30)
    ap.add_argument("--ckpt-every", type=int, default=2500)
    args = ap.parse_args()

    ssh_opts = ["-i", args.key, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    remote_dir = f"{args.remote}/checkpoints/{args.run}"
    local = ROOT / "checkpoints" / args.run
    local.mkdir(parents=True, exist_ok=True)

    def scp(name, dest):
        cmd = ["scp", "-q", "-P", args.port, *ssh_opts, f"{args.host}:{remote_dir}/{name}", str(dest)]
        return subprocess.run(cmd, capture_output=True).returncode == 0

    def remote_best_step():
        code = f"import torch; print(torch.load('{remote_dir}/best.pt', map_location='cpu', weights_only=False)['step'])"
        cmd = ["ssh", "-p", args.port, *ssh_opts, args.host, f"cd {args.remote} && .venv/bin/python -c \"{code}\""]
        out = subprocess.run(cmd, capture_output=True, text=True)
        return int(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip().isdigit() else None

    # Checkpoints download in a background thread so the log keeps syncing meanwhile (~340 MB each).
    ckpt = {"step": -args.ckpt_every, "thread": None, "final": False}
    if (local / "best.pt").exists() and (local / "sync.json").exists():  # resuming a mirror
        ckpt["step"] = json.loads((local / "sync.json").read_text())["best_step"]

    def fetch_best():
        step = remote_best_step()
        if step is not None and step > ckpt["step"] and scp("best.pt", local / "best.pt.part"):
            os.replace(local / "best.pt.part", local / "best.pt")
            snapshots = local / "snapshots"  # keep every downloaded checkpoint, to revisit a step later
            snapshots.mkdir(exist_ok=True)
            os.link(local / "best.pt", snapshots / f"step_{step:05d}.pt")
            ckpt["step"] = step
            print(f"downloaded best.pt from step {step}", flush=True)

    print(f"mirroring {args.host}:{remote_dir} -> {local}", flush=True)
    while True:
        ok = scp("train.log", local / "train.log") and scp("config.json", local / "config.json")
        log = (local / "train.log").read_text() if (local / "train.log").exists() else ""
        finished = "done in" in log
        last_step = max([int(l.split()[1]) for l in log.splitlines() if l.startswith("step ")] or [0])
        busy = ckpt["thread"] is not None and ckpt["thread"].is_alive()
        if not busy and (last_step - ckpt["step"] >= args.ckpt_every or (finished and not ckpt["final"])):
            ckpt["final"] = finished
            ckpt["thread"] = threading.Thread(target=fetch_best, daemon=True)
            ckpt["thread"].start()
            busy = True
        done = ckpt["final"] and not busy
        if done:  # tell runpod_finish.sh the final checkpoint is safe here, so the pod can be removed
            subprocess.run(["ssh", "-p", args.port, *ssh_opts, args.host, f"touch {remote_dir}/fetched_final"],
                           capture_output=True)
        (local / "sync.json").write_text(json.dumps({
            "host": args.host, "synced_at": time.time() if ok else None,
            "best_step": max(ckpt["step"], 0), "downloading": busy, "final_done": done}))
        if done:
            print("run finished and final checkpoint is here", flush=True)
            return
        time.sleep(args.every)

if __name__ == "__main__":
    main()
