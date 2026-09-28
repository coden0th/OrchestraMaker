"""Local web UI for listening to takes and watching which notes are played.

Run:  .venv/bin/python scripts/serve.py        then open http://127.0.0.1:8000
Serves only webui/ and outputs/, on localhost only.

A separate site for one model, e.g. v1.5 on another port:
      .venv/bin/python scripts/serve.py --port 8001 --title v1.5 --runs v1_5 \
          --takes "v1_5/*" "progress/v1_5*" "eval/*v1_5*" "aria_vs_v1_5*" "v1_5_*"
"""

import argparse
import hashlib
import json
import time
from fnmatch import fnmatch
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
SITE = {"title": "", "takes": [], "runs": [], "pool": None}  # set from the command line
RATINGS = OUTPUTS / "ratings.json"  # take url -> {"rating": 1-5, "comment", "time", "sha1" of the take file}


def list_takes():
    takes = []
    for path in OUTPUTS.rglob("*.json"):
        try:
            take = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue  # a file being written right now
        if "versions" not in take:
            continue
        group = str(path.parent.relative_to(OUTPUTS))
        if SITE["takes"] and not any(fnmatch(group, pat) for pat in SITE["takes"]):
            continue
        first = take["versions"][0]
        takes.append({
            "url": "/" + str(path.relative_to(ROOT)),
            "group": group,
            "name": take["name"],
            "meta": take["meta"],
            "problems": sum(len(t["problems"]) for t in first["tracks"]),
            "mtime": path.stat().st_mtime,
        })
    return sorted(takes, key=lambda t: (t["group"], t["name"]))


def parse_run(run_dir: Path) -> dict:
    """Training curve and status from a run's train.log (local, or mirrored by sync_pod.py)."""
    run = {"name": run_dir.name, "train": [], "val": [], "rate": None, "eta": None, "done": False, "params": None}
    step = 0
    for line in (run_dir / "train.log").read_text(errors="ignore").splitlines():
        parts = line.split()
        if line.startswith("step ") and len(parts) >= 10:
            step = int(parts[1])
            run["train"].append([step, float(parts[3])])
            run["rate"], run["eta"] = float(parts[6].rstrip("k")) * 1000, float(parts[9])
        elif line.startswith("  val "):
            values = dict(zip(parts[1:-2:2], map(float, parts[2:-2:2])))
            run["val"].append([step, values])
        elif line.startswith("model: "):
            run["params"] = parts[1]
        elif line.startswith("done in"):
            run["done"] = True
    config = run_dir / "config.json"
    run["max_steps"] = json.loads(config.read_text())["max_steps"] if config.exists() else None
    sync = run_dir / "sync.json"
    run["sync"] = json.loads(sync.read_text()) if sync.exists() else None
    run["updated"] = (run_dir / "train.log").stat().st_mtime
    return run


def list_runs():
    runs = []
    for log in sorted((ROOT / "checkpoints").glob("*/train.log")):
        if SITE["runs"] and log.parent.name not in SITE["runs"]:
            continue
        try:
            runs.append(parse_run(log.parent))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return sorted(runs, key=lambda r: -r["updated"])


def read_ratings() -> dict:
    return json.loads(RATINGS.read_text()) if RATINGS.exists() else {}


def rating_pool() -> list[str]:
    """Take urls for blind rating (scripts/build_rating_pool.py), if this site has a pool."""
    return json.loads(Path(SITE["pool"]).read_text())["takes"] if SITE["pool"] else []


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/webui/")
            self.end_headers()
        elif self.path.split("?")[0] in ("/api/takes", "/api/runs", "/api/site", "/api/ratings", "/api/pool"):
            route = self.path.split("?")[0]
            data = {"/api/takes": list_takes, "/api/runs": list_runs, "/api/site": lambda: SITE,
                    "/api/ratings": read_ratings, "/api/pool": rating_pool}[route]()
            self.send_json(data)
        elif self.path.startswith(("/webui/", "/outputs/")):
            super().do_GET()
        else:
            self.send_error(404)

    def do_POST(self):
        """POST /api/rate {"url": take url, "rating": 1-5 or null to clear, "comment": optional text}"""
        if self.path != "/api/rate":
            return self.send_error(404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            url, rating = req["url"], req.get("rating")
            if not url.startswith("/outputs/") or rating not in (None, 1, 2, 3, 4, 5):
                raise ValueError
        except (ValueError, KeyError, json.JSONDecodeError):
            return self.send_error(400)
        ratings = read_ratings()
        if rating is None:
            ratings.pop(url, None)
        else:
            take = (ROOT / url.lstrip("/")).resolve()
            if not take.is_relative_to(OUTPUTS.resolve()) or not take.is_file():
                return self.send_error(404)
            ratings[url] = {"rating": rating, "comment": str(req.get("comment", ""))[:500], "time": time.time(),
                            "sha1": hashlib.sha1(take.read_bytes()).hexdigest()}  # detects a re-rendered take
        tmp = RATINGS.with_suffix(".tmp")
        tmp.write_text(json.dumps(ratings, indent=1))
        tmp.replace(RATINGS)  # atomic: a crash never leaves half a ratings file
        self.send_json({"ok": True, "rated": len(ratings)})

    def send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")  # takes get re-rendered under the same name
        super().end_headers()

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--title", default="", help="shown next to OrchestraMaker, e.g. v1.5")
    ap.add_argument("--takes", nargs="*", default=[], help="only take groups matching these patterns")
    ap.add_argument("--runs", nargs="*", default=[], help="only these training runs")
    ap.add_argument("--pool", help="rating pool manifest (scripts/build_rating_pool.py) for blind rating")
    args = ap.parse_args()
    SITE.update(title=args.title, takes=args.takes, runs=args.runs, pool=args.pool)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(ROOT)))
    print(f"OrchestraMaker UI{' (' + args.title + ')' if args.title else ''}: http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
