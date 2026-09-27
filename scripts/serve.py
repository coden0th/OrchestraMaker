"""Local web UI for listening to takes and watching which notes are played.

Run:  .venv/bin/python scripts/serve.py        then open http://127.0.0.1:8000
Serves only webui/ and outputs/, on localhost only.
"""

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"


def list_takes():
    takes = []
    for path in OUTPUTS.rglob("*.json"):
        try:
            take = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue  # a file being written right now
        if "versions" not in take:
            continue
        first = take["versions"][0]
        takes.append({
            "url": "/" + str(path.relative_to(ROOT)),
            "group": str(path.parent.relative_to(OUTPUTS)),
            "name": take["name"],
            "meta": take["meta"],
            "problems": sum(len(t["problems"]) for t in first["tracks"]),
            "mtime": path.stat().st_mtime,
        })
    return sorted(takes, key=lambda t: (t["group"], t["name"]))


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/webui/")
            self.end_headers()
        elif self.path.split("?")[0] == "/api/takes":
            body = json.dumps(list_takes()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith(("/webui/", "/outputs/")):
            super().do_GET()
        else:
            self.send_error(404)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")  # takes get re-rendered under the same name
        super().end_headers()

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(ROOT)))
    print(f"OrchestraMaker UI: http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
