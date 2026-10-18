"""Local labelling page for event pairs. Binds to 127.0.0.1 only.

Usage:
    python -m nazariya.label.serve                       # labels/event_pairs_v1.jsonl
    python -m nazariya.label.serve --pairs labels/event_pairs_v1.jsonl --port 8765

Labels go to labels/<pairs-name>.labels.jsonl (one line per pair, latest label wins), which is
committed. Article text is read from the database at startup and is never written to disk.
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sqlalchemy import select

from ..config import ROOT
from ..db import article_text, articles, as_utc, get_engine, outlets

STATIC = Path(__file__).parent / "static"
LABELS = {"same_event", "same_story", "unrelated", "unsure"}
EXCERPT_CHARS = 1500


class Store:
    def __init__(self, pairs_path: Path):
        self.pairs = [json.loads(l) for l in pairs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.path = pairs_path.with_name(pairs_path.stem + ".labels.jsonl")
        self.labels: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.labels[rec["pair_id"]] = rec
        self.history: list[str] = []
        self.lock = threading.Lock()
        self.docs = self._load_docs()

    def _load_docs(self) -> dict[int, dict]:
        ids = {p["a_id"] for p in self.pairs} | {p["b_id"] for p in self.pairs}
        with get_engine().connect() as conn:
            rows = conn.execute(
                select(articles.c.id, articles.c.title, articles.c.url, articles.c.lang,
                       articles.c.published_at, outlets.c.name, article_text.c.text)
                .join(outlets, outlets.c.id == articles.c.outlet_id)
                .outerjoin(article_text, article_text.c.article_id == articles.c.id)
                .where(articles.c.id.in_(ids))
            ).all()
        return {
            r.id: {"title": r.title, "url": r.url, "lang": r.lang, "outlet": r.name,
                   "published": as_utc(r.published_at).strftime("%d %b %Y, %H:%M UTC"),
                   "excerpt": (r.text or "")[:EXCERPT_CHARS]}
            for r in rows
        }

    def save(self) -> None:
        ordered = [self.labels[p["pair_id"]] for p in self.pairs if p["pair_id"] in self.labels]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered), encoding="utf-8")
        tmp.replace(self.path)

    def state(self) -> dict:
        nxt = next((p for p in self.pairs if p["pair_id"] not in self.labels), None)
        item = None
        if nxt:
            item = {"pair_id": nxt["pair_id"], "a": self.docs.get(nxt["a_id"]), "b": self.docs.get(nxt["b_id"])}
        return {"done": len(self.labels), "total": len(self.pairs), "item": item,
                "can_undo": bool(self.history)}

    def label(self, pair_id: str, label: str) -> None:
        with self.lock:
            self.labels[pair_id] = {"pair_id": pair_id, "label": label,
                                    "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            self.history.append(pair_id)
            self.save()

    def undo(self) -> None:
        with self.lock:
            if self.history:
                self.labels.pop(self.history.pop(), None)
                self.save()


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def json(self, obj) -> None:
            self.send(200, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            if self.path == "/":
                self.send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self.json(store.state())
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/label" and body.get("label") in LABELS and body.get("pair_id"):
                store.label(body["pair_id"], body["label"])
            elif self.path == "/api/undo":
                store.undo()
            else:
                self.send(400, b"bad request", "text/plain")
                return
            self.json(store.state())

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(ROOT / "labels" / "event_pairs_v1.jsonl"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    store = Store(Path(args.pairs))
    missing = sum(1 for d in store.docs.values() if not d["excerpt"])
    print(f"{len(store.pairs)} pairs, {len(store.labels)} already labelled, {missing} articles without text")
    print(f"labels → {store.path.relative_to(ROOT)}")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"open {url}  (Ctrl+C to stop; progress is saved after every label)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
