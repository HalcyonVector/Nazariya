"""Local labelling page for event pairs. Binds to 127.0.0.1 only.

Usage:
    python -m nazariya.label.serve                       # labels/event_pairs_v1.jsonl
    python -m nazariya.label.serve --pairs labels/event_pairs_v1.jsonl --port 8765
    python -m nazariya.label.serve --all                 # ignore the review queue, label everything blind

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

from ..config import ROOT
from .docs import load_docs, read_jsonl, sibling

STATIC = Path(__file__).parent / "static"
LABELS = {"same_event", "same_story", "unrelated", "unsure"}


class Store:
    """Review mode (default once labels/<name>.review.json exists): walk the review queue and show
    the LLM's suggestion, except for 'blind' pairs. Every label records whether a suggestion was
    shown, so agreement can be measured separately on blind pairs (no anchoring)."""

    def __init__(self, pairs_path: Path, review: bool):
        self.pairs = {p["pair_id"]: p for p in read_jsonl(pairs_path)}
        self.path = sibling(pairs_path, "labels.jsonl")
        self.labels = {r["pair_id"]: r for r in read_jsonl(self.path)}
        self.llm = {r["pair_id"]: r for r in read_jsonl(sibling(pairs_path, "llm.jsonl"))}
        review_path = sibling(pairs_path, "review.json")
        self.review = review and review_path.exists()
        if self.review:
            spec = json.loads(review_path.read_text(encoding="utf-8"))
            self.order, self.blind = spec["queue"], set(spec["blind"])
        else:
            self.order, self.blind = list(self.pairs), set(self.pairs)
        self.history: list[str] = []
        self.lock = threading.Lock()
        needed = [self.pairs[pid] for pid in self.order]
        self.docs = load_docs({p["a_id"] for p in needed} | {p["b_id"] for p in needed})

    def save(self) -> None:
        ordered = [self.labels[pid] for pid in self.pairs if pid in self.labels]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered), encoding="utf-8")
        tmp.replace(self.path)

    def suggestion(self, pair_id: str) -> dict | None:
        if pair_id in self.blind or pair_id not in self.llm:
            return None
        r = self.llm[pair_id]
        return {"label": r["label"], "confidence": r["confidence"], "reason": r["reason"]}

    def state(self) -> dict:
        nxt = next((pid for pid in self.order if pid not in self.labels), None)
        item = None
        if nxt:
            p = self.pairs[nxt]
            item = {"pair_id": nxt, "a": self.docs.get(p["a_id"]), "b": self.docs.get(p["b_id"]),
                    "suggestion": self.suggestion(nxt)}
        return {"done": sum(pid in self.labels for pid in self.order), "total": len(self.order),
                "item": item, "can_undo": bool(self.history), "review": self.review}

    def label(self, pair_id: str, label: str) -> None:
        with self.lock:
            llm = self.llm.get(pair_id)
            self.labels[pair_id] = {
                "pair_id": pair_id, "label": label,
                "shown_suggestion": self.suggestion(pair_id) is not None,
                "llm_label": llm["label"] if llm else None,
                "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
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
    ap.add_argument("--all", action="store_true", help="label every pair blind, ignoring the review queue")
    args = ap.parse_args()
    store = Store(Path(args.pairs), review=not args.all)
    missing = sum(1 for d in store.docs.values() if not d["excerpt"])
    mode = f"review queue of {len(store.order)} ({len(store.blind & set(store.order))} blind)" if store.review else "all pairs, blind"
    print(f"{mode}; {len(store.labels)} labelled so far; {missing} articles without text")
    print(f"labels → {store.path}")
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
