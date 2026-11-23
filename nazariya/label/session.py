"""The human labelling session: yes/no checks of what the live pipeline decided.

Questions (headlines and links only, so the question file can be committed):
  pair      "Are these two articles about the same event?"
            - PAIRS_SAME pairs the clustering put in one event (different outlets; ~30% English-Hindi)
            - PAIRS_APART look-alike pairs it kept apart: different events within 48 hours whose
              articles name the same person, party or state
  link      "Does this headline refer to <name> (<description>)?" for LINKS random links the entity
            linker made in headlines

Answers give: clustering precision (share of 'same' calls a human agrees with), the miss rate among
look-alikes it kept apart, and the linker's precision in headlines.

Usage:
    python -m nazariya.label.session build      # labels/session_v2.jsonl from the current database
    python -m nazariya.label.session serve      # local page on 127.0.0.1; answers saved as you go
    python -m nazariya.label.session score      # reports/session_v2.md
"""

from __future__ import annotations

import argparse
import json
import math
import random
import threading
import webbrowser
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sqlalchemy import select

from ..config import ROOT
from ..db import article_events, articles, as_utc, entities, get_engine, mentions, outlets

QUESTIONS = ROOT / "labels" / "session_v2.jsonl"
ANSWERS = ROOT / "labels" / "session_v2.answers.jsonl"
REPORT = ROOT / "reports" / "session_v2.md"
STATIC = Path(__file__).parent / "static" / "session.html"
PAIRS_SAME, PAIRS_APART, LINKS = 40, 40, 50
RNG = random.Random(2026)


def _doc(r, names) -> dict:
    return {"id": r.id, "outlet": names.get(r.outlet_id, r.outlet_id), "lang": r.lang, "title": r.title, "url": r.url,
            "published_at": as_utc(r.published_at).isoformat() if r.published_at else None}


def build(engine=None) -> int:
    engine = engine or get_engine()
    with engine.connect() as conn:
        names = dict(conn.execute(select(outlets.c.id, outlets.c.name)).all())
        rows = conn.execute(
            select(articles.c.id, articles.c.outlet_id, articles.c.lang, articles.c.title, articles.c.url,
                   articles.c.published_at, article_events.c.event_id)
            .join(article_events, article_events.c.article_id == articles.c.id)
            .where(articles.c.title.is_not(None))
        ).all()
        ment = conn.execute(
            select(mentions.c.article_id, mentions.c.qid, mentions.c.in_title, entities.c.type, entities.c.label_en,
                   entities.c.description)
            .join(entities, entities.c.qid == mentions.c.qid)
            .where(entities.c.type.in_(["person", "party", "state"]))
        ).all()
    by_event, by_id = defaultdict(list), {}
    for r in rows:
        by_event[r.event_id].append(r)
        by_id[r.id] = r
    qs = []

    # 1. pairs the clustering joined: two articles from different outlets in one event
    multi = [rs for rs in by_event.values() if len({r.outlet_id for r in rs}) >= 3]
    RNG.shuffle(multi)
    for rs in multi:
        if sum(q["kind"] == "pair" and q["predicted"] for q in qs) >= PAIRS_SAME:
            break
        want_cross = RNG.random() < 0.3
        cands = [(a, b) for i, a in enumerate(rs) for b in rs[i + 1:] if a.outlet_id != b.outlet_id
                 and ((a.lang != b.lang) == want_cross)]
        if cands:
            a, b = RNG.choice(cands)
            qs.append({"kind": "pair", "predicted": True, "a": _doc(a, names), "b": _doc(b, names)})

    # 2. look-alikes kept apart: different events, within 48 h, naming the same (not too common) entity
    by_entity = defaultdict(set)
    for m in ment:
        if m.article_id in by_id and m.qid != "Q9357528":
            by_entity[m.qid].add(m.article_id)
    pool = [q for q, ids in by_entity.items() if 3 <= len(ids) <= 40]
    RNG.shuffle(pool)
    seen = set()
    for q in pool:
        if sum(x["kind"] == "pair" and not x["predicted"] for x in qs) >= PAIRS_APART:
            break
        ids = list(by_entity[q])
        RNG.shuffle(ids)
        for i, a_id in enumerate(ids):
            a = by_id[a_id]
            b = next((by_id[x] for x in ids[i + 1:] if by_id[x].event_id != a.event_id and by_id[x].outlet_id != a.outlet_id
                      and abs((as_utc(by_id[x].published_at) - as_utc(a.published_at)).total_seconds()) <= 48 * 3600), None)
            if b and (a.event_id, b.event_id) not in seen:
                seen.add((a.event_id, b.event_id))
                qs.append({"kind": "pair", "predicted": False, "a": _doc(a, names), "b": _doc(b, names)})
                break

    # 3. links made in headlines
    in_title = [m for m in ment if m.in_title and m.article_id in by_id]
    for m in RNG.sample(in_title, min(LINKS, len(in_title))):
        qs.append({"kind": "link", "predicted": True, "a": _doc(by_id[m.article_id], names),
                   "entity": {"qid": m.qid, "type": m.type, "label": m.label_en, "description": m.description}})

    RNG.shuffle(qs)
    for i, q in enumerate(qs, 1):
        q["qid"] = f"s2-{i:03d}"
    QUESTIONS.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs), encoding="utf-8")
    return len(qs)


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def serve(port: int = 8770, open_browser: bool = True) -> None:
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, STATIC.read_bytes(), "text/html")
            if self.path == "/api/session":
                answers = {a["qid"]: a["answer"] for a in _read(ANSWERS)}
                return self._send(200, json.dumps({"questions": _read(QUESTIONS), "answers": answers}, ensure_ascii=False))
            self._send(404, "{}")

        def do_POST(self):
            if self.path != "/api/answer":
                return self._send(404, "{}")
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if body.get("answer") not in ("yes", "no", "unsure") or not str(body.get("qid", "")).startswith("s2-"):
                return self._send(400, '{"error": "bad answer"}')
            rec = {"qid": body["qid"], "answer": body["answer"], "at": datetime.now(timezone.utc).isoformat()}
            with lock, ANSWERS.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            self._send(200, '{"ok": true}')

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    url = f"http://127.0.0.1:{port}/"
    print(f"labelling session at {url}  (Ctrl+C to stop; answers are saved as you go)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    srv.serve_forever()


def _wilson(k: int, n: int) -> str:
    if not n:
        return "n/a"
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{p:.3f} (95% CI {max(0, c - h):.3f} to {min(1, c + h):.3f}, n = {n})"


def score() -> str:
    qs = {q["qid"]: q for q in _read(QUESTIONS)}
    ans = {a["qid"]: a["answer"] for a in _read(ANSWERS)}  # latest answer wins
    got = [(qs[k], v) for k, v in ans.items() if k in qs and v != "unsure"]
    same = [v for q, v in got if q["kind"] == "pair" and q["predicted"]]
    apart = [v for q, v in got if q["kind"] == "pair" and not q["predicted"]]
    links = [v for q, v in got if q["kind"] == "link"]
    lines = ["# Human labelling session v2", "",
             f"{len(ans)} of {len(qs)} questions answered ({sum(v == 'unsure' for v in ans.values())} marked can't tell).", "",
             "| measure | value |", "|---|---|",
             f"| Clustering precision: 'same event' calls a human agrees with | {_wilson(sum(v == 'yes' for v in same), len(same))} |",
             f"| Look-alikes kept apart that were really one event (missed merges) | {_wilson(sum(v == 'yes' for v in apart), len(apart))} |",
             f"| Entity-link precision in headlines | {_wilson(sum(v == 'yes' for v in links), len(links))} |", "",
             "Pairs the clustering joined were sampled from events three or more outlets covered; look-alikes are "
             "different events within 48 hours naming the same person, party or state. These are checks of the "
             "production pipeline, not of the earlier LLM silver labels."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "serve", "score"])
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--no-open", action="store_true", help="do not open a browser window")
    a = ap.parse_args()
    if a.action == "build":
        print(f"{build()} questions -> {QUESTIONS}")
    elif a.action == "serve":
        serve(a.port, not a.no_open)
    else:
        print(score())


if __name__ == "__main__":
    main()
