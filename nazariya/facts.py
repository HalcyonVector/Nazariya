"""Condense recent events into the long-lived fact tables the models read.

Raw articles are deleted after RETAIN_DAYS (storage), but the models need months of history. So
every run summarises each recently active event into:

  event_facts  one row per event: size, timing, languages, states named, people named, death toll,
               wire copy, government release, Delhi dateline, frame, English-Hindi drift
  coverage     one row per (event, outlet): when the outlet first covered it, how many articles,
               how many were agency copy
  copies       one row per article whose text matches an earlier article of another outlet

Rows are recomputed while an event's articles still exist and then stay frozen. Nothing here holds
article text.

Usage:
    python -m nazariya.facts [--days 4]
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.engine import Engine

from .db import (
    article_events, article_frames, article_text, article_vectors, articles, as_utc, copies, coverage,
    entities, event_facts, events, get_engine, insert_ignore, mentions, outlets, upsert,
)

REFRESH_DAYS = 4  # events active this recently are recomputed; older rows are final
DATELINE_RE = re.compile(r"^\W{0,3}(new delhi|नई दिल्ली)\b", re.I)
_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)"
TOLL_RES = [
    re.compile(_NUM + r"\s+(?:people\s+|persons\s+|workers\s+|pilgrims\s+|devotees\s+)?"
               r"(?:dead|killed|die|died|dies|deaths?|feared dead|lose their lives|lost their lives)\b", re.I),
    re.compile(r"(?:death toll|toll)\s+(?:rises|climbs|mounts|reaches|touches|crosses|at|to)\s+(?:to\s+)?" + _NUM, re.I),
    re.compile(_NUM + r"\s+(?:लोगों|श्रद्धालुओं|मजदूरों|यात्रियों)?\s*की\s+मौत"),
    re.compile(r"मृतकों\s+की\s+संख्या\s+" + _NUM),
]
CHUNK = 500


def death_toll(title: str) -> int | None:
    best = None
    for rx in TOLL_RES:
        for m in rx.finditer(title or ""):
            n = int(m.group(1).replace(",", ""))
            if 0 < n < 100000:
                best = n if best is None else max(best, n)
    return best


DELHI = "Q9357528"


def _qids(aid: int, ment: dict, lede: dict) -> set[str]:
    """Entities named in an article. A single Delhi mention in an article that opens with a
    "New Delhi:" dateline is the dateline itself, not the story's place, and is dropped."""
    dateline = bool(DATELINE_RE.match(lede.get(aid, "")))
    return {q for q, n in ment.get(aid, []) if not (q == DELHI and dateline and n <= 1)}


def _chunks(xs: list, n: int = CHUNK):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def _vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float16).astype(np.float32)


def run(engine: Engine | None = None, days: float = REFRESH_DAYS, now: datetime | None = None) -> dict:
    engine = engine or get_engine()
    now = now or datetime.now(timezone.utc)
    t0 = time.monotonic()
    with engine.connect() as conn:
        evs = conn.execute(select(events).where(events.c.last_seen >= now - timedelta(days=days))).mappings().all()
        ev_ids = [e["id"] for e in evs]
        out_meta = {r.id: r for r in conn.execute(select(outlets.c.id, outlets.c.lang, outlets.c.role))}
        etype = dict(conn.execute(select(entities.c.qid, entities.c.type)).all())
        members: dict[int, list] = defaultdict(list)
        art_ids = []
        for chunk in _chunks(ev_ids):
            for r in conn.execute(
                select(article_events.c.event_id, articles.c.id, articles.c.outlet_id, articles.c.title,
                       articles.c.lang, articles.c.published_at, articles.c.date_fixed, articles.c.agency,
                       articles.c.syndicated_of, articles.c.status)
                .join(articles, articles.c.id == article_events.c.article_id)
                .where(article_events.c.event_id.in_(chunk))
            ):
                members[r.event_id].append(r)
                art_ids.append(r.id)
        ment: dict[int, list[tuple[str, int]]] = defaultdict(list)
        lede: dict[int, str] = {}
        frame: dict[int, str] = {}
        vecs: dict[int, bytes] = {}
        for chunk in _chunks(art_ids):
            for aid, qid, n in conn.execute(select(mentions.c.article_id, mentions.c.qid, mentions.c.n)
                                            .where(mentions.c.article_id.in_(chunk))):
                ment[aid].append((qid, n))
            for aid, txt in conn.execute(select(article_text.c.article_id, article_text.c.text)
                                         .where(article_text.c.article_id.in_(chunk))):
                lede[aid] = txt[:80]
            frame.update(conn.execute(select(article_frames.c.article_id, article_frames.c.primary_frame)
                                      .where(article_frames.c.article_id.in_(chunk))).all())
            vecs.update(conn.execute(select(article_vectors.c.article_id, article_vectors.c.vec)
                                     .where(article_vectors.c.article_id.in_(chunk))).all())
        old_drift = {}
        for chunk in _chunks(ev_ids):
            old_drift.update({r.event_id: (r.lang_drift, r.lang_drift_n or 0) for r in conn.execute(
                select(event_facts.c.event_id, event_facts.c.lang_drift, event_facts.c.lang_drift_n)
                .where(event_facts.c.event_id.in_(chunk)))})

    fact_rows, cov_rows = [], []
    for e in evs:
        ms = [m for m in members.get(e["id"], []) if m.status != "duplicate"]
        if not ms:
            continue
        dated = [m for m in ms if not m.date_fixed and m.published_at is not None] or ms
        first = min(dated, key=lambda m: as_utc(m.published_at))
        qids = set().union(*(_qids(m.id, ment, lede) for m in ms))
        fq = _qids(first.id, ment, lede)
        ftoll = death_toll(first.title)
        by_outlet: dict[str, list] = defaultdict(list)
        for m in ms:
            by_outlet[m.outlet_id].append(m)
        langs = Counter((out_meta[o].lang if o in out_meta else None) or "?" for o in by_outlet)
        frames = Counter(frame[m.id] for m in ms if m.id in frame)
        drift, drift_n = old_drift.get(e["id"], (None, 0))
        en = [_vec(vecs[m.id]) for m in ms if m.id in vecs and (m.lang or "") == "en"]
        hi = [_vec(vecs[m.id]) for m in ms if m.id in vecs and (m.lang or "") == "hi"]
        if en and hi and len(en) + len(hi) >= drift_n:
            a, b = np.mean(en, axis=0), np.mean(hi, axis=0)
            drift = float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            drift_n = len(en) + len(hi)
        fact_rows.append({
            "event_id": e["id"], "first_seen": as_utc(e["first_seen"]), "last_seen": as_utc(e["last_seen"]),
            "n_articles": len(ms), "n_outlets": len(by_outlet), "title": e["title"],
            "first_outlet": first.outlet_id, "first_lang": first.lang,
            "langs": dict(langs), "states": sorted(q for q in qids if etype.get(q) == "state"),
            "n_people": sum(etype.get(q) in ("person", "foreign_person") for q in qids),
            "n_politicians": sum(etype.get(q) == "person" for q in qids),
            "toll": max((t for m in ms if (t := death_toll(m.title)) is not None), default=None),
            "wire": any(m.agency for m in ms) or "ani" in by_outlet,
            "govt": "pib" in by_outlet,
            "delhi_dateline": any(DATELINE_RE.match(lede.get(m.id, "")) for m in ms),
            "frame": frames.most_common(1)[0][0] if frames else None,
            "first_report": {
                "toll": ftoll, "politicians": sum(etype.get(q) == "person" for q in fq),
                "people": sum(etype.get(q) in ("person", "foreign_person") for q in fq),
                "states": sorted(q for q in fq if etype.get(q) == "state"),
                "dateline": bool(DATELINE_RE.match(lede.get(first.id, ""))),
                "wire": bool(first.agency) or first.outlet_id == "ani", "govt": first.outlet_id == "pib",
                "frame": frame.get(first.id), "lang": first.lang, "outlet": first.outlet_id,
                "text": first.id in lede,
            },
            "lang_drift": drift, "lang_drift_n": drift_n, "updated_at": now,
        })
        for o, oms in by_outlet.items():
            own = [as_utc(m.published_at) for m in oms
                   if not m.date_fixed and m.published_at and not m.agency and not m.syndicated_of]
            cov_rows.append({
                "event_id": e["id"], "outlet_id": o,
                "first_at": min(as_utc(m.published_at) or now for m in oms),
                "first_own_at": min(own) if own else None,
                "n_articles": len(oms), "n_wire": sum(bool(m.agency or m.syndicated_of) for m in oms),
                "n_text": sum(m.status == "extracted" for m in oms),
            })

    with engine.begin() as conn:
        upsert(conn, event_facts, fact_rows, ["event_id"])
        done = [r["event_id"] for r in fact_rows]
        for chunk in _chunks(done):
            conn.execute(delete(coverage).where(coverage.c.event_id.in_(chunk)))
        insert_ignore(conn, coverage, cov_rows, ["event_id", "outlet_id"])
        n_copies = _copies(conn, now - timedelta(days=days))
    return {"events": len(fact_rows), "coverage_rows": len(cov_rows), "copies": n_copies,
            "seconds": round(time.monotonic() - t0, 1)}


def _copies(conn, since: datetime) -> int:
    src = articles.alias("src")
    rows = conn.execute(
        select(articles.c.id, articles.c.outlet_id, src.c.outlet_id.label("source"), articles.c.agency,
               articles.c.published_at)
        .join(src, src.c.id == articles.c.syndicated_of)
        .where(articles.c.syndicated_of.is_not(None), articles.c.discovered_at >= since,
               src.c.outlet_id != articles.c.outlet_id)
    ).all()
    insert_ignore(conn, copies, [{"article_id": r.id, "outlet_id": r.outlet_id, "source_outlet_id": r.source,
                                  "agency": r.agency, "published_at": as_utc(r.published_at)} for r in rows],
                  ["article_id"])
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=REFRESH_DAYS)
    print(run(days=ap.parse_args().days))


if __name__ == "__main__":
    main()
