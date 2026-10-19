"""Pipeline step: link entities in articles and store mentions.

Scans every finished article from the last SCAN_WINDOW (the life of the text cache) that has not
been scanned by the current linker version: headline + body when we hold the text, otherwise
headline + feed summary, otherwise the headline alone. Changing the alias table or the rules
changes the version, so recent articles are rescanned automatically.

Usage:
    python -m nazariya.entities.run              # incremental
    python -m nazariya.entities.run --limit 500  # at most N articles this run
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, or_, select

from ..cluster import FINISHED
from ..db import (
    article_text, articles, entities, entity_scans, get_engine, insert_ignore, mentions, upsert,
)
from .linker import Linker

SCAN_WINDOW = timedelta(days=14)
BATCH = 500


def sync_entities(engine, linker: Linker) -> int:
    rows = [{"qid": e["qid"], "type": e["type"], "label_en": e["label_en"], "label_hi": e.get("label_hi"),
             "description": e.get("description"), "sitelinks": e.get("sitelinks")}
            for e in linker.entities.values()]
    with engine.begin() as conn:
        upsert(conn, entities, rows, ["qid"])
    return len(rows)


def run(limit: int | None = None) -> dict:
    engine = get_engine()
    linker = Linker()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    t0 = time.monotonic()
    n_entities = sync_entities(engine, linker)
    with engine.connect() as conn:
        todo = conn.execute(
            select(articles.c.id)
            .outerjoin(entity_scans, entity_scans.c.article_id == articles.c.id)
            .where(articles.c.status.in_(FINISHED), articles.c.published_at >= now - SCAN_WINDOW,
                   or_(entity_scans.c.article_id.is_(None), entity_scans.c.version != linker.version))
            .order_by(articles.c.published_at.desc())
            .limit(limit)
        ).scalars().all()

    scanned = linked = 0
    for start in range(0, len(todo), BATCH):
        ids = todo[start:start + BATCH]
        with engine.connect() as conn:
            rows = conn.execute(
                select(articles.c.id, articles.c.title, articles.c.summary, article_text.c.text)
                .outerjoin(article_text, article_text.c.article_id == articles.c.id)
                .where(articles.c.id.in_(ids))
            ).all()
        ment, scans = [], []
        for r in rows:
            body = r.text or r.summary or ""
            for qid, v in linker.link(r.title or "", body).items():
                ment.append({"article_id": r.id, "qid": qid, "n": v["n"], "in_title": v["in_title"]})
            scans.append({"article_id": r.id, "version": linker.version, "scanned_at": now,
                          "text_kind": "body" if r.text else ("summary" if r.summary else "headline")})
        with engine.begin() as conn:
            conn.execute(delete(mentions).where(mentions.c.article_id.in_(ids)))  # rescans replace
            insert_ignore(conn, mentions, ment, ["article_id", "qid"])
            upsert(conn, entity_scans, scans, ["article_id"])
        scanned += len(rows)
        linked += len(ment)
    return {"entities": n_entities, "linker_version": linker.version, "scanned": scanned,
            "mentions": linked, "seconds": round(time.monotonic() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    print(run(ap.parse_args().limit))


if __name__ == "__main__":
    main()
