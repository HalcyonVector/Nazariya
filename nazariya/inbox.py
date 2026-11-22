"""Hand-off of home-PC articles to the cloud pipeline.

A few outlets block GitHub's datacenter IPs, so a scheduled task on a home PC ingests them into its
own local SQLite file. It then exports the recent ones (metadata, the short-lived text cache and,
when the PC has embedded them, their vectors) to a gzipped JSONL file, which travels encrypted as a
GitHub release asset. The cloud run imports it into the main database before clustering.

Article ids differ between the two databases, so rows are matched by URL. Links to other rows
(duplicate_of, syndicated_of) are dropped because those ids mean nothing on the other side.

Usage:
    python -m nazariya.inbox export data/home_inbox.jsonl.gz [--days 3]
    python -m nazariya.inbox import data/home_inbox.jsonl.gz
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.engine import Engine

from .db import (
    article_text, article_vectors, articles, as_utc, get_engine, insert_ignore, outlets, upsert,
)

DROP = {"id", "duplicate_of", "syndicated_of"}
DATES = {"published_at", "discovered_at", "extracted_at"}


def _enc(v):
    return v.isoformat() if isinstance(v, datetime) else v


def _dec(row: dict, keys) -> dict:
    return {k: (as_utc(datetime.fromisoformat(v)) if k in keys and v else v) for k, v in row.items()}


def export(engine: Engine, path: Path, days: float = 3, now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    n = 0
    with engine.connect() as conn, gzip.open(path, "wt", encoding="utf-8") as out:
        for o in conn.execute(select(outlets)).mappings():
            out.write(json.dumps({"outlet": dict(o)}, ensure_ascii=False) + "\n")
        rows = conn.execute(
            select(articles, article_text.c.text, article_text.c.expires_at,
                   article_vectors.c.model, article_vectors.c.vec, article_vectors.c.text_kind)
            .outerjoin(article_text, article_text.c.article_id == articles.c.id)
            .outerjoin(article_vectors, article_vectors.c.article_id == articles.c.id)
            .where(articles.c.discovered_at >= since)
        ).mappings()
        for r in rows:
            art = {k: _enc(r[k]) for k in articles.c.keys() if k not in DROP}
            rec = {"article": art}
            if r["text"] is not None:
                rec["text"] = {"text": r["text"], "expires_at": _enc(r["expires_at"])}
            if r["vec"] is not None:
                rec["vector"] = {"model": r["model"], "text_kind": r["text_kind"],
                                 "vec": base64.b64encode(r["vec"]).decode()}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def import_(engine: Engine, path: Path, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    outs, recs = [], []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            (outs if "outlet" in rec else recs).append(rec)
    with engine.begin() as conn:
        upsert(conn, outlets, [r["outlet"] for r in outs], ["id"])
        # an article already here keeps its row; the newer home copy only fills in what is missing
        insert_ignore(conn, articles, [_dec(r["article"], DATES) for r in recs], ["url"])
        urls = [r["article"]["url"] for r in recs]
        ids = {}
        for i in range(0, len(urls), 500):
            ids.update(conn.execute(select(articles.c.url, articles.c.id)
                                    .where(articles.c.url.in_(urls[i:i + 500]))).all())
        texts, vecs = [], []
        for r in recs:
            aid = ids.get(r["article"]["url"])
            if aid is None:
                continue
            if "text" in r:
                texts.append({"article_id": aid, **_dec(r["text"], {"expires_at"})})
            if "vector" in r:
                v = r["vector"]
                vecs.append({"article_id": aid, "model": v["model"], "text_kind": v["text_kind"],
                             "vec": base64.b64decode(v["vec"]), "created_at": now})
        insert_ignore(conn, article_text, texts, ["article_id"])
        insert_ignore(conn, article_vectors, vecs, ["article_id"])
    return {"articles": len(recs), "texts": len(texts), "vectors": len(vecs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["export", "import"])
    ap.add_argument("path", type=Path)
    ap.add_argument("--days", type=float, default=3)
    a = ap.parse_args()
    engine = get_engine()
    if a.action == "export":
        print({"exported": export(engine, a.path, a.days)})
    elif a.path.exists():
        print(import_(engine, a.path))
    else:
        print(f"no inbox at {a.path}; nothing to import")


if __name__ == "__main__":
    main()
