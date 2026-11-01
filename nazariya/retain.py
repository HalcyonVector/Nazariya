"""Retention: keep the database inside the free tier (Neon: 0.5 GB) by deleting what nothing reads.

Measured on 2026-11-01 (355 MB at ~12k articles/day): article_vectors 132 MB, article_text 79 MB,
articles 73 MB, events 39 MB, article_frames 18 MB, mentions 10 MB. Policy:

  articles older than RETAIN_DAYS        deleted, with everything hanging off them (text, vectors,
                                         mentions, scans, frames, stances, event membership)
  unpinned article text older than TEXT_DAYS    deleted (the text is only needed right after extraction)
  events not seen for RETAIN_DAYS        deleted
  vectors of articles already assigned AND framed (older than VECTOR_MIN_AGE_H hours), except the
                                         representative of an open event      deleted
  fetch_runs older than FETCH_RUN_DAYS, runs older than RUN_DAYS   deleted

Pinned articles are never deleted: those whose cached text carries the PINNED_UNTIL expiry (the
labelled event pairs and the framing/stance training sample), because the label files refer to them
by id and training needs their text. Seven days of raw rows is enough for the weekly reports; older
history is meant to live in small weekly aggregates, not raw rows.

Usage:
    python -m nazariya.retain --dry-run     # count only
    python -m nazariya.retain
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.engine import Engine

from .db import (
    PINNED_UNTIL, article_events, article_frames, article_text, article_vectors, articles, events, fetch_runs,
    get_engine, runs,
)

RETAIN_DAYS = 8
TEXT_DAYS = 3
FETCH_RUN_DAYS = 14
VECTOR_MIN_AGE_H = 6
RUN_DAYS = 60
BATCH = 4000


def _pinned():
    return select(article_text.c.article_id).where(article_text.c.expires_at >= PINNED_UNTIL)


def _delete_in_batches(engine: Engine, table, id_col, where, dry_run: bool) -> int:
    """Delete rows matching `where` in batches of BATCH ids (one transaction each), returning the count."""
    if dry_run:
        with engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(table).where(where)).scalar() or 0
    total = 0
    while True:
        with engine.begin() as conn:
            ids = conn.execute(select(id_col).where(where).limit(BATCH)).scalars().all()
            if not ids:
                return total
            total += conn.execute(delete(table).where(id_col.in_(ids))).rowcount


def run(dry_run: bool = False, now: datetime | None = None, engine: Engine | None = None) -> dict:
    engine = engine or get_engine()
    now = now or datetime.now(timezone.utc)
    t0 = time.monotonic()
    out = {}
    out["articles"] = _delete_in_batches(
        engine, articles, articles.c.id,
        (articles.c.published_at < now - timedelta(days=RETAIN_DAYS)) & articles.c.id.not_in(_pinned()), dry_run)
    out["old_text"] = _delete_in_batches(
        engine, article_text, article_text.c.article_id,
        (article_text.c.expires_at < PINNED_UNTIL)
        & article_text.c.article_id.in_(select(articles.c.id).where(articles.c.published_at < now - timedelta(days=TEXT_DAYS))),
        dry_run)
    open_reps = select(events.c.rep_article_id).where(events.c.centroid.is_not(None),
                                                      events.c.rep_article_id.is_not(None))
    out["used_vectors"] = _delete_in_batches(
        engine, article_vectors, article_vectors.c.article_id,
        (article_vectors.c.created_at < now - timedelta(hours=VECTOR_MIN_AGE_H))
        & article_vectors.c.article_id.in_(select(article_events.c.article_id))
        & article_vectors.c.article_id.in_(select(article_frames.c.article_id))
        & article_vectors.c.article_id.not_in(open_reps), dry_run)
    out["events"] = _delete_in_batches(
        engine, events, events.c.id, events.c.last_seen < now - timedelta(days=RETAIN_DAYS), dry_run)
    out["fetch_runs"] = _delete_in_batches(
        engine, fetch_runs, fetch_runs.c.id, fetch_runs.c.fetched_at < now - timedelta(days=FETCH_RUN_DAYS), dry_run)
    out["runs"] = _delete_in_batches(
        engine, runs, runs.c.id, runs.c.started_at < now - timedelta(days=RUN_DAYS), dry_run)
    with engine.connect() as conn:
        out["pinned_articles_kept"] = conn.execute(select(func.count()).select_from(article_text).where(
            article_text.c.expires_at >= PINNED_UNTIL)).scalar()
    out["dry_run"] = dry_run
    out["seconds"] = round(time.monotonic() - t0, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="only count what would be deleted")
    print(run(ap.parse_args().dry_run))


if __name__ == "__main__":
    main()
