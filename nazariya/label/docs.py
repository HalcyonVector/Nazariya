"""Shared helpers for labelling: pair files and article lookups (text read from the DB at runtime)."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

from ..db import article_text, articles, as_utc, get_engine, outlets

EXCERPT_CHARS = 1500


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def sibling(pairs_path: Path, suffix: str) -> Path:
    """labels/event_pairs_v1.jsonl → labels/event_pairs_v1.<suffix>"""
    return pairs_path.with_name(f"{pairs_path.stem}.{suffix}")


def load_docs(ids: set[int]) -> dict[int, dict]:
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
