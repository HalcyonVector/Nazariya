"""Database schema (SQLAlchemy Core). SQLite locally, Postgres (Neon/Supabase) via DATABASE_URL.

Copyright stance: `articles` holds metadata and derived values only. Extracted text goes to
`article_text`, a short-lived processing cache (see TEXT_TTL in ingest.py) that is purged
on every run and never exported or published.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import (
    JSON, BigInteger, Boolean, Column, DateTime, ForeignKey, Index, Integer, MetaData,
    String, Table, Text, create_engine, select,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import Connection, Engine

from .config import ROOT, load_dotenv

metadata = MetaData()
BATCH_ROWS = 200

outlets = Table(
    "outlets", metadata,
    Column("id", String(40), primary_key=True),
    Column("name", String(120), nullable=False),
    Column("lang", String(8)),
    Column("region", String(20)),
    Column("role", String(20)),
    Column("homepage", Text),
)

articles = Table(
    "articles", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("outlet_id", String(40), ForeignKey("outlets.id"), nullable=False),
    Column("url", Text, nullable=False, unique=True),  # canonical URL
    Column("title", Text, nullable=False),
    Column("summary", Text),  # publisher's own feed description, truncated
    Column("lang", String(8)),
    Column("section", String(80)),
    Column("published_at", DateTime(timezone=True)),
    Column("date_fixed", Boolean, nullable=False, default=False),
    Column("discovered_at", DateTime(timezone=True), nullable=False),
    Column("source_kind", String(10)),
    # pending → extracted | headline_only | duplicate | failed | skipped (budget/window)
    Column("status", String(16), nullable=False, default="pending"),
    Column("attempts", Integer, nullable=False, default=0),
    Column("access", String(16)),  # full | lede_only | headline_only
    Column("text_len", Integer),
    Column("simhash", BigInteger),  # signed two's-complement form of the 64-bit hash
    Column("agency", String(10)),  # PTI | ANI | IANS | UNI | Reuters | AFP | wire
    Column("duplicate_of", Integer),  # same outlet, near-identical text (re-published URL)
    Column("syndicated_of", Integer),  # other outlet, near-identical text (wire copy)
    Column("error", Text),
    Column("extracted_at", DateTime(timezone=True)),
)
Index("ix_articles_outlet_published", articles.c.outlet_id, articles.c.published_at)
Index("ix_articles_status", articles.c.status)
Index("ix_articles_discovered", articles.c.discovered_at)

article_text = Table(
    "article_text", metadata,
    Column("article_id", Integer, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True),
    Column("text", Text, nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
)

runs = Table(
    "runs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
    Column("stats", JSON),
)

fetch_runs = Table(
    "fetch_runs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Integer, ForeignKey("runs.id"), nullable=False),
    Column("outlet_id", String(40), ForeignKey("outlets.id"), nullable=False),
    Column("source_url", Text, nullable=False),
    Column("kind", String(10)),
    Column("position", Integer),  # 0 = primary source, >0 = fallback used
    Column("ok", Boolean, nullable=False),
    Column("accepted", Boolean, nullable=False),  # items from this source were ingested
    Column("http_status", Integer),
    Column("error", Text),
    Column("items_seen", Integer, nullable=False, default=0),
    Column("items_in_scope", Integer, nullable=False, default=0),
    Column("items_filtered", Integer, nullable=False, default=0),
    Column("items_new", Integer, nullable=False, default=0),
    Column("newest_item_at", DateTime(timezone=True)),
    Column("duration_ms", Integer),
    Column("fetched_at", DateTime(timezone=True), nullable=False),
)
Index("ix_fetch_runs_outlet_time", fetch_runs.c.outlet_id, fetch_runs.c.fetched_at)


def database_url() -> str:
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        (ROOT / "data").mkdir(exist_ok=True)
        return f"sqlite:///{ROOT / 'data' / 'nazariya.db'}"
    # Neon/Supabase hand out postgres:// URLs; SQLAlchemy wants an explicit driver
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def get_engine(url: str | None = None) -> Engine:
    engine = create_engine(url or database_url(), future=True)
    metadata.create_all(engine)
    return engine


def insert_ignore(conn: Connection, table: Table, rows: list[dict], conflict: list[str]) -> None:
    """INSERT ... ON CONFLICT DO NOTHING for SQLite and Postgres."""
    if not rows:
        return
    dialect = postgresql if conn.dialect.name == "postgresql" else sqlite
    # multi-row VALUES; batch to stay under SQLite's bound-variable limit
    for i in range(0, len(rows), BATCH_ROWS):
        conn.execute(dialect.insert(table).values(rows[i:i + BATCH_ROWS]).on_conflict_do_nothing(index_elements=conflict))


def upsert(conn: Connection, table: Table, rows: list[dict], key: list[str]) -> None:
    if not rows:
        return
    dialect = postgresql if conn.dialect.name == "postgresql" else sqlite
    for i in range(0, len(rows), BATCH_ROWS):
        stmt = dialect.insert(table).values(rows[i:i + BATCH_ROWS])
        cols = {c: stmt.excluded[c] for c in rows[0] if c not in key}
        conn.execute(stmt.on_conflict_do_update(index_elements=key, set_=cols))


def existing_urls(conn: Connection, urls: list[str]) -> set[str]:
    found: set[str] = set()
    for i in range(0, len(urls), 500):
        chunk = urls[i:i + 500]
        found.update(conn.execute(select(articles.c.url).where(articles.c.url.in_(chunk))).scalars())
    return found


def as_utc(dt: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; everything we store is UTC."""
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
