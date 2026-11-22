"""Database schema (SQLAlchemy Core). SQLite locally, Postgres (Neon/Supabase) via DATABASE_URL.

Copyright stance: `articles` holds metadata and derived values only. Extracted text goes to
`article_text`, a short-lived processing cache (see TEXT_TTL in ingest.py) that is purged
on every run and never exported or published.
"""

from __future__ import annotations

import atexit
import os
import sys
import time
from datetime import datetime, timezone

from sqlalchemy import (
    JSON, BigInteger, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, LargeBinary,
    MetaData, String, Table, Text, create_engine, select, text,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy import event
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
    # ids are never reused after retention deletes rows: the long-lived fact tables refer to them
    sqlite_autoincrement=True,
)
Index("ix_articles_outlet_published", articles.c.outlet_id, articles.c.published_at)
Index("ix_articles_status", articles.c.status)
Index("ix_articles_discovered", articles.c.discovered_at)

# Texts of articles in labelled gold sets are kept for reproducible evaluation (still never
# exported or published); everything else expires after the ingest TEXT_TTL.
PINNED_UNTIL = datetime(9999, 12, 31, tzinfo=timezone.utc)

article_text = Table(
    "article_text", metadata,
    Column("article_id", Integer, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True),
    Column("text", Text, nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
)

# ---------- event clustering ----------

# Embeddings for recent articles only (purged after VECTOR_TTL in cluster.py; ~2 KB each).
article_vectors = Table(
    "article_vectors", metadata,
    Column("article_id", Integer, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True),
    Column("model", String(40), nullable=False),
    Column("vec", LargeBinary, nullable=False),  # float16, L2-normalised
    Column("text_kind", String(10), nullable=False),  # lede (headline + body start) | headline
    Column("created_at", DateTime(timezone=True), nullable=False),
)

events = Table(
    "events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("first_seen", DateTime(timezone=True), nullable=False),  # earliest member published_at
    Column("last_seen", DateTime(timezone=True), nullable=False),  # latest member published_at
    Column("n_articles", Integer, nullable=False),
    Column("n_outlets", Integer, nullable=False),
    Column("title", Text),  # member headline closest to the centroid
    Column("rep_article_id", Integer),
    Column("centroid", LargeBinary),  # float32 sum of member vectors; NULL once the event closes
    Column("model", String(40), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    sqlite_autoincrement=True,
)
Index("ix_events_last_seen", events.c.last_seen)

article_events = Table(
    "article_events", metadata,
    Column("article_id", Integer, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True),
    Column("event_id", Integer, ForeignKey("events.id", ondelete="CASCADE"), nullable=False),
    Column("similarity", Float),  # to the event centroid at assignment time
    Column("method", String(10), nullable=False),  # online | merge
    Column("assigned_at", DateTime(timezone=True), nullable=False),
)
Index("ix_article_events_event", article_events.c.event_id)

# ---------- entities ----------

entities = Table(
    "entities", metadata,
    Column("qid", String(20), primary_key=True),  # Wikidata id
    Column("type", String(20), nullable=False),  # person | party | state | institution | ...
    Column("label_en", Text, nullable=False),
    Column("label_hi", Text),
    Column("description", Text),
    Column("sitelinks", Integer),
)

mentions = Table(
    "mentions", metadata,
    Column("article_id", Integer, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True),
    Column("qid", String(20), ForeignKey("entities.qid"), primary_key=True),
    Column("n", Integer, nullable=False),  # mentions in headline + body
    Column("in_title", Boolean, nullable=False),
)
Index("ix_mentions_qid", mentions.c.qid)

entity_scans = Table(
    "entity_scans", metadata,
    Column("article_id", Integer, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True),
    Column("version", String(12), nullable=False),  # linker version (hash of alias table + rules)
    Column("text_kind", String(10), nullable=False),  # body | summary | headline
    Column("scanned_at", DateTime(timezone=True), nullable=False),
)

# ---------- framing & stance (model predictions) ----------

article_frames = Table(
    "article_frames", metadata,
    Column("article_id", Integer, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True),
    Column("primary_frame", String(30), nullable=False),
    Column("scores", JSON, nullable=False),  # {frame: probability}
    Column("model", String(60), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

article_stances = Table(
    "article_stances", metadata,
    Column("article_id", Integer, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True),
    Column("qid", String(20), ForeignKey("entities.qid"), primary_key=True),
    Column("stance", String(20), nullable=False),
    Column("probs", JSON, nullable=False),  # {stance: probability}
    Column("model", String(60), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

# ---------- long-lived facts (nazariya/facts.py) ----------
# Raw articles are deleted after RETAIN_DAYS; these small tables are what the models read, and
# retention never touches them. No foreign keys, so they survive their source rows.

event_facts = Table(
    "event_facts", metadata,
    Column("event_id", Integer, primary_key=True),
    Column("first_seen", DateTime(timezone=True), nullable=False),
    Column("last_seen", DateTime(timezone=True), nullable=False),
    Column("n_articles", Integer, nullable=False),
    Column("n_outlets", Integer, nullable=False),
    Column("title", Text),
    Column("first_outlet", String(40)),
    Column("first_lang", String(8)),
    Column("langs", JSON),             # {lang: n outlets}
    Column("states", JSON),            # [state qid, ...] named in member articles
    Column("n_people", Integer),       # distinct linked persons
    Column("n_politicians", Integer),  # linked persons who are Indian politicians
    Column("toll", Integer),           # largest death toll in a member headline, if any
    Column("wire", Boolean),           # any member carries agency copy
    Column("govt", Boolean),           # a government press release (PIB) is a member
    Column("delhi_dateline", Boolean), # any member text opens with a New Delhi dateline
    Column("frame", String(30)),       # most common predicted frame among members
    # the same features for the first report alone (how the event looked when it broke): the
    # selection model uses these, since totals over all members partly result from wide coverage
    Column("first_report", JSON),
    Column("lang_drift", Float),       # 1 - cosine(mean English vector, mean Hindi vector)
    Column("lang_drift_n", Integer),   # vectors behind lang_drift
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
Index("ix_event_facts_first_seen", event_facts.c.first_seen)

coverage = Table(
    "coverage", metadata,
    Column("event_id", Integer, primary_key=True),
    Column("outlet_id", String(40), primary_key=True),
    Column("first_at", DateTime(timezone=True), nullable=False),  # earliest member article
    Column("first_own_at", DateTime(timezone=True)),              # earliest dated non-wire article
    Column("n_articles", Integer, nullable=False),
    Column("n_wire", Integer, nullable=False),                    # agency copy or copied text
    Column("n_text", Integer, nullable=False, default=0),         # members with extracted text
    # the outlet's earliest article on the event, so the site can link out to it
    Column("first_article_id", Integer),
    Column("first_url", Text),
    Column("first_title", Text),
)
Index("ix_coverage_outlet", coverage.c.outlet_id)

copies = Table(
    "copies", metadata,
    Column("article_id", Integer, primary_key=True),
    Column("outlet_id", String(40), nullable=False),
    Column("source_outlet_id", String(40), nullable=False),  # outlet whose earlier text it matches
    Column("agency", String(10)),                            # set when the copy is credited wire copy
    Column("published_at", DateTime(timezone=True)),
)

title_edits = Table(
    "title_edits", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("article_id", Integer, nullable=False),
    Column("outlet_id", String(40), nullable=False),
    Column("url", Text, nullable=False),
    Column("old_title", Text, nullable=False),
    Column("new_title", Text, nullable=False),
    Column("published_at", DateTime(timezone=True)),
    Column("seen_at", DateTime(timezone=True), nullable=False),
)
Index("ix_title_edits_seen", title_edits.c.seen_at)

# Each outlet's place on the outlet map once a week (rotated onto the previous week's map), so the
# dossier can show how an outlet's news judgement moves over time.
outlet_positions = Table(
    "outlet_positions", metadata,
    Column("week", String(10), primary_key=True),  # ISO week, e.g. 2026-W48
    Column("outlet_id", String(40), primary_key=True),
    Column("x", Float, nullable=False),
    Column("y", Float, nullable=False),
    Column("events", Integer, nullable=False),
)

# The front page frozen once a week ("issues"), kept for good so past weeks stay readable.
editions = Table(
    "editions", metadata,
    Column("issue", String(10), primary_key=True),  # ISO week the issue covers, e.g. 2026-W48
    Column("number", Integer, nullable=False),
    Column("published_at", DateTime(timezone=True), nullable=False),
    Column("body", Text, nullable=False),            # the analysis.json of that moment
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


# Free-tier Postgres (Neon) bills compute while it is awake, and it suspends after SUSPEND_AFTER_S
# without queries. Every statement's time is recorded so each run can print how long it kept the
# database awake; summing these over a month against the plan's compute-hour limit is the budget.
SUSPEND_AFTER_S = 300
_activity: list[float] = []
_reporting = False


def estimated_active_seconds(stamps: list[float] | None = None) -> float:
    """Seconds the database was kept awake: gaps shorter than the suspend delay count in full, and
    the last statement keeps it awake for one more delay."""
    ts = sorted(_activity if stamps is None else stamps)
    if not ts:
        return 0.0
    return sum(min(b - a, SUSPEND_AFTER_S) for a, b in zip(ts, ts[1:])) + SUSPEND_AFTER_S


def _report_activity() -> None:
    if _activity:
        span = _activity[-1] - _activity[0]
        print(f"[db] {len(_activity)} statements over {span / 60:.1f} min; estimated compute-active "
              f"{estimated_active_seconds() / 60:.1f} min (incl. {SUSPEND_AFTER_S // 60}-min idle tail)",
              file=sys.stderr)


def get_engine(url: str | None = None) -> Engine:
    global _reporting
    # pre-ping: Neon suspends idle compute and drops connections during long CPU-bound steps
    engine = create_engine(url or database_url(), future=True, pool_pre_ping=True)
    if engine.dialect.name == "postgresql":
        event.listen(engine, "before_cursor_execute", lambda *a, **k: _activity.append(time.time()))
        if not _reporting:
            atexit.register(_report_activity)
            _reporting = True
    elif engine.dialect.name == "sqlite":
        # SQLite ignores ON DELETE CASCADE unless asked; production (Postgres) always cascades
        event.listen(engine, "connect", lambda conn, _: conn.execute("PRAGMA foreign_keys=ON"))
    metadata.create_all(engine)
    add_missing_columns(engine)
    enable_rls(engine)
    return engine


def add_missing_columns(engine: Engine) -> list[str]:
    """create_all() never alters an existing table, and the pipeline's database is a long-lived
    file. Columns added to the schema later are added here; they must be nullable or defaulted."""
    from sqlalchemy import inspect
    added = []
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have:
                    continue
                ddl = f'{col.name} {col.type.compile(dialect=engine.dialect)}'
                if col.default is not None and getattr(col.default, "is_scalar", False):
                    ddl += f" DEFAULT {col.default.arg!r}"
                conn.exec_driver_sql(f"ALTER TABLE {table.name} ADD COLUMN {ddl}")
                added.append(f"{table.name}.{col.name}")
    if added:
        print(f"[db] added columns: {', '.join(added)}", file=sys.stderr)
    return added


# Supabase serves every table in `public` through its REST API to anyone with the project's anon
# key, and its security advisor flags tables with row-level security off. Nazariya never uses that
# API: the pipeline connects as the table owner, which bypasses RLS (it is not FORCEd), and the site
# reads static JSON. So RLS goes on with no policies, closing the REST path and changing nothing for
# the pipeline. Only tables this role owns are touched (ALTER TABLE on others would fail). Runs on
# every start, after create_all, so new tables are covered; tables already on are skipped.
ENABLE_RLS_SQL = """
DO $$
DECLARE
    t record;
BEGIN
    FOR t IN
        SELECT c.relname
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relkind IN ('r', 'p')
          AND NOT c.relrowsecurity
          AND pg_get_userbyid(c.relowner) = current_user
    LOOP
        EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', t.relname);
    END LOOP;
END
$$;
"""


def enable_rls(engine: Engine) -> None:
    """Turn on row-level security for every public table this role owns. Idempotent; Postgres only."""
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as conn:
        conn.execute(text(ENABLE_RLS_SQL))


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
