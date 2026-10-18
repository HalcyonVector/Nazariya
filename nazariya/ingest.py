"""Ingestion pipeline: discover → scope/section filter → store → extract → dedup/syndication.

Each run:
  1. discovery: for every outlet, try its source chain (config/feeds.yaml) in order until one
     returns fresh, in-scope items; record every attempt in fetch_runs; insert new articles
  2. extraction: fetch up to --budget pending articles per outlet (newest first, robots-allowed),
     run trafilatura, compute SimHash, detect wire bylines, mark duplicates / syndicated copies
  3. purge expired entries from the article_text cache

Usage:
    python -m nazariya.ingest                  # all outlets
    python -m nazariya.ingest --mvp            # MVP outlets only
    python -m nazariya.ingest --only thehindu,jagran
    python -m nazariya.ingest --no-extract     # discovery only
    python -m nazariya.ingest --budget 40      # max article fetches per outlet per run
    python -m nazariya.ingest --runner cloud   # GitHub Actions: skip outlets marked runner: home
    python -m nazariya.ingest --runner home    # home PC task: only outlets marked runner: home
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import trafilatura
from sqlalchemy import bindparam, delete, insert, select, update
from sqlalchemy.engine import Engine

from .config import load_feeds, load_filters, load_outlets, runner_of
from .db import (
    article_text, articles, existing_urls, fetch_runs, get_engine, insert_ignore, runs,
    outlets as outlets_table, upsert,
)
from .net import Fetcher, make_client, user_agent_for
from .sources import Item, fetch_source
from .textutil import (
    canonical_url, detect_agency, detect_lang, fix_published, hamming, in_scope, path_segments,
    rewrite_host, section_of, simhash, to_signed64, to_unsigned64,
)

MAX_ITEM_AGE = timedelta(days=3)  # ignore older sitemap/feed entries
STALE_SOURCE = timedelta(hours=36)  # newest item older than this → fall through to next source
TEXT_TTL = timedelta(days=14)  # processing cache lifetime; long enough to re-run models weekly
DEDUP_WINDOW = timedelta(hours=72)
SAME_OUTLET_MAX_BITS = 3  # SimHash distance for "same article re-published under a new URL"
SYNDICATED_MAX_BITS = 6  # cross-outlet wire copy; tune on labelled pairs
MIN_TEXT_CHARS = 200
FULL_TEXT_CHARS = 800
MAX_ATTEMPTS = 3
# Hindi dailies publish ~1k state/city stories a day each; all are kept as metadata, but
# extraction is budgeted newest-first and anything still pending after this is skipped.
DEFAULT_BUDGET = 100
EXTRACT_WINDOW = timedelta(hours=48)


@dataclass
class OutletStats:
    source_pos: int | None = None
    source_kind: str | None = None
    new: int = 0
    filtered: int = 0
    outcomes: Counter = field(default_factory=Counter)


@dataclass
class RecentHash:
    article_id: int
    outlet_id: str
    hash: int


class Ingester:
    def __init__(self, engine: Engine, fetcher: Fetcher, filters: dict, run_id: int, now: datetime):
        self.engine = engine
        self.fetcher = fetcher
        self.exclude_segments = set(filters.get("exclude_segments") or [])
        self.exclude_hosts = set(filters.get("exclude_hosts") or [])
        self.run_id = run_id
        self.now = now
        self.recent: list[RecentHash] = []

    # ---------- discovery ----------

    def excluded(self, url: str) -> bool:
        host = url.split("/")[2].removeprefix("www.")
        return host in self.exclude_hosts or any(s in self.exclude_segments for s in path_segments(url))

    def in_outlet_scope(self, o: dict, items: list[Item]) -> list[Item]:
        rewrites = o.get("rewrite_hosts") or {}
        extra = o.get("extra_hosts") or ()
        return [i for i in items if in_scope(rewrite_host(i.url, rewrites), o["homepage"], extra)]

    async def discover(self, o: dict, feed: dict, stats: OutletStats) -> None:
        ua = user_agent_for(o)
        attempts: list[dict] = []
        chosen: tuple[str, list[Item], dict] | None = None
        for pos, src in enumerate(feed["sources"]):
            t0 = time.monotonic()
            res = await fetch_source(self.fetcher, src["kind"], src["url"], ua)
            items = res.items if src["kind"] == "gnews" else self.in_outlet_scope(o, res.items)
            newest = max((i.published for i in items if i.published), default=None)
            row = {
                "run_id": self.run_id, "outlet_id": o["id"], "source_url": res.resolved_url or src["url"],
                "kind": src["kind"], "position": pos, "ok": res.ok, "accepted": False,
                "http_status": res.status, "error": res.error, "items_seen": len(res.items),
                "items_in_scope": len(items), "items_filtered": 0, "items_new": 0,
                "newest_item_at": newest, "duration_ms": int(1000 * (time.monotonic() - t0)),
                "fetched_at": self.now,
            }
            attempts.append(row)
            if res.ok and not items:
                row["error"] = "no in-scope items"
            elif res.ok and newest is not None and self.now - newest > STALE_SOURCE:
                row["error"] = f"stale: newest item {newest:%Y-%m-%d %H:%M} UTC"
            elif res.ok:
                row["accepted"] = True
                chosen = (src["kind"], items, row)
                break

        if chosen:
            kind, items, row = chosen
            stats.source_pos, stats.source_kind = row["position"], kind
            headline_only = kind == "gnews" or feed.get("access") == "headline_only"
            rewrites = o.get("rewrite_hosts") or {}
            candidates: dict[str, dict] = {}
            for it in items:
                published, fixed = fix_published(it.published, self.now)
                if published is not None and self.now - published > MAX_ITEM_AGE:
                    continue
                url = it.url if kind == "gnews" else canonical_url(it.url, rewrites)
                if kind != "gnews" and self.excluded(url):
                    row["items_filtered"] += 1
                    continue
                candidates.setdefault(url, {
                    "outlet_id": o["id"],
                    "url": url,
                    "title": it.title,
                    "summary": it.summary or None,
                    "lang": detect_lang(f"{it.title} {it.summary}", default=o["lang"]),
                    "section": None if kind == "gnews" else section_of(url),
                    # undated items (e.g. PIB) are dated by first sighting
                    "published_at": published or self.now,
                    "date_fixed": fixed or published is None,
                    "discovered_at": self.now,
                    "source_kind": kind,
                    "status": "headline_only" if headline_only else "pending",
                    "access": "headline_only" if headline_only else None,
                    "attempts": 0,
                })
            with self.engine.begin() as conn:
                known = existing_urls(conn, list(candidates))
                fresh = [r for u, r in candidates.items() if u not in known]
                insert_ignore(conn, articles, fresh, ["url"])
            row["items_new"] = len(fresh)
            stats.new, stats.filtered = len(fresh), row["items_filtered"]

        with self.engine.begin() as conn:
            conn.execute(insert(fetch_runs), attempts)

    # ---------- extraction ----------

    def load_recent_hashes(self) -> None:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(articles.c.id, articles.c.outlet_id, articles.c.simhash).where(
                    articles.c.simhash.is_not(None),
                    articles.c.status == "extracted",
                    articles.c.discovered_at >= self.now - DEDUP_WINDOW,
                )
            ).all()
        self.recent = [RecentHash(r.id, r.outlet_id, to_unsigned64(r.simhash)) for r in rows]

    def near_duplicates(self, h: int, outlet_id: str) -> tuple[int | None, int | None]:
        """(earliest same-outlet duplicate, earliest other-outlet syndicated source)."""
        same = other = None
        for r in self.recent:
            d = hamming(h, r.hash)
            if r.outlet_id == outlet_id and d <= SAME_OUTLET_MAX_BITS:
                same = r.article_id if same is None else min(same, r.article_id)
            elif r.outlet_id != outlet_id and d <= SYNDICATED_MAX_BITS:
                other = r.article_id if other is None else min(other, r.article_id)
        return same, other

    async def extract_one(self, o: dict, row, ua: str) -> tuple[dict, str | None]:
        """Returns (column updates, text to cache)."""
        if not await self.fetcher.allowed(row.url, ua):
            return {"status": "headline_only", "access": "headline_only", "error": "robots.txt disallows"}, None
        resp = await self.fetcher.get(row.url, ua)
        if resp is None or resp.status_code != 200:
            attempts = row.attempts + 1
            return {
                "attempts": attempts,
                "status": "failed" if attempts >= MAX_ATTEMPTS else "pending",
                "error": "request failed" if resp is None else f"HTTP {resp.status_code}",
            }, None

        doc = trafilatura.bare_extraction(resp.text, url=row.url, favor_recall=True, with_metadata=True)
        text = (getattr(doc, "text", None) or "") if doc is not None else ""
        title = row.title or (getattr(doc, "title", None) or "")
        upd: dict = {"attempts": row.attempts + 1, "extracted_at": self.now, "text_len": len(text),
                     "title": title, "error": None}
        if len(text) < MIN_TEXT_CHARS:
            upd.update(status="headline_only", access="headline_only" if not text else "lede_only")
            return upd, None

        h = simhash(text)
        upd.update(
            status="extracted",
            access="full" if len(text) >= FULL_TEXT_CHARS else "lede_only",
            lang=detect_lang(f"{title} {text[:1000]}", default=o["lang"]),
            simhash=to_signed64(h),
            agency=detect_agency(text),
        )
        same, other = self.near_duplicates(h, o["id"])
        if same is not None:
            upd.update(status="duplicate", duplicate_of=same)
            return upd, None
        if other is not None:
            upd["syndicated_of"] = other
        self.recent.append(RecentHash(row.id, o["id"], h))
        return upd, text

    async def extract(self, o: dict, budget: int, stats: OutletStats) -> None:
        with self.engine.begin() as conn:
            stats.outcomes["skipped"] += conn.execute(
                update(articles)
                .where(articles.c.outlet_id == o["id"], articles.c.status == "pending",
                       articles.c.published_at < self.now - EXTRACT_WINDOW)
                .values(status="skipped", error="not extracted within window")
            ).rowcount
            pending = conn.execute(
                select(articles.c.id, articles.c.url, articles.c.title, articles.c.attempts)
                .where(articles.c.outlet_id == o["id"], articles.c.status == "pending")
                .order_by(articles.c.published_at.desc())
                .limit(budget)
            ).all()
        if not stats.outcomes["skipped"]:
            del stats.outcomes["skipped"]
        ua = user_agent_for(o)
        # the Fetcher caps each host at 2 concurrent requests
        results = await asyncio.gather(*(self.extract_one(o, row, ua) for row in pending))

        # One transaction per outlet, one executemany per distinct set of updated columns.
        # Per-article commits cost several network round trips each against a remote Postgres,
        # and these sync calls block the event loop (stalling every other outlet's fetches).
        groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
        texts = []
        for row, (upd, text) in zip(pending, results):
            stats.outcomes[upd.get("status", "pending")] += 1
            groups[tuple(sorted(upd))].append({"b_id": row.id, **{f"v_{k}": v for k, v in upd.items()}})
            if text:
                texts.append({"article_id": row.id, "text": text, "expires_at": self.now + TEXT_TTL})
        with self.engine.begin() as conn:
            for cols, params in groups.items():
                stmt = (update(articles).where(articles.c.id == bindparam("b_id"))
                        .values({c: bindparam(f"v_{c}") for c in cols}))
                conn.execute(stmt, params)
            upsert(conn, article_text, texts, ["article_id"])

    def purge_text_cache(self) -> int:
        with self.engine.begin() as conn:
            return conn.execute(delete(article_text).where(article_text.c.expires_at < self.now)).rowcount


def select_outlets(all_outlets: list[dict], feeds: dict, only: str | None, mvp: bool,
                   runner: str = "all") -> list[dict]:
    chosen = all_outlets if runner == "all" else [o for o in all_outlets if runner_of(o) == runner]
    if only:
        wanted = set(only.split(","))
        chosen = [o for o in chosen if o["id"] in wanted]
    if mvp:
        chosen = [o for o in chosen if o.get("mvp")]
    missing = [o["id"] for o in chosen if o["id"] not in feeds]
    if missing:
        print(f"skipping outlets with no entry in feeds.yaml: {', '.join(missing)}")
    return [o for o in chosen if o["id"] in feeds]


async def run(args: argparse.Namespace) -> None:
    all_outlets, feeds, filters = load_outlets(), load_feeds(), load_filters()
    chosen = select_outlets(all_outlets, feeds, args.only, args.mvp, args.runner)
    engine = get_engine()
    now = datetime.now(timezone.utc).replace(microsecond=0)

    with engine.begin() as conn:
        upsert(conn, outlets_table, [
            {k: o.get(k) for k in ("id", "name", "lang", "region", "role", "homepage")} for o in all_outlets
        ], ["id"])
        run_id = conn.execute(insert(runs).values(started_at=now)).inserted_primary_key[0]

    stats = {o["id"]: OutletStats() for o in chosen}
    t0 = time.monotonic()
    async with make_client() as client:
        ing = Ingester(engine, Fetcher(client), filters, run_id, now)
        await asyncio.gather(*(ing.discover(o, feeds[o["id"]], stats[o["id"]]) for o in chosen))
        t_discover = time.monotonic() - t0
        if not args.no_extract:
            ing.load_recent_hashes()
            await asyncio.gather(*(ing.extract(o, args.budget, stats[o["id"]]) for o in chosen))
        purged = ing.purge_text_cache()

    elapsed = time.monotonic() - t0
    totals = Counter()
    print(f"\n{'outlet':18s} {'src':>4s} {'kind':8s} {'new':>5s} {'filt':>5s}  extraction")
    for o in chosen:
        s = stats[o["id"]]
        totals.update(s.outcomes)
        totals["new"] += s.new
        totals["filtered"] += s.filtered
        src = "FAIL" if s.source_pos is None else str(s.source_pos)
        outcomes = " ".join(f"{k}={v}" for k, v in sorted(s.outcomes.items()))
        print(f"{o['id']:18s} {src:>4s} {s.source_kind or '':8s} {s.new:5d} {s.filtered:5d}  {outcomes}")
    summary = {
        "outlets": len(chosen),
        "outlets_failed": sum(s.source_pos is None for s in stats.values()),
        "outlets_on_fallback": sum((s.source_pos or 0) > 0 for s in stats.values()),
        "discover_s": round(t_discover, 1),
        "total_s": round(elapsed, 1),
        "text_cache_purged": purged,
        **totals,
    }
    with engine.begin() as conn:
        conn.execute(update(runs).where(runs.c.id == run_id).values(
            finished_at=datetime.now(timezone.utc), stats=summary))
    print(f"\nrun {run_id}: {summary}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated outlet ids")
    ap.add_argument("--mvp", action="store_true", help="only outlets marked mvp: true")
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--runner", choices=["all", "cloud", "home"], default="all",
                    help="which outlets to ingest, by their `runner` in outlets.yaml")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="article fetches per outlet per run")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
