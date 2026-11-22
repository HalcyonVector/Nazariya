"""Stories that disappear: re-check a sample of 1 to 3 day old articles and log the ones removed.

Every run checks up to PER_RUN articles (newest first among those due, spread across outlets) that
were extracted successfully and are between MIN_AGE and MAX_AGE old. An article counts as removed
only when it is clearly gone (HTTP 404 or 410, or a redirect to the site's front page) on two checks
at least CONFIRM_GAP apart. Blocks (401/403), rate limits (429), server errors and timeouts never
count. robots.txt is obeyed, as for every other request.

Usage:
    python -m nazariya.recheck [--per-run 300]
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from sqlalchemy import select

from .config import load_outlets
from .db import articles, as_utc, get_engine, removals, upsert
from .net import Fetcher, make_client, user_agent_for

PER_RUN = 300
MIN_AGE = timedelta(hours=24)
MAX_AGE = timedelta(hours=72)
CONFIRM_GAP = timedelta(hours=4)
GONE = {404, 410}


def is_gone(status: int | None, requested: str, final: str | None) -> bool:
    if status in GONE:
        return True
    if status == 200 and final:
        a, b = urlparse(requested), urlparse(final)
        # a story URL that now lands on the bare front page has been taken down
        return a.path not in ("", "/") and b.path in ("", "/")
    return False


def due(engine, now: datetime, per_run: int = PER_RUN) -> list:
    with engine.connect() as conn:
        seen = {r.article_id: r for r in conn.execute(select(removals))}
        rows = conn.execute(
            select(articles.c.id, articles.c.outlet_id, articles.c.url, articles.c.title, articles.c.published_at)
            .where(articles.c.status == "extracted", articles.c.published_at <= now - MIN_AGE,
                   articles.c.published_at >= now - MAX_AGE)
            .order_by(articles.c.published_at.desc())
        ).all()
    picked, per_outlet = [], defaultdict(int)
    cap = max(5, per_run // 25)
    for r in rows:
        s = seen.get(r.id)
        if s and (s.confirmed_at or (s.last_checked_at and now - as_utc(s.last_checked_at) < CONFIRM_GAP)):
            continue
        if s and not s.first_gone_at and s.checks >= 1:
            continue  # checked once and still there: done
        if per_outlet[r.outlet_id] >= cap:
            continue
        per_outlet[r.outlet_id] += 1
        picked.append((r, s))
        if len(picked) >= per_run:
            break
    return picked


async def check(engine, now: datetime | None = None, per_run: int = PER_RUN) -> dict:
    now = now or datetime.now(timezone.utc)
    outs = {o["id"]: o for o in load_outlets()}
    batch = due(engine, now, per_run)
    async with make_client() as client:
        f = Fetcher(client)

        async def one(r):
            ua = user_agent_for(outs.get(r.outlet_id, {}))
            if not await f.allowed(r.url, ua):
                return None, None
            resp = await f.get(r.url, ua)
            return (resp.status_code, str(resp.url)) if resp is not None else (None, None)

        results = await asyncio.gather(*(one(r) for r, _ in batch))
    out, gone, confirmed = [], 0, 0
    for (r, s), (status, final) in zip(batch, results):
        if status is None:
            continue
        g = is_gone(status, r.url, final)
        first = as_utc(s.first_gone_at) if s and s.first_gone_at else None
        conf = None
        if g:
            gone += 1
            if first and now - first >= CONFIRM_GAP:
                conf = now
                confirmed += 1
            first = first or now
        else:
            first = None  # back again: a blip, not a removal
        out.append({"article_id": r.id, "outlet_id": r.outlet_id, "url": r.url, "title": r.title,
                    "published_at": as_utc(r.published_at), "status": status, "first_gone_at": first,
                    "confirmed_at": conf, "last_checked_at": now, "checks": (s.checks if s else 0) + 1})
    with engine.begin() as conn:
        upsert(conn, removals, out, ["article_id"])
    return {"checked": len(out), "gone_now": gone, "confirmed": confirmed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-run", type=int, default=PER_RUN)
    print(asyncio.run(check(get_engine(), per_run=ap.parse_args().per_run)))


if __name__ == "__main__":
    main()
