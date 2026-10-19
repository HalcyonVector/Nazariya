"""Export JSON snapshots for the static site (site/data/). Headlines, links and derived data only;
no article text.

stories.json  events covered by ≥ MIN_OUTLETS outlets in the last STORY_WINDOW, ranked by
              outlet count decayed by age. Each lists every member article (outlet, headline,
              link, language, time, wire agency) and, per outlet, whether we had data at all:
              an outlet with no successful fetch during the story's window is "no data", which
              the page draws differently from "did not cover".
outlets.json  all outlets with role, language, runner and current ingestion health.

Usage:
    python -m nazariya.export            # → site/data/*.json
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from .cluster import MODEL, TAU
from .config import ROOT, load_outlets, runner_of
from .db import article_events, articles, as_utc, events, fetch_runs, get_engine
from .health import compute as compute_health

STORY_WINDOW = timedelta(hours=48)
MIN_OUTLETS = 2
MAX_STORIES = 200
HALF_LIFE_H = 18  # ranking: an 18-hour-old story counts half as much as a fresh one
NO_DATA_SLACK = timedelta(hours=12)  # a fetch this soon after the story's last article still counts
# Templated hyper-local items from one paper (e.g. dozens of temple-aarti reports) cluster together;
# a "story" where one outlet supplies most articles is not cross-outlet coverage, so it is left out.
MAX_OUTLET_SHARE = 0.6
MIN_ARTICLES_FOR_SHARE_RULE = 10
OUT = ROOT / "site" / "data"


def iso(dt: datetime | None) -> str | None:
    return None if dt is None else as_utc(dt).isoformat(timespec="minutes").replace("+00:00", "Z")


def export(now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    engine = get_engine()
    registry = {o["id"]: o for o in load_outlets()}
    with engine.connect() as conn:
        evs = conn.execute(
            select(events.c.id, events.c.title, events.c.first_seen, events.c.last_seen,
                   events.c.n_articles, events.c.n_outlets)
            .where(events.c.last_seen >= now - STORY_WINDOW, events.c.n_outlets >= MIN_OUTLETS)
        ).all()
        members = conn.execute(
            select(article_events.c.event_id, articles.c.outlet_id, articles.c.title, articles.c.url,
                   articles.c.lang, articles.c.published_at, articles.c.agency, articles.c.access,
                   articles.c.syndicated_of, article_events.c.similarity)
            .join(articles, articles.c.id == article_events.c.article_id)
            .where(article_events.c.event_id.in_([e.id for e in evs]))
        ).all() if evs else []
        fetches = conn.execute(
            select(fetch_runs.c.outlet_id, fetch_runs.c.fetched_at)
            .where(fetch_runs.c.accepted.is_(True), fetch_runs.c.fetched_at >= now - STORY_WINDOW - timedelta(days=1))
        ).all()

    ok_times: dict[str, list[datetime]] = defaultdict(list)
    for f in fetches:
        ok_times[f.outlet_id].append(as_utc(f.fetched_at))
    by_event: dict[int, list] = defaultdict(list)
    for m in members:
        by_event[m.event_id].append(m)

    stories = []
    for e in evs:
        first, last = as_utc(e.first_seen), as_utc(e.last_seen)
        arts = sorted(by_event[e.id], key=lambda m: as_utc(m.published_at))
        per_outlet = Counter(m.outlet_id for m in arts)
        if len(arts) >= MIN_ARTICLES_FOR_SHARE_RULE and per_outlet.most_common(1)[0][1] / len(arts) > MAX_OUTLET_SHARE:
            continue
        covered = set(per_outlet)
        window_end = min(now, last + NO_DATA_SLACK)
        no_data = sorted(
            oid for oid in registry
            if oid not in covered and not any(first <= t <= window_end for t in ok_times.get(oid, []))
        )
        age_h = (now - last).total_seconds() / 3600
        stories.append({
            "id": e.id,
            "title": e.title,
            "first_seen": iso(first),
            "last_seen": iso(last),
            "n_articles": e.n_articles,
            "n_outlets": e.n_outlets,
            "score": round(e.n_outlets * math.pow(0.5, age_h / HALF_LIFE_H), 3),
            "langs": sorted({m.lang for m in arts if m.lang}),
            "no_data": no_data,
            "articles": [{
                "outlet": m.outlet_id, "title": m.title, "url": m.url, "lang": m.lang,
                "published": iso(m.published_at), "agency": m.agency, "access": m.access,
                "syndicated": m.syndicated_of is not None,
            } for m in arts],
        })
    stories.sort(key=lambda s: -s["score"])
    stories = stories[:MAX_STORIES]

    health = {r["id"]: r for r in compute_health(now)}
    outlets_out = []
    for oid, o in registry.items():
        h = health.get(oid, {})
        outlets_out.append({
            "id": oid, "name": o["name"], "lang": o["lang"], "region": o["region"], "role": o["role"],
            "runner": runner_of(o), "status": " ".join(h.get("flags", [])) or ("ok" if h else "unknown"),
            "last_ok_h": h.get("last_ok_h"), "new_24h": h.get("new_24h", 0),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"generated_at": iso(now), "window_hours": int(STORY_WINDOW.total_seconds() // 3600),
            "min_outlets": MIN_OUTLETS, "model": MODEL, "tau": TAU, "stories": len(stories)}
    (OUT / "stories.json").write_text(json.dumps({"meta": meta, "stories": stories}, ensure_ascii=False), encoding="utf-8")
    (OUT / "outlets.json").write_text(json.dumps({"meta": meta, "outlets": outlets_out}, ensure_ascii=False), encoding="utf-8")
    return meta


def main() -> None:
    print(export())


if __name__ == "__main__":
    main()
