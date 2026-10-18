"""Per-outlet ingestion health: is each source alive, on a fallback, unusually quiet, or failing extraction?

Usage:
    python -m nazariya.health                 # print + write reports/health.md
    python -m nazariya.health --fail-on-down  # exit 1 if any outlet is DOWN (for CI alerts)

Flags:
    DOWN      no accepted fetch in the last 6h
    FALLBACK  latest run used a fallback source (position > 0)
    LOW       new articles in last 24h < 30% of the trailing 7-day daily median
    EXTRACT   < 50% of extraction attempts in last 24h produced text (outlet expected full text)

These states are what the blindspot analysis must treat as "no data", not "didn't cover".
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from .config import ROOT, load_feeds, load_outlets
from .db import articles, as_utc, fetch_runs, get_engine

DOWN_AFTER = timedelta(hours=6)
LOW_RATIO = 0.3
LOW_MIN_MEDIAN = 10
EXTRACT_MIN_RATE = 0.5


def compute(now: datetime) -> list[dict]:
    engine = get_engine()
    since = now - timedelta(days=8)
    with engine.connect() as conn:
        fr = conn.execute(
            select(fetch_runs.c.outlet_id, fetch_runs.c.run_id, fetch_runs.c.position,
                   fetch_runs.c.accepted, fetch_runs.c.error, fetch_runs.c.fetched_at)
            .where(fetch_runs.c.fetched_at >= since)
        ).all()
        arts = conn.execute(
            select(articles.c.outlet_id, articles.c.discovered_at, articles.c.status, articles.c.extracted_at)
            .where(articles.c.discovered_at >= since)
        ).all()

    last_ok: dict[str, datetime] = {}
    latest_run: dict[str, int] = {}
    latest_rows: dict[str, list] = defaultdict(list)
    for r in fr:
        t = as_utc(r.fetched_at)
        if r.accepted and (r.outlet_id not in last_ok or t > last_ok[r.outlet_id]):
            last_ok[r.outlet_id] = t
        if r.run_id > latest_run.get(r.outlet_id, -1):
            latest_run[r.outlet_id] = r.run_id
            latest_rows[r.outlet_id] = []
        if r.run_id == latest_run[r.outlet_id]:
            latest_rows[r.outlet_id].append(r)

    daily: dict[str, Counter] = defaultdict(Counter)
    new_24h: Counter = Counter()
    extraction: dict[str, Counter] = defaultdict(Counter)
    for a in arts:
        d = as_utc(a.discovered_at)
        if now - d <= timedelta(days=1):
            new_24h[a.outlet_id] += 1
        else:
            daily[a.outlet_id][d.date()] += 1
        ext = as_utc(a.extracted_at)
        if ext and now - ext <= timedelta(days=1):
            extraction[a.outlet_id][a.status] += 1

    feeds = load_feeds()
    rows = []
    for o in load_outlets():
        oid = o["id"]
        if oid not in latest_run:
            continue  # never ingested (e.g. excluded by --only)
        flags = []
        ok_at = last_ok.get(oid)
        if ok_at is None or now - ok_at > DOWN_AFTER:
            flags.append("DOWN")
        accepted = [r for r in latest_rows[oid] if r.accepted]
        pos = accepted[0].position if accepted else None
        if pos:
            flags.append("FALLBACK")
        days = sorted(daily[oid].values())
        median = statistics.median(days) if days else None
        if median is not None and median >= LOW_MIN_MEDIAN and new_24h[oid] < LOW_RATIO * median:
            flags.append("LOW")
        ex = extraction[oid]
        attempts = ex["extracted"] + ex["duplicate"] + ex["headline_only"] + ex["failed"]
        rate = (ex["extracted"] + ex["duplicate"]) / attempts if attempts else None
        if feeds.get(oid, {}).get("access") == "full" and rate is not None and attempts >= 5 and rate < EXTRACT_MIN_RATE:
            flags.append("EXTRACT")
        errors = [f"[{r.position}] {r.error}" for r in latest_rows[oid] if r.error]
        rows.append({
            "id": oid, "name": o["name"], "lang": o["lang"], "flags": flags,
            "last_ok_h": None if ok_at is None else round((now - ok_at).total_seconds() / 3600, 1),
            "source_pos": pos, "new_24h": new_24h[oid], "median_daily": median,
            "extract_rate": None if rate is None else round(rate, 2), "extract_attempts": attempts,
            "errors": errors,
        })
    return rows


def render(rows: list[dict], now: datetime) -> str:
    bad = [r for r in rows if r["flags"]]
    lines = [
        "# Ingestion health",
        "",
        f"Generated {now:%Y-%m-%d %H:%M UTC} · {len(rows) - len(bad)}/{len(rows)} outlets healthy",
        "",
        "| outlet | lang | flags | last ok (h) | source | new 24h | 7d median/day | extract rate (n) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (not r["flags"], r["id"])):
        rate = "-" if r["extract_rate"] is None else f"{r['extract_rate']:.0%} ({r['extract_attempts']})"
        lines.append(
            f"| {r['name']} | {r['lang']} | {' '.join(r['flags']) or 'ok'} | {r['last_ok_h'] if r['last_ok_h'] is not None else '—'} | "
            f"{'—' if r['source_pos'] is None else r['source_pos']} | {r['new_24h']} | "
            f"{'-' if r['median_daily'] is None else r['median_daily']} | {rate} |"
        )
    errs = [r for r in rows if r["errors"]]
    if errs:
        lines += ["", "## Source errors in latest run", ""]
        for r in errs:
            lines.append(f"- **{r['name']}**: " + "; ".join(r["errors"]))
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-on-down", action="store_true")
    args = ap.parse_args()
    now = datetime.now(timezone.utc)
    rows = compute(now)
    report = render(rows, now)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "health.md").write_text(report, encoding="utf-8")
    print(report)
    down = [r["id"] for r in rows if "DOWN" in r["flags"]]
    if args.fail_on_down and down:
        print(f"DOWN: {', '.join(down)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
