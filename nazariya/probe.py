"""Discover and health-check feeds for every candidate outlet.

For each outlet in config/outlets.yaml:
  1. collect candidate feeds: registry hints, homepage <link rel=alternate>, common paths
  2. collect news sitemaps from robots.txt (plus common sitemap paths)
  3. parse each candidate: item count, items in the last 48h, newest item age, % dated
  4. pick the best source and test full-text extraction on a few recent articles
     (only URLs robots.txt allows)
  5. if nothing works, fall back to Google News RSS (headlines + timestamps only)

Usage:
    python -m nazariya.probe                 # all outlets
    python -m nazariya.probe --only thehindu,jagran
    python -m nazariya.probe --no-extract    # skip article fetches
    python -m nazariya.probe --write-feeds   # also regenerate config/feeds.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus, urljoin, urlparse

import trafilatura
import yaml
from lxml import etree, html as lhtml

from .config import ROOT, load_outlets
from .net import Fetcher, make_client, user_agent_for
from .sources import fetch_source, is_article_sitemap
from .textutil import DEVANAGARI, in_scope, rewrite_host

COMMON_FEED_PATHS = ["/feed", "/feed/", "/rss", "/rss.xml", "/feeds", "/rssfeeds"]
COMMON_SITEMAP_PATHS = ["/news-sitemap.xml", "/sitemap-news.xml", "/sitemap_news.xml", "/news_sitemap.xml"]
GNEWS_RSS = "https://news.google.com/rss/search?q={q}&hl={hl}&gl=IN&ceid=IN:{lang}"
RECENT_HOURS = 48
EXTRACT_SAMPLES = 3
MIN_FULLTEXT_CHARS = 800


@dataclass
class SourceCheck:
    url: str
    kind: str  # rss | sitemap | gnews
    origin: str  # hint | homepage | common | robots | fallback
    ok: bool = False
    status: int | None = None
    error: str | None = None
    resolved_url: str | None = None
    items: int = 0
    items_recent: int = 0
    pct_dated: float = 0.0
    newest_age_h: float | None = None
    sample_links: list[str] = field(default_factory=list)
    sample_titles: list[str] = field(default_factory=list)


@dataclass
class OutletResult:
    id: str
    name: str
    lang: str
    role: str
    mvp: bool
    user_agent: str = "bot"
    homepage_status: int | None = None
    sources: list[SourceCheck] = field(default_factory=list)
    best: str | None = None
    best_kind: str | None = None
    extract_chars: list[int] = field(default_factory=list)
    extract_blocked: int = 0
    access: str = "unknown"  # full | lede_only | headline_only | unknown
    devanagari_titles: float = 0.0
    verdict: str = "fail"


def gnews_url(outlet: dict, window: str = "2d") -> str:
    domain = urlparse(outlet["homepage"]).netloc.removeprefix("www.")
    hl = "hi" if outlet["lang"] == "hi" else "en-IN"
    return GNEWS_RSS.format(q=quote_plus(f"site:{domain} when:{window}"), hl=hl, lang=outlet["lang"])


class Prober:
    def __init__(self, fetcher: Fetcher, extract: bool):
        self.fetcher = fetcher
        self.extract = extract

    async def check(self, o: dict, kind: str, url: str, origin: str, now: float) -> SourceCheck:
        res = await fetch_source(self.fetcher, kind, url, user_agent_for(o))
        sc = SourceCheck(url=url, kind=kind, origin=origin, ok=res.ok, status=res.status,
                         error=res.error, resolved_url=res.resolved_url)
        if res.ok and kind != "gnews":  # gnews links are Google redirects
            rewrites = o.get("rewrite_hosts") or {}
            res.items = [i for i in res.items
                         if in_scope(rewrite_host(i.url, rewrites), o["homepage"], o.get("extra_hosts") or ())]
            if not res.items:
                sc.ok, sc.error = False, "no in-scope items"
        if not sc.ok:
            return sc
        stamps = [i.published.timestamp() if i.published else None for i in res.items]
        dated = [s for s in stamps if s]
        sc.items = len(res.items)
        sc.pct_dated = round(100 * len(dated) / len(res.items), 1)
        # undated feeds (e.g. PIB) can't show freshness; the ingester dates items by first sighting
        sc.items_recent = sum(1 for s in dated if now - s <= RECENT_HOURS * 3600) if dated else len(res.items)
        sc.newest_age_h = round((now - max(dated)) / 3600, 1) if dated else None
        newest_first = sorted(res.items, key=lambda i: i.published.timestamp() if i.published else 0, reverse=True)
        sc.sample_links = [i.url for i in newest_first[:10]]
        sc.sample_titles = [i.title for i in newest_first[:10]]
        return sc

    async def homepage_feeds(self, homepage: str, ua: str) -> tuple[int | None, list[str]]:
        resp = await self.fetcher.get(homepage, ua)
        if resp is None:
            return None, []
        found = []
        if resp.status_code == 200:
            try:
                doc = lhtml.fromstring(resp.content)
                for link in doc.xpath('//link[@rel="alternate"][@href]'):
                    typ = (link.get("type") or "").lower()
                    if "rss" in typ or "atom" in typ:
                        found.append(urljoin(str(resp.url), link.get("href")))
            except (etree.ParserError, ValueError):
                pass
        return resp.status_code, found

    async def extract_sample(self, links: list[str], ua: str) -> tuple[list[int], int]:
        chars, blocked = [], 0
        for link in links[:EXTRACT_SAMPLES]:
            if not await self.fetcher.allowed(link, ua):
                blocked += 1
                continue
            resp = await self.fetcher.get(link, ua)
            if resp is None or resp.status_code != 200:
                chars.append(0)
                continue
            text = trafilatura.extract(resp.text, url=link, favor_recall=True) or ""
            chars.append(len(text))
        return chars, blocked

    async def probe(self, o: dict) -> OutletResult:
        now = time.time()
        ua = user_agent_for(o)
        res = OutletResult(id=o["id"], name=o["name"], lang=o["lang"], role=o["role"],
                           mvp=bool(o.get("mvp")), user_agent=o.get("user_agent", "bot"))
        homepage = o["homepage"]
        base = f"{urlparse(homepage).scheme}://{urlparse(homepage).netloc}"

        (status, alt_feeds), (_, robot_sitemaps) = await asyncio.gather(
            self.homepage_feeds(homepage, ua), self.fetcher.robots_for(homepage, ua)
        )
        res.homepage_status = status

        rss: dict[str, str] = {}
        for u in o.get("hints") or []:
            rss.setdefault(u, "hint")
        for u in alt_feeds:
            rss.setdefault(u, "homepage")
        for p in COMMON_FEED_PATHS:
            rss.setdefault(base + p, "common")

        sitemaps: dict[str, str] = {}
        for u in robot_sitemaps:
            if is_article_sitemap(u):
                sitemaps.setdefault(u, "robots")
        for p in COMMON_SITEMAP_PATHS:
            sitemaps.setdefault(base + p, "common")

        checks = await asyncio.gather(
            *(self.check(o, "rss", u, src, now) for u, src in rss.items()),
            *(self.check(o, "sitemap", u, src, now) for u, src in sitemaps.items()),
        )
        # drop duplicates that resolve to the same content (e.g. /feed and /feed/)
        seen, uniq = set(), []
        for c in checks:
            key = (c.kind, tuple(c.sample_links[:3])) if c.ok else (c.kind, c.url)
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        rewrites = o.get("rewrite_hosts") or {}
        for c in uniq:
            c.sample_links = [rewrite_host(u, rewrites) for u in c.sample_links]
        res.sources = uniq

        good = [c for c in uniq if c.ok and c.items_recent > 0]
        if not good:
            fb = await self.check(o, "gnews", gnews_url(o), "fallback", now)
            res.sources.append(fb)
            if fb.ok and fb.items_recent > 0:
                good = [fb]
        if good:
            # most recent items wins; hints break ties since they're curated India/national sections
            best = max(good, key=lambda c: (c.items_recent, c.origin == "hint", c.pct_dated))
            res.best, res.best_kind = best.url, best.kind
            titles = [t for t in best.sample_titles if t]
            if titles:
                res.devanagari_titles = round(sum(bool(DEVANAGARI.search(t)) for t in titles) / len(titles), 2)
            if best.kind == "gnews":
                res.access = "headline_only"  # links are Google redirects; no article fetch
            elif self.extract:
                res.extract_chars, res.extract_blocked = await self.extract_sample(best.sample_links, ua)
                if res.extract_chars:
                    med = statistics.median(res.extract_chars)
                    res.access = "full" if med >= MIN_FULLTEXT_CHARS else ("lede_only" if med > 0 else "headline_only")
                elif res.extract_blocked:
                    res.access = "headline_only"
            fresh = best.newest_age_h is not None and best.newest_age_h <= 12
            res.verdict = "ok" if fresh and res.access in ("full", "unknown") else "degraded"
        return res


def write_report(results: list[OutletResult], path: Path) -> None:
    ok = sum(r.verdict == "ok" for r in results)
    lines = [
        "# Feed probe report",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} · {ok}/{len(results)} outlets ok",
        "",
        "`items_48h` = items in best source published in last 48h · `extract` = median chars of "
        f"trafilatura text over {EXTRACT_SAMPLES} newest articles (robots-allowed only) · "
        "`ua` = user agent used (bot, or browser where the CDN blocks bots)",
        "",
        "| outlet | lang | mvp | ua | verdict | best source | kind | items_48h | newest (h) | extract | access |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: (r.verdict != "ok", r.lang, r.id)):
        best = next((s for s in r.sources if s.url == r.best), None)
        med = int(statistics.median(r.extract_chars)) if r.extract_chars else "-"
        lines.append(
            f"| {r.name} | {r.lang} | {'✓' if r.mvp else ''} | {r.user_agent} | {r.verdict} | "
            f"{r.best or '—'} | {r.best_kind or ''} | {best.items_recent if best else 0} | "
            f"{best.newest_age_h if best else '-'} | {med} | {r.access} |"
        )
    lines += ["", "## Failures and alternatives", ""]
    for r in results:
        if r.verdict == "ok":
            continue
        lines.append(f"### {r.name} (`{r.id}`) — homepage HTTP {r.homepage_status}")
        for s in r.sources:
            if s.ok or s.origin in ("hint", "homepage", "robots"):
                info = f"{s.items} items, {s.items_recent} recent" if s.ok else s.error
                lines.append(f"- {s.kind} [{s.origin}] {s.url} → {info}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_feeds(results: list[OutletResult], outlets: list[dict], path: Path) -> None:
    """Per-outlet fallback chain: best source, other working sources, then Google News.

    Sitemap indexes are stored as the index URL (resolved to the newest child at fetch
    time), so date-stamped child sitemaps don't go stale.
    """
    by_id = {o["id"]: o for o in outlets}
    feeds = {}
    for r in sorted(results, key=lambda r: r.id):
        working = sorted(
            (s for s in r.sources if s.ok and s.items_recent > 0 and s.kind != "gnews"),
            key=lambda s: (s.url != r.best, -s.items_recent),
        )
        chain = [{"kind": s.kind, "url": s.url} for s in working[:3]]
        chain.append({"kind": "gnews", "url": gnews_url(by_id[r.id], "1d")})
        feeds[r.id] = {"access": r.access, "sources": chain}
    header = (
        "# Generated by `python -m nazariya.probe --write-feeds`; edit by hand afterwards.\n"
        "# Sources are tried in order; gnews is the headline-only last resort.\n"
        "# access: full | lede_only | headline_only (measured by the probe)\n"
    )
    path.write_text(header + yaml.safe_dump({"feeds": feeds}, sort_keys=False, allow_unicode=True), encoding="utf-8")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated outlet ids")
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--write-feeds", action="store_true", help="(re)generate config/feeds.yaml")
    args = ap.parse_args()

    all_outlets = load_outlets()
    outlets = all_outlets
    if args.only:
        wanted = set(args.only.split(","))
        outlets = [o for o in outlets if o["id"] in wanted]

    async with make_client() as client:
        prober = Prober(Fetcher(client), extract=not args.no_extract)
        results = []
        for coro in asyncio.as_completed([prober.probe(o) for o in outlets]):
            r = await coro
            results.append(r)
            print(f"{r.verdict:9s} {r.id:18s} {r.best_kind or '':8s} {r.access}", flush=True)

    out = ROOT / "data" / "probe"
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.json"
    if args.only and results_path.exists():
        # partial run: merge into previous results so the report and feeds stay complete
        previous = {r["id"]: r for r in json.loads(results_path.read_text(encoding="utf-8"))}
        for r in results:
            previous[r.id] = asdict(r)
        merged = list(previous.values())
    else:
        merged = [asdict(r) for r in results]
    results_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    full = [OutletResult(**{**m, "sources": [SourceCheck(**s) for s in m["sources"]]}) for m in merged]

    (ROOT / "reports").mkdir(exist_ok=True)
    write_report(full, ROOT / "reports" / "feed_probe.md")
    if args.write_feeds:
        write_feeds(full, all_outlets, ROOT / "config" / "feeds.yaml")
    print(f"\n{sum(r.verdict == 'ok' for r in full)}/{len(full)} ok -> reports/feed_probe.md")


if __name__ == "__main__":
    asyncio.run(main())
