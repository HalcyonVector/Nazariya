"""Fetch and parse one listing source (RSS/Atom, Google News sitemap, Google News RSS) into Items."""

from __future__ import annotations

import html
import re
from calendar import timegm
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import feedparser
from lxml import etree

from .net import BOT_UA, Fetcher

IST = timezone(timedelta(hours=5, minutes=30))
# sitemap-index children worth following: Google News sitemaps and rolling "today" sitemaps
SITEMAP_KEYWORDS = ("news", "today")
NON_ARTICLE_SITEMAP = re.compile(r"image|video|photo|gallery|webstor|web-stor|visual-stor|podcast", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
UTF8_BOM = bytes([0xEF, 0xBB, 0xBF])


@dataclass
class Item:
    url: str
    title: str
    published: datetime | None
    summary: str = ""


@dataclass
class SourceResult:
    url: str
    kind: str  # rss | sitemap | gnews
    ok: bool = False
    status: int | None = None
    error: str | None = None
    resolved_url: str | None = None  # child sitemap actually read, when url is an index
    items: list[Item] = field(default_factory=list)


def is_article_sitemap(url: str) -> bool:
    """News/today sitemaps, excluding image/video/web-story ones that list media pages."""
    low = url.lower()
    return any(k in low for k in SITEMAP_KEYWORDS) and not NON_ARTICLE_SITEMAP.search(low)


def parse_datetime(value: str | None) -> datetime | None:
    """ISO-8601 → aware UTC datetime. Naive timestamps from Indian sites are taken as IST."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt.astimezone(timezone.utc)


def clean_text(value: str | None, limit: int = 500) -> str:
    text = html.unescape(TAG_RE.sub(" ", value or ""))
    return " ".join(text.split())[:limit]


def parse_feed(content: bytes, kind: str) -> list[Item]:
    items = []
    # a UTF-8 BOM or whitespace before <?xml makes feedparser drop everything (seen on pib.gov.in)
    content = content.removeprefix(UTF8_BOM).lstrip()
    for e in feedparser.parse(content).entries:
        link = e.get("link")
        if not link:
            continue
        t = e.get("published_parsed") or e.get("updated_parsed")
        title = clean_text(e.get("title"), 1000)
        if kind == "gnews":
            # Google News titles are "Headline - Outlet Name"
            source = (e.get("source") or {}).get("title")
            if source and title.endswith(f" - {source}"):
                title = title[: -len(source) - 3]
        items.append(
            Item(
                url=link,
                title=title,
                published=datetime.fromtimestamp(timegm(t), timezone.utc) if t else None,
                summary="" if kind == "gnews" else clean_text(e.get("summary")),
            )
        )
    return items


def parse_sitemap_urls(root: etree._Element) -> list[Item]:
    items = []
    for u in root.iter("{*}url"):
        loc = u.find("{*}loc")
        if loc is None or not loc.text:
            continue
        pub = u.find(".//{*}publication_date")
        lastmod = u.find("{*}lastmod")
        title = u.find(".//{*}title")
        stamp = pub.text if pub is not None else (lastmod.text if lastmod is not None else None)
        items.append(
            Item(
                url=loc.text.strip(),
                title=clean_text(title.text if title is not None else "", 1000),
                published=parse_datetime(stamp),
            )
        )
    return items


def pick_child_sitemap(root: etree._Element) -> str | None:
    """From a sitemap index, choose the most recently modified news/today child."""
    children = []
    for sm in root.iter("{*}sitemap"):
        loc = sm.find("{*}loc")
        if loc is None or not loc.text:
            continue
        lastmod = sm.find("{*}lastmod")
        children.append((loc.text.strip(), parse_datetime(lastmod.text if lastmod is not None else None)))
    if not children:
        return None
    preferred = [c for c in children if is_article_sitemap(c[0])] or [
        c for c in children if not NON_ARTICLE_SITEMAP.search(c[0])
    ]
    if not preferred:
        return None
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    dated = [c for c in preferred if c[1]]
    return max(dated, key=lambda c: c[1] or epoch)[0] if dated else preferred[0][0]


def parse_xml(content: bytes) -> etree._Element | None:
    try:
        return etree.fromstring(content, parser=etree.XMLParser(recover=True, huge_tree=True))
    except etree.XMLSyntaxError:
        return None


async def fetch_source(fetcher: Fetcher, kind: str, url: str, ua: str = BOT_UA) -> SourceResult:
    res = SourceResult(url=url, kind=kind)
    resp = await fetcher.get(url, ua)
    if resp is None:
        res.error = "request failed"
        return res
    res.status = resp.status_code
    if resp.status_code != 200:
        res.error = f"HTTP {resp.status_code}"
        return res

    if kind in ("rss", "gnews"):
        res.items = parse_feed(resp.content, kind)
    else:
        root = parse_xml(resp.content)
        if root is not None and etree.QName(root).localname == "sitemapindex":
            child = pick_child_sitemap(root)
            if child is None:
                res.error = "empty sitemap index"
                return res
            res.resolved_url = child
            resp = await fetcher.get(child, ua)
            if resp is None or resp.status_code != 200:
                res.error = f"child sitemap {'failed' if resp is None else resp.status_code}"
                return res
            root = parse_xml(resp.content)
        if root is None:
            res.error = "not xml"
            return res
        res.items = parse_sitemap_urls(root)

    if not res.items:
        res.error = "no items"
        return res
    res.ok = True
    return res
