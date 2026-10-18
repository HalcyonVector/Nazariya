"""Shared HTTP layer: polite fetching, per-outlet user agents, robots.txt checks."""

from __future__ import annotations

import asyncio
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

BOT_UA = "NazariyaBot/0.1 (research project; coverage analysis; respects robots.txt)"
# Only for outlets whose CDN rejects any non-browser UA (set `user_agent: browser` in
# config/outlets.yaml). robots.txt rules are still obeyed for our bot token.
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)
ROBOTS_TOKEN = "NazariyaBot"


def user_agent_for(outlet: dict) -> str:
    return BROWSER_UA if outlet.get("user_agent") == "browser" else BOT_UA


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"Accept-Language": "en-IN,hi;q=0.8"},
        follow_redirects=True,
        timeout=httpx.Timeout(20.0),
        limits=httpx.Limits(max_connections=24),
    )


class Fetcher:
    """GETs with a global concurrency cap, a per-host cap, one retry, and a robots cache."""

    def __init__(self, client: httpx.AsyncClient, global_limit: int = 16, per_host: int = 2):
        self.client = client
        self.sem = asyncio.Semaphore(global_limit)
        self.per_host = per_host
        self.host_sems: dict[str, asyncio.Semaphore] = {}
        self.robots: dict[tuple[str, str], tuple[RobotFileParser, list[str]]] = {}
        self.robots_locks: dict[tuple[str, str], asyncio.Lock] = {}

    async def get(self, url: str, ua: str = BOT_UA) -> httpx.Response | None:
        host = urlparse(url).netloc
        host_sem = self.host_sems.setdefault(host, asyncio.Semaphore(self.per_host))
        async with self.sem, host_sem:
            for attempt in range(2):
                try:
                    return await self.client.get(url, headers={"User-Agent": ua})
                except httpx.HTTPError:
                    if attempt == 0:
                        await asyncio.sleep(2)
            return None

    async def robots_for(self, url: str, ua: str = BOT_UA) -> tuple[RobotFileParser, list[str]]:
        """Parsed robots.txt for url's host (fetched with ua) plus its Sitemap: lines.

        Per RFC 9309: 4xx means no restrictions; 5xx or unreachable means disallow all.
        """
        p = urlparse(url)
        key = (p.netloc, ua)
        if key in self.robots:
            return self.robots[key]
        async with self.robots_locks.setdefault(key, asyncio.Lock()):
            if key in self.robots:
                return self.robots[key]
            rp = RobotFileParser()
            sitemaps: list[str] = []
            resp = await self.get(f"{p.scheme}://{p.netloc}/robots.txt", ua)
            if resp is not None and resp.status_code == 200:
                lines = resp.text.splitlines()
                rp.parse(lines)
                sitemaps = [l.split(":", 1)[1].strip() for l in lines if l.lower().startswith("sitemap:")]
            elif resp is not None and 400 <= resp.status_code < 500:
                rp.parse([])
            else:
                rp.parse(["User-agent: *", "Disallow: /"])
            self.robots[key] = (rp, sitemaps)
            return rp, sitemaps

    async def allowed(self, url: str, ua: str = BOT_UA) -> bool:
        rp, _ = await self.robots_for(url, ua)
        return rp.can_fetch(ROBOTS_TOKEN, url)
