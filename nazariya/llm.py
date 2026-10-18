"""Multi-provider JSON chat client (Groq, Cerebras, Gemini via OpenAI-compatible APIs; Ollama local).

A Labeller wraps one model served by an ordered list of hosts (config/llm.yaml). Per-minute rate
limits are waited out; when a host reports a daily quota is spent, the next host takes over.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import httpx
import yaml

from .config import CONFIG, load_dotenv


class DailyQuotaExhausted(Exception):
    pass


class AllHostsExhausted(Exception):
    pass


def parse_reset(value: str | None) -> float:
    """Reset headers look like '727ms', '7.5s', '1m26.4s'."""
    if not value:
        return 0.0
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", value):
        total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total


def is_daily_limit(text: str) -> bool:
    t = text.lower()
    # Groq: "tokens per day (TPD)"; Gemini: "...PerDay..." quota ids; others: "daily"
    return any(k in t for k in ("per day", "perday", "tpd", "rpd", "daily"))


@dataclass
class Host:
    provider: str
    model: str
    kind: str
    base_url: str
    key: str | None
    params: dict


class Labeller:
    def __init__(self, name: str, timeout: float = 180):
        load_dotenv()
        cfg = yaml.safe_load((CONFIG / "llm.yaml").read_text(encoding="utf-8"))
        self.name = name
        self.hosts: list[Host] = []
        for entry in cfg["labellers"][name]:
            prov = cfg["providers"][entry["provider"]]
            key = os.environ.get(prov["key_env"]) if prov.get("key_env") else None
            if prov.get("key_env") and not key:
                continue  # no key configured: skip this host
            self.hosts.append(Host(entry["provider"], entry["model"], prov["kind"], prov["base_url"].rstrip("/"),
                                   key, entry.get("params") or {}))
        if not self.hosts:
            raise SystemExit(f"labeller {name!r}: no usable host (missing API keys in .env?)")
        self.client = httpx.Client(timeout=timeout)
        self.current = 0
        self.min_interval = 0.0
        self.last_call = 0.0

    @property
    def host(self) -> Host:
        return self.hosts[self.current]

    def chat_json(self, system: str, user: str, max_tokens: int = 1500) -> tuple[dict, Host]:
        """Return (parsed JSON object, host that answered)."""
        while self.current < len(self.hosts):
            try:
                return self._call(self.host, system, user, max_tokens), self.host
            except DailyQuotaExhausted as e:
                print(f"  {self.host.provider}/{self.host.model}: daily quota spent ({e}); trying next host")
                self.current += 1
        raise AllHostsExhausted(f"labeller {self.name!r}: every host's daily quota is spent")

    def _call(self, h: Host, system: str, user: str, max_tokens: int) -> dict:
        wait = self.min_interval - (time.monotonic() - self.last_call)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(8):
            self.last_call = time.monotonic()
            if h.kind == "ollama":
                r = self.client.post(f"{h.base_url}/api/chat", json={
                    "model": h.model, "stream": False, "format": "json",
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "options": {"temperature": 0, "num_predict": max_tokens}, **h.params,
                })
            else:
                r = self.client.post(f"{h.base_url}/chat/completions", headers={"Authorization": f"Bearer {h.key}"}, json={
                    "model": h.model, "temperature": 0, "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    **h.params,
                })
            if r.status_code == 429:
                if is_daily_limit(r.text):
                    raise DailyQuotaExhausted(r.text[:160].replace("\n", " "))
                delay = float(r.headers.get("retry-after") or 0) or parse_reset(r.headers.get("x-ratelimit-reset-tokens")) or 2 ** attempt
                time.sleep(min(delay + 0.5, 90))
                continue
            if r.status_code >= 500 or r.status_code == 408:
                time.sleep(min(5 * (attempt + 1), 60))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"{h.provider} HTTP {r.status_code}: {r.text[:300]}")
            # pace proactively when the provider tells us the per-minute token budget is low
            remaining = r.headers.get("x-ratelimit-remaining-tokens")
            if remaining is not None and int(remaining) < 2500:
                time.sleep(parse_reset(r.headers.get("x-ratelimit-reset-tokens")) + 0.5)
            data = r.json()
            content = data["message"]["content"] if h.kind == "ollama" else data["choices"][0]["message"]["content"]
            return json.loads(content)
        raise RuntimeError(f"{h.provider}: gave up after retries")
