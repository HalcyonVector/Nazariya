"""Load the fact tables into plain Python/numpy structures for the models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.engine import Engine

from ..config import CONFIG
from ..db import as_utc, copies, coverage, entities, event_facts, outlets, releases, removals, title_edits


@dataclass
class Outlet:
    id: str
    name: str
    lang: str
    region: str
    group: str  # "English national" | "English regional" | "Hindi"


@dataclass
class Facts:
    now: datetime
    since: datetime
    outlets: dict[str, Outlet]
    events: dict[int, dict]                      # event_id -> event_facts row
    cov: dict[int, dict[str, dict]]              # event_id -> outlet_id -> coverage row
    copies: list[dict] = field(default_factory=list)
    edits: list[dict] = field(default_factory=list)
    releases: list[dict] = field(default_factory=list)
    rechecks: list[dict] = field(default_factory=list)
    entity_names: dict[str, str] = field(default_factory=dict)

    @property
    def days(self) -> float:
        return (self.now - self.since).total_seconds() / 86400


def group_of(lang: str, region: str) -> str:
    if lang == "hi":
        return "Hindi"
    if lang != "en":
        return "Regional language"
    return "English national" if region == "national" else "English regional"


def load_owners() -> dict[str, dict]:
    return yaml.safe_load((CONFIG / "owners.yaml").read_text(encoding="utf-8"))["owners"]


def load_states() -> list[dict]:
    return yaml.safe_load((CONFIG / "states.yaml").read_text(encoding="utf-8"))["states"]


def load(engine: Engine, days: float = 28, now: datetime | None = None) -> Facts:
    """Facts for events that started in the last `days` days, news outlets only (the PIB and ANI
    reference feeds are kept as event features, not as outlets)."""
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    with engine.connect() as conn:
        outs = {r.id: Outlet(r.id, r.name, r.lang or "en", r.region or "national", group_of(r.lang, r.region))
                for r in conn.execute(select(outlets).where(outlets.c.role == "outlet"))}
        evs = {}
        for r in conn.execute(select(event_facts).where(event_facts.c.first_seen >= since)).mappings():
            d = dict(r)
            d["first_seen"], d["last_seen"] = as_utc(d["first_seen"]), as_utc(d["last_seen"])
            evs[d["event_id"]] = d
        cov: dict[int, dict[str, dict]] = {}
        for r in conn.execute(select(coverage).join(event_facts, event_facts.c.event_id == coverage.c.event_id)
                              .where(event_facts.c.first_seen >= since)).mappings():
            if r["outlet_id"] in outs:
                d = dict(r)
                d["first_at"], d["first_own_at"] = as_utc(d["first_at"]), as_utc(d["first_own_at"])
                cov.setdefault(d["event_id"], {})[d["outlet_id"]] = d
        cps = [dict(r) for r in conn.execute(select(copies).where(copies.c.published_at >= since)).mappings()]
        eds = [dict(r) for r in conn.execute(select(title_edits).where(title_edits.c.seen_at >= since)
                                             .order_by(title_edits.c.seen_at.desc())).mappings()]
    from ..ingest import _norm_title
    # edits stored before punctuation-only differences were ignored (e.g. "News" -> "News:")
    from ..textutil import is_live_page
    eds = [e for e in eds if _norm_title(e["old_title"]) != _norm_title(e["new_title"])
           and not is_live_page(e["url"], e["old_title"], e["new_title"])]  # live blogs change by design
    for e in eds:
        e["seen_at"], e["published_at"] = as_utc(e["seen_at"]), as_utc(e["published_at"])
    with engine.connect() as conn:
        rels = [dict(r) for r in conn.execute(select(releases).where(releases.c.published_at >= since)).mappings()]
    for r in rels:
        r["published_at"] = as_utc(r["published_at"])
    with engine.connect() as conn:
        chk = [dict(r) for r in conn.execute(select(removals).where(removals.c.last_checked_at >= since)).mappings()]
    for r in chk:
        for k in ("published_at", "first_gone_at", "confirmed_at", "last_checked_at"):
            r[k] = as_utc(r[k])
    # events no news outlet covered (only PIB/ANI) are not news coverage
    evs = {k: v for k, v in evs.items() if k in cov}
    for e in evs.values():
        e["n_news_outlets"] = len(cov[e["event_id"]])
    with engine.connect() as conn:
        names = {r.qid: r.label_en for r in conn.execute(select(entities.c.qid, entities.c.label_en))}
    return Facts(now, since, outs, evs, cov, cps, eds, rels, chk, names)
