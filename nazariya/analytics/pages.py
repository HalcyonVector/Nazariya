"""Per-story and per-outlet pages, weekly outlet positions, and frozen weekly editions.

  site/data/stories/<event_id>.json   every event 3+ outlets covered in the window: each outlet's
                                      first article (linked), timing, wire copy, headline edits,
                                      and the outlets the selection model expected but which
                                      stayed silent
  site/data/stories/index.json        those events, newest first
  site/data/outlets/<outlet_id>.json  an outlet's dossier: map position and its history, publishing
                                      clock, states it over- and under-covers, big stories it
                                      skipped, stories it broke, copying, headline edits
  site/data/editions/<issue>.json     the front page as frozen each Monday, kept for good

Article text is never written; only headlines, links and derived numbers.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sqlalchemy import select
from sqlalchemy.engine import Engine

from ..config import ROOT
from ..db import as_utc, editions, outlet_positions, upsert
from . import selection
from .data import Facts, load_owners, load_states

DATA = ROOT / "site" / "data"
IST = timezone(timedelta(hours=5, minutes=30))
STORY_MIN_OUTLETS = 3
SILENT_MIN_P = 0.3
SKIP_MIN_P = 0.25  # "big stories it skipped": at least a 1 in 4 modelled chance of covering
REFERENCE = {"ani": "ANI (news agency)", "pib": "PIB (government press releases)"}


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")


def _fresh_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


# ---------------------------------------------------------------- stories

def stories(f: Facts) -> dict:
    states = {s["qid"]: s["name"] for s in load_states()}
    edits_by_article = defaultdict(list)
    for e in f.edits:
        edits_by_article[e["article_id"]].append({"old": e["old_title"], "new": e["new_title"], "seen_at": _iso(e["seen_at"])})
    ids = sorted((eid for eid, e in f.events.items() if e["n_news_outlets"] >= STORY_MIN_OUTLETS),
                 key=lambda eid: f.events[eid]["first_seen"], reverse=True)
    # under- and over-covered events get pages too, whatever their size
    ids += [eid for eid in _flagged(f) if eid not in set(ids)]
    probs = selection.predict(f, ids)
    out = _fresh_dir(DATA / "stories")
    index = []
    for eid in ids:
        e = f.events[eid]
        cov = f.cov[eid]
        first = (e.get("first_report") or {}).get("outlet")
        p = probs.get(eid, {})
        outlets = []
        for o, c in sorted(cov.items(), key=lambda kv: kv[1]["first_at"]):
            outlets.append({
                "id": o, "name": f.outlets[o].name, "group": f.outlets[o].group,
                "first_at": _iso(c["first_at"]), "first_own_at": _iso(c["first_own_at"]),
                "n": c["n_articles"], "wire": c["n_wire"], "url": c.get("first_url"), "title": c.get("first_title"),
                "edits": edits_by_article.get(c.get("first_article_id"), []), "p": round(p[o], 3) if o in p else None,
            })
        silent = sorted(({"id": o, "name": f.outlets[o].name, "group": f.outlets[o].group, "p": round(v, 3)}
                         for o, v in p.items() if o not in cov and v >= SILENT_MIN_P), key=lambda r: -r["p"])
        fr = e.get("first_report") or {}
        story = {
            "id": eid, "title": e["title"], "first_seen": _iso(e["first_seen"]), "last_seen": _iso(e["last_seen"]),
            "n_articles": e["n_articles"], "n_outlets": e["n_news_outlets"], "langs": e.get("langs") or {},
            "states": [states.get(q, q) for q in e.get("states") or []], "toll": e.get("toll"), "frame": e.get("frame"),
            "lang_drift": e.get("lang_drift"), "broke_in": f.outlets[first].name if first in f.outlets else REFERENCE.get(first, first),
            "broke_as": "a government press release" if fr.get("govt") else "agency copy" if fr.get("wire") else None,
            "expected": round(sum(p.values()), 1) if p else None, "outlets": outlets, "silent": silent,
        }
        _write(out / f"{eid}.json", story)
        index.append({"id": eid, "title": e["title"], "first_seen": story["first_seen"], "n_outlets": story["n_outlets"],
                      "langs": story["langs"]})
    _write(out / "index.json", index)
    return {"stories": len(index)}


def _flagged(f: Facts) -> list[int]:
    s = selection.LAST
    if not s.get("models"):
        return []
    res = {}
    for under in (True, False):
        for r in selection._surprises(f, s["X"], s["models"], under=under):
            res[r["event_id"]] = True
    return list(res)


# ---------------------------------------------------------------- outlet positions (weekly)

def record_positions(engine: Engine, f: Facts, result: dict) -> dict:
    """Store this week's map positions, rotated onto last week's so the trail is comparable."""
    m = result.get("outlet_map", {})
    if not m.get("ready"):
        return {"positions": 0}
    week = _week(f.now)
    pts = {p["id"]: p for p in m["points"]}
    with engine.connect() as conn:
        prev = conn.execute(select(outlet_positions.c.week).where(outlet_positions.c.week < week)
                            .order_by(outlet_positions.c.week.desc()).limit(1)).scalar()
        ref = {r.outlet_id: (r.x, r.y) for r in conn.execute(select(outlet_positions).where(outlet_positions.c.week == prev))} if prev else {}
    common = [o for o in pts if o in ref]
    xy = np.array([[pts[o]["x"], pts[o]["y"]] for o in pts])
    if len(common) >= 4:
        A = np.array([[pts[o]["x"], pts[o]["y"]] for o in common])
        B = np.array([ref[o] for o in common])
        R, _ = orthogonal_procrustes(A - A.mean(0), B - B.mean(0))
        xy = (xy - A.mean(0)) @ R + B.mean(0)
    rows = [{"week": week, "outlet_id": o, "x": float(x), "y": float(y), "events": pts[o]["events"]}
            for o, (x, y) in zip(pts, xy)]
    with engine.begin() as conn:
        upsert(conn, outlet_positions, rows, ["week", "outlet_id"])
    return {"positions": len(rows), "week": week}


def _week(dt: datetime) -> str:
    y, w, _ = dt.astimezone(IST).isocalendar()
    return f"{y}-W{w:02d}"


# ---------------------------------------------------------------- outlet dossiers

def dossiers(engine: Engine, f: Facts, result: dict) -> dict:
    out = _fresh_dir(DATA / "outlets")
    states = load_states()
    sname = {s["qid"]: s for s in states}
    with engine.connect() as conn:
        hist = defaultdict(list)
        for r in conn.execute(select(outlet_positions).order_by(outlet_positions.c.week)):
            hist[r.outlet_id].append({"week": r.week, "x": round(r.x, 4), "y": round(r.y, 4)})
    pts = {p["id"]: p for p in result.get("outlet_map", {}).get("points", [])}
    wire = {r["id"]: r for r in result.get("wire", {}).get("rows", [])}
    lead = {r["id"]: r for r in result.get("lead_lag", {}).get("rank", [])}
    edits_by = defaultdict(list)
    for e in f.edits:
        edits_by[e["outlet_id"]].append({"url": e["url"], "old": e["old_title"], "new": e["new_title"], "seen_at": _iso(e["seen_at"]),
                                         "minutes_after": round((e["seen_at"] - e["published_at"]).total_seconds() / 60) if e["published_at"] else None})
    copy_in, copy_out = defaultdict(Counter), defaultdict(Counter)
    for c in f.copies:
        if not c["agency"]:
            copy_in[c["outlet_id"]][c["source_outlet_id"]] += 1
            copy_out[c["source_outlet_id"]][c["outlet_id"]] += 1

    # how much of each outlet's state-linked coverage goes to each state, against all outlets
    state_cov, total_cov = defaultdict(Counter), Counter()
    for eid, e in f.events.items():
        for q in e.get("states") or []:
            for o in f.cov[eid]:
                state_cov[o][q] += 1
                total_cov[q] += 1
    tot_all = sum(total_cov.values())

    # big stories each outlet skipped: last week's events with a high modelled probability for it
    recent = [eid for eid, e in f.events.items() if (f.now - e["first_seen"]).days < 7]
    probs = selection.predict(f, recent)
    skipped = defaultdict(list)
    for eid, ps in probs.items():
        for o, pv in ps.items():
            if o not in f.cov[eid] and pv >= SKIP_MIN_P:
                skipped[o].append({"id": eid, "title": f.events[eid]["title"], "p": round(pv, 3),
                                   "n_outlets": f.events[eid]["n_news_outlets"], "page": f.events[eid]["n_news_outlets"] >= STORY_MIN_OUTLETS})
    broke = defaultdict(list)
    for eid, e in f.events.items():
        fo = (e.get("first_report") or {}).get("outlet")
        if fo in f.outlets and e["n_news_outlets"] >= STORY_MIN_OUTLETS:
            broke[fo].append({"id": eid, "title": e["title"], "n_outlets": e["n_news_outlets"], "first_seen": _iso(e["first_seen"])})

    owners = load_owners()
    n = 0
    for o, meta in f.outlets.items():
        rows = [ev[o] for ev in f.cov.values() if o in ev]
        if not rows:
            continue
        hours = Counter(r["first_at"].astimezone(IST).hour for r in rows)
        days = Counter(r["first_at"].astimezone(IST).weekday() for r in rows)
        mine = sum(state_cov[o].values())
        st = []
        for q, c in state_cov[o].items():
            if q in sname and total_cov[q] >= 10:
                st.append({"code": sname[q]["code"], "name": sname[q]["name"], "n": c,
                           "ratio": round((c / mine) / (total_cov[q] / tot_all), 3)})
        st.sort(key=lambda r: -r["ratio"])
        own = owners.get(o) or {}
        doss = {
            "id": o, "name": meta.name, "group": meta.group, "lang": meta.lang, "region": meta.region,
            "owner": own.get("owner"), "owner_group": own.get("group"), "owner_source": own.get("source"),
            "siblings": [{"id": k, "name": f.outlets[k].name} for k, v in owners.items()
                         if k != o and k in f.outlets and own.get("group") and v.get("group") == own.get("group")],
            "events": len(rows), "articles": sum(r["n_articles"] for r in rows),
            "own": wire.get(o, {}).get("own"), "first_score": lead.get(o, {}).get("score"),
            "position": {"x": pts[o]["x"], "y": pts[o]["y"], "neighbours": pts[o]["neighbours"]} if o in pts else None,
            "history": hist.get(o, []), "clock": [hours.get(h, 0) for h in range(24)], "weekdays": [days.get(d, 0) for d in range(7)],
            "states_over": st[:6], "states_under": [r for r in st[::-1] if r["ratio"] < 1][:6],
            "skipped": sorted(skipped[o], key=lambda r: -r["p"])[:12],
            "broke": sorted(broke[o], key=lambda r: -r["n_outlets"])[:12],
            "copies_from": [{"id": s, "name": f.outlets[s].name, "n": c} for s, c in copy_in[o].most_common(6) if s in f.outlets],
            "copied_by": [{"id": t, "name": f.outlets[t].name, "n": c} for t, c in copy_out[o].most_common(6) if t in f.outlets],
            "edits": edits_by[o][:15], "n_edits": len(edits_by[o]),
        }
        _write(out / f"{o}.json", doss)
        n += 1
    # every outlet's map position, for the small map on each dossier
    _write(out / "index.json", [{"id": o, "name": m.name, "group": m.group,
                                 "x": pts[o]["x"] if o in pts else None, "y": pts[o]["y"] if o in pts else None}
                                for o, m in sorted(f.outlets.items(), key=lambda kv: kv[1].name)])
    return {"dossiers": n}


# ---------------------------------------------------------------- weekly editions

def editions_step(engine: Engine, result: dict, now: datetime) -> dict:
    """On the first run of each Monday (IST), freeze the front page as that week's issue. Every
    issue is then written out on every run, so the archive survives the site being rebuilt."""
    local = now.astimezone(IST)
    issue = _week(now - timedelta(days=1))  # the week that just ended
    with engine.begin() as conn:
        have = conn.execute(select(editions.c.issue)).scalars().all()
        made = None
        if local.weekday() == 0 and issue not in have and result.get("events"):
            number = len(have) + 1
            body = dict(result, issue={"key": issue, "number": number, "published_at": now.isoformat()})
            conn.execute(editions.insert().values(issue=issue, number=number, published_at=now,
                                                  body=json.dumps(body, ensure_ascii=False, default=str)))
            made = issue
        rows = conn.execute(select(editions).order_by(editions.c.number.desc())).all()
    out = _fresh_dir(DATA / "editions")
    for r in rows:
        (out / f"{r.issue}.json").write_text(r.body, encoding="utf-8")
    _write(out / "index.json", [{"issue": r.issue, "number": r.number, "published_at": _iso(as_utc(r.published_at))} for r in rows])
    return {"editions": len(rows), "new": made}
