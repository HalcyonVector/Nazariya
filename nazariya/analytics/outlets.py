"""Outlet-level models: the outlet map, who leads and who follows, wire dependence, who copies whom,
and headline edits."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import median

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import orthogonal_procrustes

from .data import Facts

MIN_EVENT_OUTLETS = 3     # events shared by fewer outlets say little about similarity
MIN_PAIR_EVENTS = 15      # pairs of outlets with fewer shared events get no lead-lag cell
BOOTSTRAP = 40
RNG = np.random.default_rng(7)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ---------------------------------------------------------------- outlet map

def _ca(X: np.ndarray, k: int):
    """Correspondence analysis of a non-negative matrix: row and column principal coordinates and
    the share of inertia per dimension."""
    P = X / X.sum()
    r, c = P.sum(1), P.sum(0)
    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    rows = (U[:, :k] * s[:k]) / np.sqrt(r)[:, None]
    cols = (Vt[:k].T * s[:k]) / np.sqrt(c)[:, None]
    inertia = s ** 2
    return rows, cols, inertia[:k] / inertia.sum()


def outlet_map(f: Facts) -> dict:
    evs = [e for e in f.events.values() if e["n_news_outlets"] >= MIN_EVENT_OUTLETS]
    ids = sorted({o for e in evs for o in f.cov[e["event_id"]]})
    if len(evs) < 50 or len(ids) < 5:
        return {"ready": False, "reason": f"{len(evs)} shared events so far; the map needs at least 50"}
    oi = {o: i for i, o in enumerate(ids)}
    X = np.zeros((len(ids), len(evs)))
    for j, e in enumerate(evs):
        for o in f.cov[e["event_id"]]:
            X[oi[o], j] = 1
    keep = X.sum(1) >= 10
    ids = [o for o, k in zip(ids, keep) if k]
    X = X[keep]
    K = min(5, len(ids) - 1)
    rows, cols, share = _ca(X, K)

    # stability: re-run on bootstrap samples of events, rotate each onto the full solution
    boots = []
    for _ in range(BOOTSTRAP):
        idx = RNG.integers(0, X.shape[1], X.shape[1])
        Xb = X[:, idx]
        if (Xb.sum(1) == 0).any():
            continue
        rb, _, _ = _ca(Xb, K)
        R, _ = orthogonal_procrustes(rb[:, :2], rows[:, :2])
        boots.append(rb[:, :2] @ R)
    spread = np.std(np.stack(boots), axis=0) if boots else np.zeros((len(ids), 2))

    Z = linkage(rows, "ward")
    n_clusters = min(4, len(ids) // 3) or 1
    labels = fcluster(Z, n_clusters, "maxclust")

    # what the axes mean: the events at each end of each axis
    def poles(d: int) -> dict:
        order = np.argsort(cols[:, d])
        pick = lambda js: [evs[j]["title"] for j in js if evs[j]["title"]][:5]
        return {"low": pick(order[:40]), "high": pick(order[::-1][:40])}

    dist = np.linalg.norm(rows[:, None, :] - rows[None, :, :], axis=2)
    points = []
    for i, o in enumerate(ids):
        nb = [ids[j] for j in np.argsort(dist[i]) if j != i][:3]
        points.append({
            "id": o, "name": f.outlets[o].name, "lang": f.outlets[o].lang, "group": f.outlets[o].group,
            "x": round(float(rows[i, 0]), 4), "y": round(float(rows[i, 1]), 4),
            "sx": round(float(spread[i, 0]), 4), "sy": round(float(spread[i, 1]), 4),
            "cluster": int(labels[i]), "events": int(X[i].sum()),
            "neighbours": [{"id": n, "name": f.outlets[n].name, "d": round(float(dist[i, ids.index(n)]), 3)} for n in nb],
        })
    return {"ready": True, "events": len(evs), "outlets": len(ids), "bootstrap": len(boots),
            "inertia": [round(float(x), 3) for x in share[:2]], "axes": [poles(0), poles(1)], "points": points}


# ---------------------------------------------------------------- lead-lag

def lead_lag(f: Facts) -> dict:
    """Who publishes first on events several outlets covered. Only an outlet's own (non-wire) dated
    articles count, so reposting agency copy quickly is not leading."""
    excluded = _undated_outlets(f)
    pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    ahead, total = Counter(), Counter()
    n_ev = 0
    for e in f.events.values():
        times = {o: c["first_own_at"] for o, c in f.cov[e["event_id"]].items()
                 if c["first_own_at"] is not None and o not in excluded}
        if len(times) < MIN_EVENT_OUTLETS:
            continue
        n_ev += 1
        med = median(t.timestamp() for t in times.values())
        for o, t in times.items():
            total[o] += 1
            ahead[o] += t.timestamp() < med
        items = sorted(times.items())
        for i, (a, ta) in enumerate(items):
            for b, tb in items[i + 1:]:
                pair[(a, b)].append((tb - ta).total_seconds() / 60)  # positive: a first
    ids = sorted(o for o in total if total[o] >= MIN_PAIR_EVENTS)
    cells = []
    for (a, b), ds in pair.items():
        if a in ids and b in ids and len(ds) >= MIN_PAIR_EVENTS:
            m = median(ds)
            cells.append({"row": a, "col": b, "minutes": round(m), "n": len(ds),
                          "share_first": round(sum(d > 0 for d in ds) / len(ds), 3)})
            cells.append({"row": b, "col": a, "minutes": -round(m), "n": len(ds),
                          "share_first": round(sum(d < 0 for d in ds) / len(ds), 3)})
    rank = []
    for o in ids:
        lo, hi = wilson(ahead[o], total[o])
        rank.append({"id": o, "name": f.outlets[o].name, "group": f.outlets[o].group,
                     "score": round(ahead[o] / total[o], 3), "lo": round(lo, 3), "hi": round(hi, 3), "n": total[o]})
    rank.sort(key=lambda r: -r["score"])
    return {"ready": n_ev >= 30 and len(ids) >= 4, "events": n_ev, "outlets": ids, "cells": cells, "rank": rank,
            "excluded": [{"id": o, "name": f.outlets[o].name} for o in sorted(excluded)]}


def _undated_outlets(f: Facts) -> set[str]:
    """Outlets whose feeds give dates without times (most stamps exactly at midnight) cannot be timed."""
    midnight, n = Counter(), Counter()
    for c in (c for ev in f.cov.values() for c in ev.values()):
        if c["first_own_at"] is not None:
            n[c["outlet_id"]] += 1
            t = c["first_own_at"]
            midnight[c["outlet_id"]] += (t.hour, t.minute, t.second) in ((0, 0, 0), (18, 30, 0))  # 00:00 IST = 18:30 UTC
    return {o for o in n if n[o] >= 20 and midnight[o] / n[o] > 0.5}


# ---------------------------------------------------------------- wire dependence and copying

def wire(f: Facts) -> dict:
    # agency bylines and copied text can only be detected in articles whose text was extracted, so
    # the share is over those; headline-only articles say nothing either way
    arts, wired = Counter(), Counter()
    for ev in f.cov.values():
        for o, c in ev.items():
            arts[o] += c.get("n_text") or 0
            wired[o] += c["n_wire"]
    by_agency: dict[str, Counter] = defaultdict(Counter)
    for c in f.copies:
        by_agency[c["outlet_id"]][c["agency"] or "other outlets"] += 1
    rows = []
    for o in arts:
        if arts[o] < 50:
            continue
        lo, hi = wilson(arts[o] - wired[o], arts[o])
        rows.append({"id": o, "name": f.outlets[o].name, "group": f.outlets[o].group, "articles": arts[o],
                     "own": round(1 - wired[o] / arts[o], 3), "lo": round(lo, 3), "hi": round(hi, 3),
                     "sources": dict(by_agency[o].most_common(5))})
    rows.sort(key=lambda r: -r["own"])
    return {"ready": len(rows) >= 4, "rows": rows}


def copy_network(f: Facts) -> dict:
    """Outlet-to-outlet copying that is not credited agency copy: the later article's text is a
    near-duplicate (SimHash) of an earlier article from another outlet."""
    edges = Counter((c["source_outlet_id"], c["outlet_id"]) for c in f.copies
                    if not c["agency"] and c["source_outlet_id"] in f.outlets and c["outlet_id"] in f.outlets)
    vol = Counter()
    for ev in f.cov.values():
        for o, c in ev.items():
            vol[o] += c["n_articles"]
    out = [{"source": s, "target": t, "n": n, "share_of_target": round(n / vol[t], 4) if vol[t] else None}
           for (s, t), n in edges.most_common() if n >= 3]
    nodes = sorted({e["source"] for e in out} | {e["target"] for e in out})
    return {"ready": bool(out), "edges": out,
            "nodes": [{"id": o, "name": f.outlets[o].name, "group": f.outlets[o].group, "articles": vol[o]} for o in nodes]}


def headline_edits(f: Facts, limit: int = 60) -> dict:
    vol = Counter()
    for ev in f.cov.values():
        for o, c in ev.items():
            vol[o] += c["n_articles"]
    per = Counter(e["outlet_id"] for e in f.edits if e["outlet_id"] in f.outlets)
    rate = [{"id": o, "name": f.outlets[o].name, "edits": n, "articles": vol[o],
             "per_1000": round(1000 * n / vol[o], 1) if vol[o] else None} for o, n in per.most_common()]
    recent = [{"outlet": f.outlets[e["outlet_id"]].name, "outlet_id": e["outlet_id"], "url": e["url"],
               "old": e["old_title"], "new": e["new_title"], "seen_at": e["seen_at"].isoformat(),
               "minutes_after": round((e["seen_at"] - e["published_at"]).total_seconds() / 60)
               if e["published_at"] else None}
              for e in f.edits if e["outlet_id"] in f.outlets][:limit]
    return {"ready": bool(recent), "total": len(f.edits), "by_outlet": rate, "recent": recent}
