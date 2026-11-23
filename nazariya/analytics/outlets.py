"""Outlet-level models: the outlet map, who leads and who follows, wire dependence, who copies whom,
and headline edits."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import median

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import orthogonal_procrustes

from .data import Facts, load_owners

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


RELEASE_MATURE_H = 48  # a release gets two days to be picked up before it is judged


def release_pickup(f: Facts) -> dict:
    """Government press releases (PIB) and how many news outlets covered the event each one landed
    in. Judged only once a release is RELEASE_MATURE_H hours old."""
    mature = [r for r in f.releases if r["published_at"] and (f.now - r["published_at"]).total_seconds() >= RELEASE_MATURE_H * 3600]
    if len(mature) < 30:
        return {"ready": False, "reason": f"{len(mature)} releases are old enough to judge so far; this needs 30"}
    def outlets_for(r):
        return len(f.cov.get(r["event_id"], {})) if r["event_id"] is not None else 0
    by = defaultdict(list)
    for r in mature:
        by[r["ministry"] or "Not stated"].append(outlets_for(r))
    rows = []
    for m, ns in by.items():
        k = sum(n > 0 for n in ns)
        lo, hi = wilson(k, len(ns))
        rows.append({"ministry": m, "releases": len(ns), "reported": round(k / len(ns), 3), "lo": round(lo, 3), "hi": round(hi, 3),
                     "mean_outlets": round(sum(ns) / len(ns), 2)})
    rows.sort(key=lambda r: (r["reported"], -r["releases"]))
    unreported = sorted((r for r in mature if outlets_for(r) == 0), key=lambda r: r["published_at"], reverse=True)
    total_k = sum(outlets_for(r) > 0 for r in mature)
    return {"ready": True, "releases": len(mature), "reported_share": round(total_k / len(mature), 3),
            "ministries": rows,
            "unreported": [{"title": r["title"], "url": r["url"], "ministry": r["ministry"] or "Not stated",
                            "published_at": r["published_at"].isoformat()} for r in unreported[:20]]}


IST_HOURS = 5.5


def newsroom_clock(f: Facts) -> dict:
    """When each outlet publishes across the day (IST), from the first article it put out on each
    story. Outlets whose feeds give dates without times are left out."""
    skip = _undated_outlets(f)
    hours: dict[str, list[int]] = defaultdict(lambda: [0] * 24)
    for ev in f.cov.values():
        for o, c in ev.items():
            if o in skip:
                continue
            h = int(((c["first_at"].hour * 60 + c["first_at"].minute) / 60 + IST_HOURS) % 24)
            hours[o][h] += 1
    rows, partial = [], []
    for o, hs in hours.items():
        n = sum(hs)
        if n < 30:
            continue
        # a newsroom publishes through most of the day; a feed that only ever shows a few hours
        # (sitemaps that list just the latest items) says when we read it, not when they publish
        if sum(1 for h in hs if h) < 16 or max(hs) / n > 0.35:
            partial.append(f.outlets[o].name)
            continue
        night = sum(hs[h] for h in list(range(22, 24)) + list(range(0, 6)))
        peak = max(range(24), key=lambda h: hs[h])
        rows.append({"id": o, "name": f.outlets[o].name, "group": f.outlets[o].group, "n": n, "hours": hs,
                     "night": round(night / n, 3), "peak": peak})
    if len(rows) < 5:
        return {"ready": False, "reason": "Too few timed stories per outlet yet"}
    rows.sort(key=lambda r: -r["night"])
    return {"ready": True, "outlets": rows, "excluded": [f.outlets[o].name for o in sorted(skip)],
            "partial": sorted(partial)}


OWNER_PERMUTATIONS = 2000


def ownership(f: Facts) -> dict:
    """Do outlets with a shared owner cover the same stories more than comparable outlets?

    Similarity of two outlets: Jaccard overlap of the events (3+ outlets) each covered. A shared owner
    often spans languages, and language alone drives what gets covered, so every same-owner pair is
    compared with the pairs of different-owner outlets in the same language combination. Significance:
    owners are shuffled among outlets of the same language and the statistic recomputed."""
    owners = load_owners()
    sets: dict[str, set] = defaultdict(set)
    for eid, e in f.events.items():
        if e["n_news_outlets"] >= MIN_EVENT_OUTLETS:
            for o in f.cov[eid]:
                sets[o].add(eid)
    ids = sorted(o for o in sets if len(sets[o]) >= 10)
    if len(ids) < 8:
        return {"ready": False, "reason": "Too few outlets with 10+ shared stories yet"}
    lang = {o: f.outlets[o].lang for o in ids}
    group = {o: (owners.get(o) or {}).get("group") for o in ids}
    jac = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            u = len(sets[a] | sets[b])
            jac[(a, b)] = len(sets[a] & sets[b]) / u if u else 0.0
    combo = lambda a, b: tuple(sorted((lang[a], lang[b])))
    base = defaultdict(list)  # language combination -> similarities of different-owner pairs

    def stat(g):
        base.clear()
        same = []
        for (a, b), v in jac.items():
            if g[a] and g[a] == g[b]:
                same.append((a, b, v))
            else:
                base[combo(a, b)].append(v)
        if not same:
            return None, []
        excess = [v - np.mean(base[combo(a, b)]) for a, b, v in same if base[combo(a, b)]]
        return float(np.mean(excess)), same

    observed, same = stat(group)
    if observed is None:
        return {"ready": False, "reason": "No two outlets with a shared owner are measured yet"}
    by_lang = defaultdict(list)
    for o in ids:
        by_lang[lang[o]].append(o)
    rng = np.random.default_rng(3)
    null = []
    for _ in range(OWNER_PERMUTATIONS):
        g = {}
        for os_ in by_lang.values():
            labels = [group[o] for o in os_]
            rng.shuffle(labels)
            g.update(zip(os_, labels))
        s_, _ = stat(g)
        if s_ is not None:
            null.append(s_)
    p = (1 + sum(n >= observed for n in null)) / (1 + len(null))
    observed, same = stat(group)  # restore base for the per-group table
    groups = defaultdict(list)
    for a, b, v in same:
        groups[group[a]].append({"a": a, "b": b, "a_name": f.outlets[a].name, "b_name": f.outlets[b].name,
                                 "similarity": round(v, 3), "baseline": round(float(np.mean(base[combo(a, b)])), 3)})
    return {"ready": True, "outlets": len(ids), "excess": round(observed, 4), "p_value": round(p, 4),
            "permutations": len(null), "groups": [{"group": g, "pairs": ps} for g, ps in sorted(groups.items())],
            "owners": {o: {"owner": (owners.get(o) or {}).get("owner"), "group": (owners.get(o) or {}).get("group"),
                           "source": (owners.get(o) or {}).get("source")} for o in f.outlets}}


def removed_stories(f: Facts) -> dict:
    """Articles re-checked a day or more after publishing and found clearly gone twice."""
    checked, gone = Counter(), Counter()
    for r in f.rechecks:
        if r["outlet_id"] in f.outlets:
            checked[r["outlet_id"]] += 1
            gone[r["outlet_id"]] += bool(r["confirmed_at"])
    total = sum(checked.values())
    if total < 200:
        return {"ready": False, "reason": f"{total} articles re-checked so far; this needs 200"}
    rows = []
    for o, n in checked.items():
        if n >= 20:
            lo, hi = wilson(gone[o], n)
            rows.append({"id": o, "name": f.outlets[o].name, "checked": n, "removed": gone[o],
                         "rate": round(gone[o] / n, 4), "lo": round(lo, 4), "hi": round(hi, 4)})
    rows.sort(key=lambda r: -r["rate"])
    recent = sorted((r for r in f.rechecks if r["confirmed_at"] and r["outlet_id"] in f.outlets),
                    key=lambda r: r["confirmed_at"], reverse=True)[:20]
    return {"ready": True, "checked": total, "removed": sum(gone.values()), "by_outlet": rows,
            "recent": [{"outlet": f.outlets[r["outlet_id"]].name, "outlet_id": r["outlet_id"], "title": r["title"],
                        "url": r["url"], "status": r["status"], "published_at": r["published_at"].isoformat() if r["published_at"] else None,
                        "gone_by": r["first_gone_at"].isoformat()} for r in recent]}


NAMED_TOP = 8           # most-named people and parties compared
NAMED_MIN_EXPECTED = 5  # a cell is shown only if the peers' rate predicts 5+ mentions
NAMED_FDR = 0.05        # false-discovery rate across all cells (Benjamini-Hochberg)


def _slug_headlines(f: Facts) -> set[str]:
    """Outlets whose headlines are mostly rebuilt from URL slugs ("Bjp names candidate ..."), where
    names such as "BJP" cannot be recognised; comparing them would measure the feed, not the newsroom."""
    from ..textutil import title_from_url
    slug, n = Counter(), Counter()
    for ev in f.cov.values():
        for o, c in ev.items():
            t, u = c.get("first_title"), c.get("first_url")
            if t and u:
                n[o] += 1
                slug[o] += t == title_from_url(u)
    return {o for o in n if n[o] >= 20 and slug[o] / n[o] > 0.5}


def who_named(f: Facts) -> dict:
    """Which people and parties each outlet puts in its headlines, against the other outlets covering
    the SAME stories.

    Headlines, not article text: every outlet's headline is readable, while article text is missing or
    cut short for some (paywalls, extraction limits), which would make them look as if they name fewer
    people. For the events an outlet covered among those 3+ outlets covered, its count of headlines
    naming X is tested against the rate in its peers' headlines on the same events (exact binomial
    test), with the false-discovery rate controlled across every outlet-name pair. Outlets whose
    headlines come from URL slugs are left out."""
    from scipy.stats import binomtest
    slug = _slug_headlines(f)
    by_event = defaultdict(list)
    for eid, e in f.events.items():
        if e["n_news_outlets"] < MIN_EVENT_OUTLETS:
            continue
        for o, c in f.cov[eid].items():
            if o not in slug and c.get("names") is not None:
                by_event[eid].append((o, set(c["names"])))
    by_event = {k: v for k, v in by_event.items() if len(v) >= 2}
    count = Counter(q for v in by_event.values() for _, ns in v for q in ns)
    top = [q for q, _ in count.most_common(NAMED_TOP)]
    if len(by_event) < 60 or not top:
        return {"ready": False, "reason": f"{len(by_event)} stories from 2+ outlets so far; this needs 60"}
    outs = sorted({o for v in by_event.values() for o, _ in v})
    cells = []
    for o in outs:
        n = 0
        own, peer, n_peer = Counter(), Counter(), 0
        for v in by_event.values():
            mine = [ns for oo, ns in v if oo == o]
            if not mine:
                continue
            others = [ns for oo, ns in v if oo != o]
            n += 1
            n_peer += len(others)
            for q in top:
                own[q] += q in mine[0]
                peer[q] += sum(q in ns for ns in others)
        for q in top:
            rate = peer[q] / n_peer if n_peer else 0.0
            if n * rate < NAMED_MIN_EXPECTED or not 0 < rate < 1:
                continue
            k = own[q]
            lo, hi = wilson(k, n)
            cells.append({"outlet": o, "qid": q, "named": k, "stories": n, "expected": round(n * rate, 1),
                          "share": round(k / n, 3), "peer_share": round(rate, 3), "ratio": round((k / n) / rate, 3),
                          "lo": round(lo / rate, 3), "hi": round(hi / rate, 3),
                          "p": float(binomtest(k, n, rate).pvalue)})
    # Benjamini-Hochberg across all cells
    order = sorted(range(len(cells)), key=lambda i: cells[i]["p"])
    m = len(cells)
    q_prev = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        q_prev = min(q_prev, cells[i]["p"] * m / rank)
        cells[i]["q"] = round(q_prev, 4)
        cells[i]["clear"] = q_prev < NAMED_FDR
        cells[i]["p"] = round(cells[i]["p"], 5)
    return {"ready": True, "stories": len(by_event), "tests": m,
            "names": [{"qid": q, "name": f.entity_names.get(q, q), "stories": count[q]} for q in top],
            "outlets": [{"id": o, "name": f.outlets[o].name, "group": f.outlets[o].group} for o in outs],
            "left_out": [f.outlets[o].name for o in sorted(slug)], "cells": cells}


LANG_GROUP = {"en": "English", "hi": "Hindi"}


def language_bridges(f: Facts) -> dict:
    """How stories cross between English, Hindi and the regional languages, and who carries them over.

    For each event 3+ outlets covered, the language group it broke in (its first report), and for
    every other language group that covered it, the outlet that was first in that group and how long
    after the first report."""
    lg = lambda o: LANG_GROUP.get(f.outlets[o].lang, "Regional languages")
    present = {lg(o) for ev in f.cov.values() for o in ev}  # language groups with any coverage yet
    flows = defaultdict(lambda: {"stories": 0, "crossed": 0, "delays": []})
    bridge = defaultdict(lambda: {"n": 0, "delays": []})
    for eid, e in f.events.items():
        if e["n_news_outlets"] < MIN_EVENT_OUTLETS:
            continue
        cov = f.cov[eid]
        first_o = min(cov, key=lambda o: cov[o]["first_at"])
        src = lg(first_o)
        t0 = cov[first_o]["first_at"]
        groups = defaultdict(list)
        for o, c in cov.items():
            groups[lg(o)].append((c["first_at"], o))
        for tgt in ("English", "Hindi", "Regional languages"):
            if tgt == src or tgt not in present:
                continue
            fl = flows[(src, tgt)]
            fl["stories"] += 1
            if groups.get(tgt):
                t, o = min(groups[tgt])
                d = max(0.0, (t - t0).total_seconds() / 3600)
                fl["crossed"] += 1
                fl["delays"].append(d)
                bridge[o]["n"] += 1
                bridge[o]["delays"].append(d)
    if sum(v["stories"] for v in flows.values()) < 100:
        return {"ready": False, "reason": "Too few stories covered by 3+ outlets yet"}
    flow_rows = [{"from": s, "to": t, "stories": v["stories"], "crossed": v["crossed"],
                  "share": round(v["crossed"] / v["stories"], 3) if v["stories"] else None,
                  "median_hours": round(float(np.median(v["delays"])), 1) if v["delays"] else None}
                 for (s, t), v in sorted(flows.items())]
    bridges = sorted(({"id": o, "name": f.outlets[o].name, "group": f.outlets[o].group, "carried": v["n"],
                       "median_hours": round(float(np.median(v["delays"])), 1)} for o, v in bridge.items() if v["n"] >= 3),
                     key=lambda r: -r["carried"])
    return {"ready": True, "flows": flow_rows, "bridges": bridges[:15]}


AGENDA_FRAMES = 8


def agenda(f: Facts) -> dict:
    """Each outlet's mix of story frames against the mix of all outlets: what it spends its
    coverage on. Frame: the most common predicted frame among an event's articles."""
    per = defaultdict(Counter)
    for eid, e in f.events.items():
        fr = e.get("frame")
        if not fr or fr == "other":
            continue
        for o in f.cov[eid]:
            per[o][fr] += 1
    total = Counter()
    for c in per.values():
        total.update(c)
    n_all = sum(total.values())
    if n_all < 500:
        return {"ready": False, "reason": "Too few framed stories yet"}
    frames = [fr for fr, _ in total.most_common(AGENDA_FRAMES)]
    rows = []
    for o, c in per.items():
        n = sum(c.values())
        if n < 40:
            continue
        cells = {}
        for fr in frames:
            k = c[fr]
            lo, hi = wilson(k, n)
            base = total[fr] / n_all
            cells[fr] = {"share": round(k / n, 3), "ratio": round((k / n) / base, 3) if base else None,
                         "lo": round(lo / base, 3) if base else None, "hi": round(hi / base, 3) if base else None}
        rows.append({"id": o, "name": f.outlets[o].name, "group": f.outlets[o].group, "stories": n, "frames": cells})
    rows.sort(key=lambda r: (r["group"], r["name"]))
    return {"ready": True, "frames": [{"key": fr, "label": fr.replace("_", " "), "share": round(total[fr] / n_all, 3)} for fr in frames],
            "outlets": rows}


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
