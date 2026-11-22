"""Attention model: how much coverage each state gets against what its size and distance from
Delhi predict.

Outcome: outlet-event pairs whose articles name the state (dateline-only Delhi mentions are
dropped in nazariya/facts.py). Poisson regression on log population and log distance from Delhi.
The Northeast is deliberately not a model term: whether distant, small states get less than their
size and distance explain is the question, so it must show up in the residuals.

Story half-life (survival analysis) lives here too: both are about which news is kept alive."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import PoissonRegressor

from .data import Facts, load_states

OPEN_HOURS = 72  # an event with no new article for this long is closed (cluster.py window)


def _km(durations: list[float], closed: list[bool], grid: list[float]) -> tuple[list[float], float | None]:
    """Kaplan-Meier survival evaluated on `grid` (days), and the median (half-life)."""
    order = sorted(zip(durations, closed))
    at_risk, s = len(order), 1.0
    curve, med, i = [], None, 0
    steps = []
    while i < len(order):
        t = order[i][0]
        d = n = 0
        while i < len(order) and order[i][0] == t:
            d += order[i][1]
            n += 1
            i += 1
        if d:
            s *= 1 - d / at_risk
            steps.append((t, s))
            if med is None and s <= 0.5:
                med = t
        at_risk -= n
    j, cur = 0, 1.0
    for g in grid:
        while j < len(steps) and steps[j][0] <= g:
            cur = steps[j][1]
            j += 1
        curve.append(round(cur, 4))
    return curve, med


def _logrank(a: tuple[list, list], b: tuple[list, list]) -> float:
    """Two-group log-rank chi-square statistic (1 df)."""
    ev = sorted({t for t, c in zip(*a) if c} | {t for t, c in zip(*b) if c})
    ta, ca = np.array(a[0]), np.array(a[1])
    tb, cb = np.array(b[0]), np.array(b[1])
    o_e = v = 0.0
    for t in ev:
        na, nb = (ta >= t).sum(), (tb >= t).sum()
        da, db = ((ta == t) & ca).sum(), ((tb == t) & cb).sum()
        n, d = na + nb, da + db
        if n < 2:
            continue
        o_e += da - d * na / n
        v += d * (na / n) * (nb / n) * (n - d) / (n - 1)
    return float(o_e ** 2 / v) if v > 0 else 0.0


def half_life(f: Facts) -> dict:
    states = load_states()
    ne = {s["qid"] for s in states if s["region"] == "northeast"}
    evs = [e for e in f.events.values() if e["n_news_outlets"] >= 2]
    if len(evs) < 100:
        return {"ready": False, "reason": f"{len(evs)} multi-outlet events so far"}
    groups: dict[str, dict[str, tuple[list, list]]] = defaultdict(lambda: defaultdict(lambda: ([], [])))
    for e in evs:
        dur = max(0.0, (e["last_seen"] - e["first_seen"]).total_seconds() / 86400)
        closed = (f.now - e["last_seen"]).total_seconds() / 3600 >= OPEN_HOURS
        st = set(e.get("states") or [])
        size = "2-4 outlets" if e["n_news_outlets"] < 5 else "5-9 outlets" if e["n_news_outlets"] < 10 else "10+ outlets"
        keys = {
            "Where": "Northeast" if st & ne else "Other states" if st else "No state named",
            "Broke in": {"en": "English", "hi": "Hindi"}.get(e.get("first_lang") or "", "A regional language"),
            "Size": size,
            "Frame": (e.get("frame") or "unknown").replace("_", " "),
        }
        for dim, val in keys.items():
            groups[dim][val][0].append(dur)
            groups[dim][val][1].append(closed)
    n_closed = sum(c for dims in groups["Size"].values() for c in dims[1])
    if n_closed < 50:
        return {"ready": False, "reason": f"{n_closed} stories have ended so far (no new article for "
                f"{OPEN_HOURS} hours); half-lives need at least 50"}
    grid = [round(x * 0.25, 2) for x in range(0, 41)]  # 0 to 10 days
    out = {}
    for dim, vals in groups.items():
        series = []
        for val, (d, c) in sorted(vals.items(), key=lambda kv: -len(kv[1][0])):
            if len(d) < 30:
                continue
            rest = ([x for v2, (d2, _) in vals.items() if v2 != val for x in d2],
                    [x for v2, (_, c2) in vals.items() if v2 != val for x in c2])
            curve, med = _km(d, c, grid)
            series.append({"name": val, "n": len(d), "closed": int(sum(c)), "half_life_days": med,
                           "curve": curve, "logrank_chi2": round(_logrank((d, c), rest), 2)})
        if dim == "Frame":
            series = series[:6]
        out[dim] = series
    return {"ready": True, "events": len(evs), "grid_days": grid, "dimensions": out}


def attention(f: Facts) -> dict:
    states = load_states()
    delhi = next(s for s in states if s["code"] == "DL")
    pairs, events, by_group = Counter(), Counter(), defaultdict(Counter)
    for eid, e in f.events.items():
        for q in e.get("states") or []:
            events[q] += 1
            for o in f.cov[eid]:
                pairs[q] += 1
                by_group[f.outlets[o].group][q] += 1
    total = sum(pairs.values())
    if total < 300:
        return {"ready": False, "reason": f"{total} state-linked outlet-event pairs so far"}

    def km(s):
        p1, p2 = math.radians(s["lat"]), math.radians(delhi["lat"])
        dl = math.radians(s["lon"] - delhi["lon"])
        return 6371 * math.acos(min(1, math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)))

    X = np.array([[math.log(s["population"]), math.log(50 + km(s))] for s in states])
    y = np.array([pairs[s["qid"]] for s in states], dtype=float)
    m = PoissonRegressor(alpha=0, max_iter=1000).fit(X, y)
    exp = m.predict(X)
    pop_total = sum(s["population"] for s in states)
    rows = []
    for s, e, yy, x in zip(states, exp, y, X):
        rows.append({
            "qid": s["qid"], "code": s["code"], "name": s["name"], "tile": s["tile"], "region": s["region"],
            "population": s["population"], "km_from_delhi": round(km(s)), "events": events[s["qid"]],
            "pairs": int(yy), "expected": round(float(e), 1), "ratio": round(float(yy / e), 3) if e > 0 else None,
            "raw_ratio": round((yy / total) / (s["population"] / pop_total), 3),
            "by_group": {g: c[s["qid"]] for g, c in by_group.items()},
        })
    return {"ready": True, "pairs": int(total), "d2": round(float(m.score(X, y)), 3),
            "coef": {"log_population": round(float(m.coef_[0]), 3), "log_km_from_delhi": round(float(m.coef_[1]), 3),
                     "intercept": round(float(m.intercept_), 3)},
            "states": rows}
