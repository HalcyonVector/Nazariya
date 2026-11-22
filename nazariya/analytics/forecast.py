"""Breakout forecast: from a story's first two hours, will it reach BIG or more outlets?

Features use only what is known EARLY_H hours after the first report: how many outlets and languages
had covered it by then, and how the first report looked (agency copy, government release, a death
toll, a named politician, a Delhi dateline, the Northeast, the language it broke in, the hour of day).
Labels come from stories that have run their course (no new article for MATURE_H hours).

Validation is chronological: fit on the older 70% of finished stories, score the newer 30%, so the
published hit rate is what the model would have achieved looking forward. The model used for the
live watch list is then refitted on all finished stories.
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from .data import Facts, load_states

EARLY_H = 2
BIG = 10
MATURE_H = 72
WATCH_MAX_AGE_H = 12
MIN_TRAIN = 300

FEATURES = ["Outlets in the first 2 h (log)", "Languages in the first 2 h", "First report is agency copy",
            "Broke as a government release", "Death toll in the first report", "Names a politician",
            "New Delhi dateline", "About the Northeast", "Broke in Hindi", "Broke in a regional language",
            "Hour of day (sin)", "Hour of day (cos)"]


def _features(f: Facts, eid: int, ne: set[str]) -> np.ndarray:
    e = f.events[eid]
    t_early = e["first_seen"] + timedelta(hours=EARLY_H)
    early = [o for o, c in f.cov[eid].items() if c["first_at"] <= t_early]
    langs = {f.outlets[o].lang for o in early}
    fr = e.get("first_report") or {}
    h = (e["first_seen"].hour + e["first_seen"].minute / 60 + 5.5) % 24  # IST
    return np.array([
        math.log1p(len(early)), len(langs), bool(fr.get("wire")), bool(fr.get("govt")), bool(fr.get("toll")),
        (fr.get("politicians") or 0) > 0, bool(fr.get("dateline")), bool(set(fr.get("states") or []) & ne),
        fr.get("lang") == "hi", fr.get("lang") not in (None, "en", "hi"),
        math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24),
    ], dtype=float)


def run(f: Facts) -> dict:
    ne = {s["qid"] for s in load_states() if s["region"] == "northeast"}
    finished = [eid for eid, e in f.events.items()
                if (f.now - e["last_seen"]).total_seconds() >= MATURE_H * 3600 and e["n_news_outlets"] >= 2]
    if len(finished) < MIN_TRAIN:
        return {"ready": False, "reason": f"{len(finished)} stories have run their course so far; the forecast needs {MIN_TRAIN}"}
    finished.sort(key=lambda eid: f.events[eid]["first_seen"])
    X = np.array([_features(f, eid, ne) for eid in finished])
    y = np.array([f.events[eid]["n_news_outlets"] >= BIG for eid in finished])
    if y.sum() < 10:
        return {"ready": False, "reason": f"Only {int(y.sum())} finished stories reached {BIG} outlets so far"}
    cut = int(len(finished) * 0.7)
    model = LogisticRegression(C=1.0, max_iter=2000, class_weight=None)
    model.fit(X[:cut], y[:cut])
    p = model.predict_proba(X[cut:])[:, 1]
    yt = y[cut:]
    score = {"test_stories": int(len(yt)), "test_breakouts": int(yt.sum()),
             "auc": round(float(roc_auc_score(yt, p)), 3) if 0 < yt.sum() < len(yt) else None,
             "brier": round(float(brier_score_loss(yt, p)), 4), "base_rate": round(float(yt.mean()), 4)}
    # hit rate among the stories the model ranked highest in the test period
    k = max(1, int(yt.sum()))
    top = np.argsort(-p)[:k]
    score["hit_rate_top"] = round(float(yt[top].mean()), 3)
    score["top_k"] = k
    # calibration: predicted vs observed in five bins
    bins = np.quantile(p, [0, .2, .4, .6, .8, 1])
    calib = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p <= hi)
        if m.sum():
            calib.append({"predicted": round(float(p[m].mean()), 3), "observed": round(float(yt[m].mean()), 3), "n": int(m.sum())})
    score["calibration"] = calib

    model.fit(X, y)  # all finished stories, for the live list
    coef = [{"feature": n, "or": round(math.exp(c), 3)} for n, c in zip(FEATURES, model.coef_[0])]
    live = [eid for eid, e in f.events.items()
            if EARLY_H * 3600 <= (f.now - e["first_seen"]).total_seconds() <= WATCH_MAX_AGE_H * 3600]
    watch = []
    if live:
        pl = model.predict_proba(np.array([_features(f, eid, ne) for eid in live]))[:, 1]
        for eid, pv in sorted(zip(live, pl), key=lambda t: -t[1])[:12]:
            e = f.events[eid]
            watch.append({"event_id": eid, "title": e["title"], "p": round(float(pv), 3), "n_outlets": e["n_news_outlets"],
                          "first_seen": e["first_seen"].isoformat(), "page": e["n_news_outlets"] >= 3})
    return {"ready": True, "early_hours": EARLY_H, "big": BIG, "trained_on": len(finished),
            "breakouts": int(y.sum()), "score": score, "coef": coef, "watch": watch}
