"""What makes an event news: a model of whether an outlet covers an event, given how the event
looked when it first broke. The same model then ranks the week's most under-covered events (far
fewer outlets than predicted)."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .data import Facts, load_states

GROUPS = ["English national", "English regional", "Hindi"]
SINGLETON_SAMPLE = 0.15
BOOTSTRAP = 25
MIN_FRAME_EVENTS = 150
RNG = np.random.default_rng(11)
LAST: dict = {}  # models of the most recent run, for predict()

BASE_FEATURES = [
    ("toll_10", "Death toll of 10 or more"),
    ("toll_1", "Death toll under 10"),
    ("politician", "Names an Indian politician"),
    ("wire", "First report is agency copy"),
    ("govt", "Broke as a government press release"),
    ("dateline", "First report datelined New Delhi"),
    ("delhi", "About Delhi"),
    ("northeast", "About a Northeast state"),
    ("other_state", "About another named state"),
    ("broke_hindi", "Broke in Hindi"),
]


def _features(f: Facts, frames: list[str]) -> tuple[list[str], list[str], dict[int, np.ndarray]]:
    states = load_states()
    ne = {s["qid"] for s in states if s["region"] == "northeast"}
    delhi = next(s["qid"] for s in states if s["code"] == "DL")
    keys = [k for k, _ in BASE_FEATURES] + [f"frame_{fr}" for fr in frames] + ["no_text"]
    labels = [l for _, l in BASE_FEATURES] + [f"Framed as {fr.replace('_', ' ')}" for fr in frames] + ["First report had no text"]
    X = {}
    for eid, e in f.events.items():
        fr = e.get("first_report") or {}
        st = set(fr.get("states") or [])
        toll = fr.get("toll") or 0
        v = [toll >= 10, 1 <= toll < 10, (fr.get("politicians") or 0) > 0, bool(fr.get("wire")),
             bool(fr.get("govt")), bool(fr.get("dateline")), delhi in st, bool(st & ne),
             bool(st - ne - {delhi}), fr.get("lang") == "hi"]
        v += [fr.get("frame") == x for x in frames]
        v += [not fr.get("text")]
        X[eid] = np.array(v, dtype=float)
    return keys, labels, X


def _rows(f: Facts, X: dict, outlets: list[str], events: list[int], weights: dict[int, float]):
    oi = {o: i for i, o in enumerate(outlets)}
    rows, y, w, g = [], [], [], []
    for eid in events:
        e = f.events[eid]
        first = (e.get("first_report") or {}).get("outlet")
        for o in outlets:
            if o == first:
                continue  # the outlet that broke it covered it by definition
            d = np.zeros(len(outlets) - 1)
            if oi[o] > 0:
                d[oi[o] - 1] = 1  # outlet fixed effect: absorbs each outlet's sheer volume
            rows.append(np.concatenate([X[eid], d]))
            y.append(o in f.cov[eid])
            w.append(weights.get(eid, 1.0))
            g.append(eid)
    return np.array(rows), np.array(y), np.array(w), np.array(g)


def _fit(Xr, y, w):
    m = LogisticRegression(C=10.0, max_iter=3000)
    m.fit(Xr, y, sample_weight=w)
    return m


def run(f: Facts) -> dict:
    frame_n = Counter((e.get("first_report") or {}).get("frame") for e in f.events.values())
    frames = sorted(fr for fr, n in frame_n.items() if fr and fr != "other" and n >= MIN_FRAME_EVENTS)
    keys, labels, X = _features(f, frames)
    multi = [eid for eid, e in f.events.items() if e["n_news_outlets"] >= 2]
    singles = [eid for eid, e in f.events.items() if e["n_news_outlets"] == 1]
    picked = [eid for eid in singles if RNG.random() < SINGLETON_SAMPLE]
    train = multi + picked
    weights = {eid: 1 / SINGLETON_SAMPLE for eid in picked}
    if len(multi) < 300:
        return {"ready": False, "reason": f"{len(multi)} multi-outlet events so far; the model needs at least 300"}

    groups, models = {}, {}
    for gname in GROUPS:
        outs = sorted(o for o, x in f.outlets.items() if x.group == gname and any(o in c for c in f.cov.values()))
        if len(outs) < 2:
            continue
        Xr, y, w, g = _rows(f, X, outs, train, weights)
        m = _fit(Xr, y, w)
        models[gname] = (m, outs)
        nf = len(keys)
        # uncertainty: refit on bootstrap samples of events
        ev = np.array(train)
        boot = []
        for _ in range(BOOTSTRAP):
            pick = set(RNG.choice(ev, len(ev)).tolist())
            mask = np.isin(g, list(pick))
            if y[mask].all() or not y[mask].any():
                continue
            boot.append(_fit(Xr[mask], y[mask], w[mask]).coef_[0][:nf])
        boot = np.array(boot)
        # held-out discrimination: folds split by event, so no event is in train and test at once
        aucs = []
        for tr, te in GroupKFold(5).split(Xr, y, g):
            p = _fit(Xr[tr], y[tr], w[tr]).predict_proba(Xr[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p, sample_weight=w[te]))
        support = Xr[:, :nf].T @ w / w.sum()
        feats = []
        for i, (k, lab) in enumerate(zip(keys, labels)):
            if k == "no_text":
                continue
            lo, hi = (np.percentile(boot[:, i], [2.5, 97.5]) if len(boot) else (np.nan, np.nan))
            feats.append({"key": k, "label": lab, "or": round(math.exp(m.coef_[0][i]), 3),
                          "lo": round(math.exp(lo), 3), "hi": round(math.exp(hi), 3),
                          "share": round(float(support[i]), 4)})
        groups[gname] = {"outlets": outs, "pairs": int(len(y)), "events": len(train),
                         "base_rate": round(float((y * w).sum() / w.sum()), 4),
                         "auc": round(float(np.mean(aucs)), 3), "features": feats}

    LAST.update(models=models, X=X)
    return {"ready": bool(groups), "groups": groups,
            "under_covered": _surprises(f, X, models, under=True),
            "over_covered": _surprises(f, X, models, under=False)}


def predict(f: Facts, events: list[int]) -> dict[int, dict[str, float]]:
    """Each outlet's modelled probability of covering each event (from the last run's fit); the
    outlet that broke the event is left out, as in the model."""
    if not LAST.get("models"):
        return {}
    out: dict[int, dict[str, float]] = {e: {} for e in events}
    for m, outs in LAST["models"].values():
        Xr, _, _, g = _rows(f, LAST["X"], outs, events, {})
        if not len(Xr):
            continue
        p = m.predict_proba(Xr)[:, 1]
        # _rows walks events in order and outlets in order, skipping each event's breaking outlet
        k = 0
        for eid in events:
            first = (f.events[eid].get("first_report") or {}).get("outlet")
            for o in outs:
                if o == first:
                    continue
                out[eid][o] = float(p[k])
                k += 1
    return out


def _surprises(f: Facts, X: dict, models: dict, under: bool, days: float = 7, top: int = 15) -> list[dict]:
    """Events from the last week whose outlet count is furthest from the model's expectation,
    as a z-score under independent per-outlet coverage probabilities."""
    recent = [eid for eid, e in f.events.items() if (f.now - e["first_seen"]).total_seconds() < days * 86400]
    exp, var, act = Counter(), Counter(), Counter()
    for m, outs in models.values():
        Xr, y, _, g = _rows(f, X, outs, recent, {})
        if not len(Xr):
            continue
        p = m.predict_proba(Xr)[:, 1]
        for eid, pi, yi in zip(g.tolist(), p.tolist(), y.tolist()):
            exp[eid] += pi
            var[eid] += pi * (1 - pi)
            act[eid] += yi
    out = []
    for eid in recent:
        if var[eid] <= 0:
            continue
        e = f.events[eid]
        first = (e.get("first_report") or {}).get("outlet")
        z = (act[eid] - exp[eid]) / math.sqrt(var[eid])
        if (under and exp[eid] >= 4 and z < -2) or (not under and z > 2):
            out.append({"event_id": eid, "title": e["title"], "first_seen": e["first_seen"].isoformat(),
                        "broke_in": f.outlets[first].name if first in f.outlets else {"ani": "ANI (news agency)", "pib": "PIB (government press releases)"}.get(first, first),
                        "expected": round(exp[eid], 1), "actual": int(act[eid]), "z": round(z, 2),
                        "outlets": sorted(f.cov[eid])})
    out.sort(key=lambda r: r["z"] if under else -r["z"])
    return out[:top]


def translation_drift(f: Facts, top: int = 15) -> dict:
    """Events covered in both languages, ranked by how far apart the English and Hindi coverage sit
    in the multilingual embedding space (1 - cosine of the language means)."""
    rows = [e for e in f.events.values()
            if e.get("lang_drift") is not None and (e.get("langs") or {}).get("en", 0) >= 2
            and (e.get("langs") or {}).get("hi", 0) >= 2]
    if len(rows) < 20:
        return {"ready": False, "reason": f"{len(rows)} events covered in both languages so far"}
    d = np.array([e["lang_drift"] for e in rows])
    rows.sort(key=lambda e: -e["lang_drift"])
    return {"ready": True, "events": len(rows), "median": round(float(np.median(d)), 4),
            "p90": round(float(np.percentile(d, 90)), 4),
            "top": [{"event_id": e["event_id"], "title": e["title"], "drift": round(e["lang_drift"], 4),
                     "en": e["langs"].get("en", 0), "hi": e["langs"].get("hi", 0),
                     "percentile": round(float((d < e["lang_drift"]).mean()) * 100)} for e in rows[:top]]}
