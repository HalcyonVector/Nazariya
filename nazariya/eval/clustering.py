"""Tune and evaluate event clustering by replaying the production algorithm over a real window.

All finished articles published in the window spanned by the labelled event pairs (±1 day) are
embedded (cached in data/eval/) and clustered with nazariya.cluster.assign_online at each τ.
Scored on the labelled pairs (pair predicted "same event" iff both land in one cluster):
pairwise precision / recall / F1, recall on English–Hindi positives, plus cluster-shape checks
(singleton share, multi-outlet events, largest event) that catch chaining into mega-clusters.

Full B-cubed / AMI needs a complete gold clustering of a window; that is part of the final human
labelling session. Until then these are PROVISIONAL (silver labels).

Usage:
    python -m nazariya.eval.clustering                # sweep τ → reports/clustering_eval.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import select

from ..cluster import FINISHED, MODEL, TAU, assign_online
from ..config import ROOT
from ..db import article_text, articles, as_utc, get_engine
from ..embed import doc_text, embed
from .labels import event_pair_labels

GRID = [round(x, 3) for x in np.arange(0.55, 0.851, 0.025)]


def window_articles(items: list[dict]) -> list:
    ids = {it["pair"]["a_id"] for it in items} | {it["pair"]["b_id"] for it in items}
    engine = get_engine()
    with engine.connect() as conn:
        span = conn.execute(select(articles.c.published_at).where(articles.c.id.in_(ids))).scalars().all()
        lo, hi = min(map(as_utc, span)) - timedelta(days=1), max(map(as_utc, span)) + timedelta(days=1)
        return conn.execute(
            select(articles.c.id, articles.c.outlet_id, articles.c.lang, articles.c.title, articles.c.summary,
                   articles.c.published_at, article_text.c.text)
            .outerjoin(article_text, article_text.c.article_id == articles.c.id)
            .where(articles.c.published_at.between(lo, hi),
                   articles.c.status.in_(FINISHED) | articles.c.id.in_(ids))
        ).all()


def vectors(rows, model: str) -> np.ndarray:
    """Vectors for `rows`, reusing the on-disk cache and embedding only articles not in it.
    (Cached vectors go stale if an article's text changes; delete the cache to force a rebuild.)"""
    cache = ROOT / "data" / "eval" / f"window_vectors_{model}.npz"
    known: dict[int, np.ndarray] = {}
    if cache.exists():
        z = np.load(cache)
        known = dict(zip(z["ids"].tolist(), z["vecs"]))
    missing = [r for r in rows if r.id not in known]
    if missing:
        texts = [doc_text(r.title, r.text) if r.text else doc_text(r.title, r.summary) for r in missing]
        for r, v in zip(missing, embed(model, texts, batch_size=64, progress=True)):
            known[r.id] = v
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ids=np.array(list(known)), vecs=np.stack(list(known.values())))
    return np.stack([known[r.id] for r in rows])


def score(labels: list[int], index: dict[int, int], items: list[dict], outlets: list[str]) -> dict:
    tp = fp = fn = 0
    cross_pos = cross_hit = 0
    for it in items:
        same = labels[index[it["pair"]["a_id"]]] == labels[index[it["pair"]["b_id"]]]
        if it["y"] and same:
            tp += 1
        elif it["y"]:
            fn += 1
        elif same:
            fp += 1
        if it["y"] and it["pair"]["langs"] in ("en-hi", "hi-en"):
            cross_pos += 1
            cross_hit += same
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    sizes = Counter(labels)
    ev_outlets: dict[int, set] = {}
    for lab, o in zip(labels, outlets):
        ev_outlets.setdefault(lab, set()).add(o)
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "cross_recall": cross_hit / cross_pos if cross_pos else float("nan"),
            "n_events": len(sizes), "singleton_share": sum(1 for s in sizes.values() if s == 1) / len(sizes),
            "multi_outlet_events": sum(1 for s in ev_outlets.values() if len(s) >= 3),
            "largest_event": max(sizes.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    items = event_pair_labels()
    rows = window_articles(items)
    index = {r.id: n for n, r in enumerate(rows)}
    items = [it for it in items if it["pair"]["a_id"] in index and it["pair"]["b_id"] in index]
    print(f"{len(rows)} articles in window, {len(items)} labelled pairs")
    vecs = vectors(rows, args.model)
    times = np.array([as_utc(r.published_at).timestamp() for r in rows])
    outlets = [r.outlet_id for r in rows]

    results = []
    for tau in GRID:
        labels, _, _ = assign_online(vecs, times, outlets, tau)
        res = {"tau": tau, **score(labels, index, items, outlets)}
        results.append(res)
        print(f"τ={tau:.3f}  P={res['precision']:.3f} R={res['recall']:.3f} F1={res['f1']:.3f}  "
              f"en-hi R={res['cross_recall']:.3f}  events={res['n_events']}  "
              f"singletons={res['singleton_share']:.0%}  ≥3-outlet={res['multi_outlet_events']}  "
              f"largest={res['largest_event']}", flush=True)

    best = max(results, key=lambda r: r["f1"])
    n_gold = sum(it["source"] == "gold" for it in items)
    lines = [
        "# Event clustering evaluation",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `python -m nazariya.eval.clustering`.",
        "",
        f"> **PROVISIONAL.** {len(items) - n_gold} of {len(items)} pair labels are LLM silver labels; "
        f"{n_gold} are human. Pairwise metrics only; B-cubed/AMI need the gold clustering from the final "
        "labelling session.",
        "",
        f"Replays the production online algorithm (`nazariya.cluster.assign_online`, model `{args.model}`, "
        f"72 h window) over all {len(rows)} finished articles published in the labelled window. A pair is "
        "predicted *same event* iff both articles land in the same event.",
        "",
        "| τ | precision | recall | F1 | en–hi recall | events | singletons | events with ≥3 outlets | largest event |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        mark = " **←**" if r is best else (" (current)" if abs(r["tau"] - TAU) < 1e-6 else "")
        lines.append(f"| {r['tau']:.3f}{mark} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                     f"{r['cross_recall']:.3f} | {r['n_events']} | {r['singleton_share']:.0%} | "
                     f"{r['multi_outlet_events']} | {r['largest_event']} |")
    lines += [
        "",
        f"**Best F1 at τ = {best['tau']:.3f}** (P {best['precision']:.3f}, R {best['recall']:.3f}). "
        f"Production currently uses τ = {TAU}.",
        "",
        "Notes: article-to-centroid similarity runs higher than article-to-article similarity for true "
        "members, so the best clustering τ need not equal the pairwise threshold from the embedding "
        "comparison. Watch the largest-event column: a sudden jump means events are chaining together.",
    ]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "clustering_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "data" / "eval" / "clustering_sweep.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("→ reports/clustering_eval.md")


if __name__ == "__main__":
    main()
