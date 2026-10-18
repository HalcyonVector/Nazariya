"""Sample article pairs for same-event labelling, stratified by similarity and language.

Pairs come from two pools, both restricted to different outlets published within 72h:
  * neighbour pool: each article's top-k most similar articles under a cheap starter model
    (e5-small). Split into similarity quantile bands so labels cover easy and hard cases
    instead of only obvious matches.
  * random pool: uniformly random pairs, mostly unrelated, to anchor the low end.
About 30% of each band is English–Hindi where available, and no article appears in more than
two pairs. A small set of pairs is repeated (sides swapped) to measure self-consistency.

The pair file holds ids, URLs and metadata only, never article text, so it can be committed.
Texts of sampled articles are pinned in the article_text cache for later evaluation.

Usage:
    python -m nazariya.label.sample            # writes labels/event_pairs_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import timedelta

import numpy as np
from sqlalchemy import select, update

from ..config import ROOT
from ..db import PINNED_UNTIL, article_text, articles, as_utc, get_engine
from ..embed import doc_text, embed

LABELS_DIR = ROOT / "labels"
WINDOW = timedelta(hours=72)
TOP_K = 10
N_BANDS = 5
PER_BAND = 70
N_RANDOM = 50
N_REPEATS = 20
CROSS_LINGUAL_SHARE = 0.3
MAX_USES = 2
STARTER_MODEL = "e5-small"


def load_articles(engine) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(
            select(articles.c.id, articles.c.outlet_id, articles.c.lang, articles.c.title,
                   articles.c.url, articles.c.published_at, article_text.c.text)
            .join(article_text, article_text.c.article_id == articles.c.id)
            .where(articles.c.status == "extracted", articles.c.syndicated_of.is_(None),
                   articles.c.lang.in_(["en", "hi"]))
        ).all()
    return [dict(r._mapping, published_at=as_utc(r.published_at)) for r in rows]


def candidate_pools(arts: list[dict], vecs: np.ndarray, rng: random.Random) -> tuple[dict, list]:
    """Return ({(i, j): sim} neighbour pairs, [(i, j, sim)] random valid pairs)."""
    n = len(arts)
    times = np.array([a["published_at"].timestamp() for a in arts])
    outlets = np.array([a["outlet_id"] for a in arts])
    window_s = WINDOW.total_seconds()
    neighbours: dict[tuple[int, int], float] = {}
    for start in range(0, n, 512):
        block = vecs[start:start + 512] @ vecs.T
        for r in range(block.shape[0]):
            i = start + r
            row = block[r]
            invalid = (outlets == outlets[i]) | (np.abs(times - times[i]) > window_s)
            row = np.where(invalid, -np.inf, row)
            for j in np.argpartition(-row, TOP_K)[:TOP_K]:
                if np.isfinite(row[j]):
                    key = (min(i, int(j)), max(i, int(j)))
                    neighbours[key] = float(row[j])
    randoms = []
    while len(randoms) < N_RANDOM * 20:
        i, j = rng.randrange(n), rng.randrange(n)
        if i != j and outlets[i] != outlets[j] and abs(times[i] - times[j]) <= window_s:
            randoms.append((min(i, j), max(i, j), float(vecs[i] @ vecs[j])))
    return neighbours, randoms


def is_cross(arts: list[dict], i: int, j: int) -> bool:
    return arts[i]["lang"] != arts[j]["lang"]


def take(pool: list[tuple[int, int, float]], k: int, arts: list[dict], uses: Counter,
         rng: random.Random) -> list[tuple[int, int, float]]:
    """Pick k pairs, ~30% cross-lingual when possible, respecting the per-article cap."""
    rng.shuffle(pool)
    cross = [p for p in pool if is_cross(arts, p[0], p[1])]
    mono = [p for p in pool if not is_cross(arts, p[0], p[1])]
    chosen: list[tuple[int, int, float]] = []

    def fill(group: list, until: int) -> None:
        for p in group:
            if len(chosen) >= until:
                return
            if p in chosen or uses[p[0]] >= MAX_USES or uses[p[1]] >= MAX_USES:
                continue
            chosen.append(p)
            uses[p[0]] += 1
            uses[p[1]] += 1

    fill(cross, round(k * CROSS_LINGUAL_SHARE))  # cross-lingual quota first
    fill(mono, k)  # then same-language
    fill(cross, k)  # top up with cross-lingual if same-language ran out
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    out = LABELS_DIR / f"event_pairs_{args.version}.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists; pick a new --version rather than overwrite a labelled set")
    rng = random.Random(args.seed)

    engine = get_engine()
    arts = load_articles(engine)
    print(f"{len(arts)} articles ({Counter(a['lang'] for a in arts)}); embedding with {STARTER_MODEL}…")
    vecs = embed(STARTER_MODEL, [doc_text(a["title"], a["text"]) for a in arts], progress=True)

    neighbours, randoms = candidate_pools(arts, vecs, rng)
    pool = sorted(((i, j, s) for (i, j), s in neighbours.items()), key=lambda p: p[2])
    edges = np.quantile([p[2] for p in pool], np.linspace(0, 1, N_BANDS + 1))
    print(f"{len(pool)} neighbour pairs; band edges {np.round(edges, 3).tolist()}")

    uses: Counter = Counter()
    sampled: list[tuple[int, int, float, str]] = []
    for b in range(N_BANDS):
        lo, hi = edges[b], edges[b + 1]
        band = [p for p in pool if lo <= p[2] <= hi]
        sampled += [(*p, f"nn{b + 1}") for p in take(band, PER_BAND, arts, uses, rng)]
    sampled += [(*p, "random") for p in take(randoms, N_RANDOM, arts, uses, rng)]
    rng.shuffle(sampled)

    records = []
    for n, (i, j, sim, band) in enumerate(sampled, 1):
        a, b = (arts[i], arts[j]) if rng.random() < 0.5 else (arts[j], arts[i])
        records.append({
            "pair_id": f"{args.version}-{n:04d}", "a_id": a["id"], "b_id": b["id"],
            "a_url": a["url"], "b_url": b["url"], "a_outlet": a["outlet_id"], "b_outlet": b["outlet_id"],
            "langs": f"{a['lang']}-{b['lang']}", "band": band, f"sim_{STARTER_MODEL}": round(sim, 4),
            "repeat_of": None,
        })
    # repeats go at the end with sides swapped, so they're seen well after the original
    for n, orig in enumerate(rng.sample(records, N_REPEATS), len(records) + 1):
        records.append({**orig, "pair_id": f"{args.version}-{n:04d}", "repeat_of": orig["pair_id"],
                        "a_id": orig["b_id"], "b_id": orig["a_id"], "a_url": orig["b_url"], "b_url": orig["a_url"],
                        "a_outlet": orig["b_outlet"], "b_outlet": orig["a_outlet"],
                        "langs": "-".join(reversed(orig["langs"].split("-")))})

    LABELS_DIR.mkdir(exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    ids = {r["a_id"] for r in records} | {r["b_id"] for r in records}
    with engine.begin() as conn:
        conn.execute(update(article_text).where(article_text.c.article_id.in_(ids)).values(expires_at=PINNED_UNTIL))

    print(f"wrote {len(records)} pairs ({len(records) - N_REPEATS} + {N_REPEATS} repeats) → {out.relative_to(ROOT)}")
    print("bands:", dict(Counter(r["band"] for r in records)))
    print("langs:", dict(Counter(r["langs"] for r in records)))
    print(f"pinned text for {len(ids)} articles")


if __name__ == "__main__":
    main()
