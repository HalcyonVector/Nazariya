"""Event clustering: embed recent articles, then assign each to an active event or start a new one.

Online, centroid-based: articles are processed in publication order; each joins the most similar
event that was active within WINDOW of its publication time if the cosine similarity to that
event's centroid is ≥ τ, otherwise it founds a new event. Centroids are running sums of member
vectors. `assign_online` is a pure function shared with the evaluation sweep
(nazariya.eval.clustering), so the tuned τ is exactly what production runs.

Which articles: every finished article (extracted, headline_only, skipped, failed) from the last
WINDOW. Articles without body text are embedded from headline + feed summary, so outlets we can
only see headlines from still show up in coverage. Same-outlet duplicates are excluded.

Usage:
    python -m nazariya.cluster              # embed + assign new articles (runs after ingest)
    python -m nazariya.cluster --tau 0.7
    python -m nazariya.cluster --budget-min 20    # cap embedding time (CPU runners)
    python -m nazariya.cluster --embed-only       # home PC: embed on the GPU, leave assignment to cloud
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import and_, bindparam, delete, insert, select, update
from sqlalchemy.engine import Engine

from .db import (
    article_events, article_text, article_vectors, articles, as_utc, entities, events, get_engine, insert_ignore,
    mentions,
)
from .embed import doc_text, embed

MODEL = "bge-m3"
TAU = 0.650  # best pairwise F1 when replaying this algorithm on labelled pairs (reports/clustering_eval.md)
WINDOW = timedelta(hours=72)  # an event stays open this long after its latest article
# An article that names states, none of which its candidate event names, joins only if it is nearly a
# duplicate of the event (stops similar stories in different states, e.g. road accidents, merging).
# On the labelled pairs this split none of 110 same-event pairs. Delhi is ignored: it is named in
# passing by too many national stories.
PLACE_OVERRIDE = 0.80
PLACE_CANDIDATES = 5
DELHI = "Q9357528"
VECTOR_TTL = timedelta(days=3)  # a vector is only needed until its article is assigned and framed
MAX_EMBED_PER_RUN = 3000  # newest first; bounds the text read (transfer) and memory per run
FINISHED = ("extracted", "headline_only", "skipped", "failed")
EMBED_BATCH = 64
EMBED_CHUNK = 256  # rows per write; also how often the time budget is checked


# ---------- algorithm ----------

@dataclass
class EventState:
    id: int | None  # None until written to the DB
    sum: np.ndarray  # float32 sum of member vectors
    count: int
    first_seen: float  # unix seconds
    last_seen: float
    outlets: Counter = field(default_factory=Counter)
    members: list[int] = field(default_factory=list)  # article ids added in this run
    sims: list[float] = field(default_factory=list)
    rep_id: int | None = None  # current representative article (closest to centroid)
    rep_title: str | None = None
    rep_vec: np.ndarray | None = None
    places: Counter = field(default_factory=Counter)  # states named by member articles


def assign_online(vecs: np.ndarray, times: np.ndarray, outlets: list[str], tau: float,
                  state: list[EventState] | None = None, window_s: float = WINDOW.total_seconds(),
                  article_ids: list[int] | None = None,
                  places: list[frozenset] | None = None) -> tuple[list[int], list[float], list[EventState]]:
    """Assign rows of `vecs` (L2-normalised, in any order) to events.

    Returns (event index per row into the returned state list, similarity per row, state).
    Rows founding a new event get similarity 1.0.
    """
    state = list(state or [])
    order = np.argsort(times, kind="stable")
    labels = [0] * len(vecs)
    sims = [0.0] * len(vecs)
    dim = vecs.shape[1]
    cent = np.zeros((len(state) + len(vecs), dim), dtype=np.float32)  # normalised centroids
    last = np.zeros(len(state) + len(vecs))
    for k, ev in enumerate(state):
        cent[k] = ev.sum / max(np.linalg.norm(ev.sum), 1e-9)
        last[k] = ev.last_seen
    n = len(state)
    for i in order:
        v, t = vecs[i], times[i]
        here = places[i] if places is not None else frozenset()
        best, best_sim = -1, -1.0
        if n:
            active = np.abs(t - last[:n]) <= window_s
            if active.any():
                s = cent[:n] @ v
                s[~active] = -np.inf
                # best candidate whose states do not contradict this article's
                for k in np.argsort(-s)[:PLACE_CANDIDATES]:
                    sk = float(s[k])
                    if sk < tau:
                        break
                    ev_places = state[k].places
                    if here and ev_places and not (here & set(ev_places)) and sk < PLACE_OVERRIDE:
                        continue
                    best, best_sim = int(k), sk
                    break
        if best >= 0 and best_sim >= tau:
            ev = state[best]
            ev.sum = ev.sum + v
            ev.count += 1
            ev.first_seen = min(ev.first_seen, t)
            ev.last_seen = max(ev.last_seen, t)
            cent[best] = ev.sum / max(np.linalg.norm(ev.sum), 1e-9)
            last[best] = ev.last_seen
        else:
            best, best_sim = n, 1.0
            ev = EventState(None, v.astype(np.float32).copy(), 1, t, t)
            state.append(ev)
            cent[n], last[n] = v, t
            n += 1
        ev.outlets[outlets[i]] += 1
        ev.places.update(here)
        if article_ids is not None:
            ev.members.append(article_ids[i])
            ev.sims.append(best_sim)
        labels[i], sims[i] = best, best_sim
    return labels, sims, state


# ---------- vectors ----------

def to_blob(v: np.ndarray, dtype=np.float16) -> bytes:
    return v.astype(dtype).tobytes()


def from_blob(b: bytes, dtype=np.float16) -> np.ndarray:
    return np.frombuffer(b, dtype=dtype).astype(np.float32)


# Event centroids are stored as one marker byte + float16 unit vector (2 KB instead of a 4 KB float32
# sum); the sum is rebuilt as unit * n_articles. Legacy rows are plain float32 sums (length % 4 == 0).
CENTROID_V2 = bytes([2])


def centroid_to_blob(total: np.ndarray) -> bytes:
    unit = total / max(float(np.linalg.norm(total)), 1e-9)
    return CENTROID_V2 + to_blob(unit)


def centroid_from_blob(blob: bytes, count: int) -> np.ndarray:
    if len(blob) % 2 == 1 and blob[:1] == CENTROID_V2:
        return from_blob(blob[1:]) * count
    return from_blob(blob, np.float32)


def embed_new(engine: Engine, now: datetime, model: str = MODEL, budget_s: float | None = None) -> int:
    """Embed finished, not-yet-assigned articles from the last WINDOW that have no vector, newest first
    (at most MAX_EMBED_PER_RUN), stopping after `budget_s` seconds; the rest carry over.

    Everything is embedded in memory and written in ONE transaction at the end, so the database sees
    no queries during the long CPU phase and can suspend (free-tier compute hours are billed while it
    is awake). Assigned articles are skipped: their vectors are deleted once they are framed. Safe to
    run from two machines at once: a vector another run already wrote is skipped."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(articles.c.id, articles.c.title, articles.c.summary, article_text.c.text)
            .outerjoin(article_text, article_text.c.article_id == articles.c.id)
            .outerjoin(article_vectors, article_vectors.c.article_id == articles.c.id)
            .outerjoin(article_events, article_events.c.article_id == articles.c.id)
            .where(article_vectors.c.article_id.is_(None), article_events.c.article_id.is_(None),
                   articles.c.status.in_(FINISHED), articles.c.published_at >= now - WINDOW)
            .order_by(articles.c.published_at.desc()).limit(MAX_EMBED_PER_RUN)
        ).all()
    # nothing to embed (no headline, summary or body) would collapse into one blob of identical vectors
    rows = [r for r in rows if (r.title or "").strip() or r.text or r.summary]
    if not rows:
        return 0
    t0 = time.monotonic()
    done: list = []
    out: list[np.ndarray] = []
    for start in range(0, len(rows), EMBED_CHUNK):
        if budget_s is not None and time.monotonic() - t0 > budget_s:
            break
        chunk = rows[start:start + EMBED_CHUNK]
        texts = [doc_text(r.title, r.text) if r.text else doc_text(r.title, r.summary) for r in chunk]
        out.append(embed(model, texts, batch_size=EMBED_BATCH))
        done += chunk
    if not done:
        return 0
    vecs = np.concatenate(out)
    with engine.begin() as conn:
        insert_ignore(conn, article_vectors, [
            {"article_id": r.id, "model": model, "vec": to_blob(v),
             "text_kind": "lede" if r.text else "headline", "created_at": now}
            for r, v in zip(done, vecs)
        ], ["article_id"])
    return len(done)


# ---------- DB pipeline ----------

def load_active_events(conn, since: datetime, model: str) -> list[EventState]:
    rows = conn.execute(
        select(events.c.id, events.c.centroid, events.c.n_articles, events.c.first_seen, events.c.last_seen,
               events.c.rep_article_id, events.c.title)
        .where(events.c.last_seen >= since, events.c.centroid.is_not(None), events.c.model == model)
    ).all()
    state = [EventState(r.id, centroid_from_blob(r.centroid, r.n_articles), r.n_articles,
                        as_utc(r.first_seen).timestamp(), as_utc(r.last_seen).timestamp(),
                        rep_id=r.rep_article_id, rep_title=r.title) for r in rows]
    if state:
        by_id = {ev.id: ev for ev in state}
        ids = list(by_id)
        for i in range(0, len(ids), 1000):
            for eid, outlet in conn.execute(
                select(article_events.c.event_id, articles.c.outlet_id)
                .join(articles, articles.c.id == article_events.c.article_id)
                .where(article_events.c.event_id.in_(ids[i:i + 1000]))
            ):
                by_id[eid].outlets[outlet] += 1
        rep_ids = [ev.rep_id for ev in state if ev.rep_id]
        rep_vecs = {}
        for i in range(0, len(rep_ids), 1000):
            rep_vecs.update(conn.execute(
                select(article_vectors.c.article_id, article_vectors.c.vec)
                .where(article_vectors.c.article_id.in_(rep_ids[i:i + 1000]))
            ).all())
        for ev in state:
            if ev.rep_id in rep_vecs:
                ev.rep_vec = from_blob(rep_vecs[ev.rep_id])
        for i in range(0, len(ids), 1000):
            for eid, qid in conn.execute(
                select(article_events.c.event_id, mentions.c.qid)
                .join(mentions, mentions.c.article_id == article_events.c.article_id)
                .join(entities, entities.c.qid == mentions.c.qid)
                .where(article_events.c.event_id.in_(ids[i:i + 1000]), entities.c.type == "state", mentions.c.qid != DELHI)
            ):
                by_id[eid].places[qid] += 1
    return state


def article_places(conn, article_ids: list[int]) -> dict[int, frozenset]:
    """States (except Delhi) named in each article, from the entity linker's mentions."""
    out: dict[int, set] = {}
    for i in range(0, len(article_ids), 1000):
        for aid, qid in conn.execute(
            select(mentions.c.article_id, mentions.c.qid)
            .join(entities, entities.c.qid == mentions.c.qid)
            .where(mentions.c.article_id.in_(article_ids[i:i + 1000]), entities.c.type == "state", mentions.c.qid != DELHI)
        ):
            out.setdefault(aid, set()).add(qid)
    return {k: frozenset(v) for k, v in out.items()}


def pick_representative(ev: EventState, new_members: dict[int, tuple[str, np.ndarray]]) -> None:
    """Representative headline = member closest to the centroid, among this run's members and the
    previous representative (older members' vectors may already be purged)."""
    c = ev.sum / max(np.linalg.norm(ev.sum), 1e-9)
    cands = [(aid, *new_members[aid]) for aid in ev.members]
    if ev.rep_id is not None and ev.rep_vec is not None:
        cands.append((ev.rep_id, ev.rep_title, ev.rep_vec))
    aid, title, vec = max(cands, key=lambda x: float(x[2] @ c))
    ev.rep_id, ev.rep_title, ev.rep_vec = aid, title, vec


WRITE_CHUNK = 2000  # events per transaction: a dropped connection loses one chunk; a rerun resumes


def assign_new(engine: Engine, now: datetime, tau: float = TAU, model: str = MODEL) -> dict:
    with engine.connect() as conn:
        new = conn.execute(
            select(articles.c.id, articles.c.outlet_id, articles.c.published_at, articles.c.title,
                   article_vectors.c.vec)
            .join(article_vectors, article_vectors.c.article_id == articles.c.id)
            .outerjoin(article_events, article_events.c.article_id == articles.c.id)
            .where(article_events.c.article_id.is_(None), article_vectors.c.model == model)
        ).all()
        if not new:
            return {"assigned": 0}
        oldest = min(as_utc(r.published_at) for r in new)
        state = load_active_events(conn, oldest - WINDOW, model)
        where = article_places(conn, [r.id for r in new])
    n_existing = len(state)
    vecs = np.stack([from_blob(r.vec) for r in new])
    times = np.array([as_utc(r.published_at).timestamp() for r in new])
    labels, sims, state = assign_online(vecs, times, [r.outlet_id for r in new], tau, state,
                                        article_ids=[r.id for r in new],
                                        places=[where.get(r.id, frozenset()) for r in new])
    new_members = {r.id: (r.title, v) for r, v in zip(new, vecs)}
    touched = [ev for ev in state if ev.members]
    for ev in touched:
        pick_representative(ev, new_members)

    def values(ev: EventState) -> dict:
        return {
            "first_seen": datetime.fromtimestamp(ev.first_seen, timezone.utc),
            "last_seen": datetime.fromtimestamp(ev.last_seen, timezone.utc),
            "n_articles": ev.count, "n_outlets": len(ev.outlets), "title": ev.rep_title,
            "rep_article_id": ev.rep_id, "centroid": centroid_to_blob(ev.sum), "model": model,
            "updated_at": now,
        }

    # Batched writes: per-row statements cost a network round trip each against a remote Postgres.
    for start in range(0, len(touched), WRITE_CHUNK):
        chunk = touched[start:start + WRITE_CHUNK]
        with engine.begin() as conn:
            fresh = [ev for ev in chunk if ev.id is None]
            if fresh:
                ids = conn.execute(insert(events).returning(events.c.id, sort_by_parameter_order=True),
                                   [values(ev) for ev in fresh]).scalars().all()
                for ev, eid in zip(fresh, ids):
                    ev.id = eid
            old = [ev for ev in chunk if ev not in fresh]
            if old:
                conn.execute(
                    update(events).where(events.c.id == bindparam("b_id"))
                    .values({k: bindparam(f"v_{k}") for k in values(old[0])}),
                    [{"b_id": ev.id, **{f"v_{k}": v for k, v in values(ev).items()}} for ev in old],
                )
            conn.execute(insert(article_events), [
                {"article_id": aid, "event_id": ev.id, "similarity": round(s, 4), "method": "online",
                 "assigned_at": now}
                for ev in chunk for aid, s in zip(ev.members, ev.sims)
            ])
    return {"assigned": len(new), "new_events": len(state) - n_existing,
            "joined_existing": sum(1 for lab in labels if lab < n_existing),
            "events_touched": len(touched)}


def housekeeping(engine: Engine, now: datetime) -> dict:
    """Close events that left the window (drop the centroid) and expire old vectors (VECTOR_TTL)."""
    with engine.begin() as conn:
        closed = conn.execute(
            update(events).where(and_(events.c.last_seen < now - WINDOW, events.c.centroid.is_not(None)))
            .values(centroid=None)
        ).rowcount
        purged = conn.execute(delete(article_vectors).where(article_vectors.c.created_at < now - VECTOR_TTL)).rowcount
    return {"events_closed": closed, "vectors_purged": purged}


def run(tau: float = TAU, model: str = MODEL, budget_min: float | None = None, embed_only: bool = False) -> dict:
    engine = get_engine()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    t0 = time.monotonic()
    stats = {"embedded": embed_new(engine, now, model, None if budget_min is None else budget_min * 60)}
    stats["embed_s"] = round(time.monotonic() - t0, 1)
    if embed_only:
        return stats
    stats.update(assign_new(engine, now, tau, model))
    stats.update(housekeeping(engine, now))
    stats["total_s"] = round(time.monotonic() - t0, 1)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--budget-min", type=float, help="stop embedding after N minutes (rest carries over)")
    ap.add_argument("--embed-only", action="store_true",
                    help="only embed (home PC helper); event assignment stays with the single cloud writer")
    args = ap.parse_args()
    print(run(args.tau, args.model, args.budget_min, args.embed_only))


if __name__ == "__main__":
    main()
