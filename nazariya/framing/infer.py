"""Pipeline step: predict frames for recent articles from their stored bge-m3 vectors.

The production frame model (models/framing_bge_lr.npz, trained by nazariya.framing.train) is a
logistic regression over the same bge-m3 article vectors the clustering step already computes,
so framing costs one small matrix product per article and covers every article with a vector,
headline-only outlets included. Scores per frame are one-vs-rest probabilities; the primary
frame comes from a separate multi-class model.

Usage:
    python -m nazariya.framing.infer
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import or_, select

from ..cluster import from_blob
from ..db import article_frames, article_vectors, articles, get_engine, upsert
from .train import PROD_MODEL

WINDOW = timedelta(hours=72)
CHUNK = 2000


class FrameModel:
    def __init__(self, path=PROD_MODEL):
        z = np.load(path, allow_pickle=False)
        self.frames = [str(f) for f in z["frames"]]
        self.keep = z["keep"]
        self.ovr_coef, self.ovr_b = z["ovr_coef"], z["ovr_intercept"]
        self.prim_coef, self.prim_b, self.prim_classes = z["prim_coef"], z["prim_intercept"], z["prim_classes"]
        self.version = f"framing-bge-lr@{z['version']}"

    def predict(self, X: np.ndarray) -> tuple[list[str], np.ndarray]:
        scores = np.zeros((len(X), len(self.frames)), dtype=np.float32)
        scores[:, self.keep] = 1 / (1 + np.exp(-(X @ self.ovr_coef.T + self.ovr_b)))
        logits = X @ self.prim_coef.T + self.prim_b
        primary = [self.frames[int(self.prim_classes[i])] for i in logits.argmax(1)]
        return primary, scores


def run(limit: int | None = None) -> dict:
    engine = get_engine()
    model = FrameModel()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    t0 = time.monotonic()
    with engine.connect() as conn:
        rows = conn.execute(
            select(article_vectors.c.article_id, article_vectors.c.vec)
            .join(articles, articles.c.id == article_vectors.c.article_id)
            .outerjoin(article_frames, article_frames.c.article_id == article_vectors.c.article_id)
            .where(article_vectors.c.model == "bge-m3", articles.c.published_at >= now - WINDOW,
                   or_(article_frames.c.article_id.is_(None), article_frames.c.model != model.version))
            .limit(limit)
        ).all()
    for start in range(0, len(rows), CHUNK):
        chunk = rows[start:start + CHUNK]
        primary, scores = model.predict(np.stack([from_blob(r.vec) for r in chunk]))
        with engine.begin() as conn:
            upsert(conn, article_frames, [
                {"article_id": r.article_id, "primary_frame": p,
                 "scores": {f: round(float(x), 3) for f, x in zip(model.frames, s) if x >= 0.05},
                 "model": model.version, "created_at": now}
                for r, p, s in zip(chunk, primary, scores)], ["article_id"])
    return {"framed": len(rows), "model": model.version, "seconds": round(time.monotonic() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    print(run(ap.parse_args().limit))


if __name__ == "__main__":
    main()
