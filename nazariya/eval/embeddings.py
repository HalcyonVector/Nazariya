"""Embedding model comparison: how well does cosine similarity separate same-event pairs?

For each model in nazariya.embed.MODELS:
  * ROC-AUC with a bootstrap 95% CI, overall and for cross-lingual (English–Hindi) pairs only
  * the similarity threshold that maximises F1 for "same event", with precision/recall there
    (this becomes the starting τ for event clustering)
  * throughput on GPU (development) and CPU (GitHub Actions / home task)

Labels: gold (human) where available, otherwise silver (LLM); reported separately.

Usage:
    python -m nazariya.eval.embeddings            # all models → reports/embedding_bakeoff.md
    python -m nazariya.eval.embeddings --models e5-small,e5-base
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score

from ..config import ROOT
from ..embed import MODELS, doc_text, embed, unload
from ..label.docs import load_docs
from .labels import event_pair_labels

BOOTSTRAP = 1000
CPU_SAMPLE = 64


def auc_ci(y: np.ndarray, s: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    if len(set(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    point = roc_auc_score(y, s)
    stats = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) == 2:
            stats.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, lo, hi


def best_f1(y: np.ndarray, s: np.ndarray) -> dict:
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    i = int(np.nanargmax(f1[:-1]))  # last point has no threshold
    return {"threshold": float(t[i]), "precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])}


def evaluate(key: str, items: list[dict], texts: dict[int, str], rng: np.random.Generator) -> dict:
    ids = sorted(texts)
    t0 = time.perf_counter()
    vecs = embed(key, [texts[i] for i in ids], batch_size=32)
    gpu_s = time.perf_counter() - t0
    row = {i: n for n, i in enumerate(ids)}
    sims = np.array([float(vecs[row[it["pair"]["a_id"]]] @ vecs[row[it["pair"]["b_id"]]]) for it in items])
    y = np.array([it["y"] for it in items])
    cross = np.array([it["pair"]["langs"] in ("en-hi", "hi-en") for it in items])
    gold = np.array([it["source"] == "gold" for it in items])

    sample = [texts[i] for i in ids[:CPU_SAMPLE]]
    embed(key, sample[:4], device="cpu")  # warm-up / load
    t0 = time.perf_counter()
    embed(key, sample, device="cpu", batch_size=16)
    cpu_per_s = len(sample) / (time.perf_counter() - t0)
    unload()

    res = {"model": key, "hf_id": MODELS[key][0], "n": len(items),
           "auc": auc_ci(y, sims, rng), "auc_cross": auc_ci(y[cross], sims[cross], rng),
           "auc_gold": auc_ci(y[gold], sims[gold], rng) if gold.sum() else None,
           "best": best_f1(y, sims), "gpu_s": gpu_s, "cpu_docs_per_s": cpu_per_s, "n_docs": len(ids),
           "sims": sims.round(4).tolist()}
    print(f"{key:10s} AUC {res['auc'][0]:.3f} [{res['auc'][1]:.3f}-{res['auc'][2]:.3f}]  "
          f"cross {res['auc_cross'][0]:.3f}  τ={res['best']['threshold']:.3f} F1={res['best']['f1']:.3f}  "
          f"cpu {cpu_per_s:.1f} docs/s", flush=True)
    return res


def fmt_auc(a) -> str:
    return "–" if a is None or np.isnan(a[0]) else f"{a[0]:.3f} ({a[1]:.2f}–{a[2]:.2f})"


def write_report(results: list[dict], items: list[dict]) -> None:
    n_gold = sum(it["source"] == "gold" for it in items)
    n_pos = sum(it["y"] for it in items)
    n_cross = sum(it["pair"]["langs"] in ("en-hi", "hi-en") for it in items)
    ranked = sorted(results, key=lambda r: -r["auc"][0])
    best = ranked[0]
    if len(ranked) > 1:
        second = ranked[1]
        gap = best["auc"][1] - second["auc"][2]  # best's lower bound minus runner-up's upper bound
        verdict = (f"Its CI lies above `{second['model']}`'s, so the lead is clear at this sample size."
                   if gap > 0 else
                   f"Its CI overlaps `{second['model']}`'s (by {-gap:.3f}), so the lead is suggestive, not established.")
    else:
        verdict = ""
    # Actions sizing: ~3k new articles/day, CPU only
    lines = [
        "# Embedding model comparison (event pairs)",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `python -m nazariya.eval.embeddings`.",
        "",
        "> **PROVISIONAL.** " + (f"{len(items) - n_gold} of {len(items)} labels are LLM silver labels "
                                 f"(gpt-oss-120b); {n_gold} are human. Re-run after the human gold labelling "
                                 "session; the `gold` column is the one to trust then."),
        "",
        f"Task: does cosine similarity of (headline + first {200} words) embeddings separate *same event* "
        f"pairs from the rest? {len(items)} pairs, {n_pos} same-event, {n_cross} English–Hindi.",
        "",
        "| model | ROC-AUC (95% CI) | AUC en–hi pairs | AUC gold only | best-F1 τ | P / R at τ | GPU time (all docs) | CPU docs/s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: -r["auc"][0]):
        b = r["best"]
        lines.append(
            f"| `{r['model']}` | {fmt_auc(r['auc'])} | {fmt_auc(r['auc_cross'])} | {fmt_auc(r['auc_gold'])} | "
            f"{b['threshold']:.3f} | {b['precision']:.2f} / {b['recall']:.2f} | {r['gpu_s']:.1f}s for {r['n_docs']} | "
            f"{r['cpu_docs_per_s']:.1f} |"
        )
    cpu_min = 3000 / best["cpu_docs_per_s"] / 60
    lines += [
        "",
        "## Reading this",
        f"- **Highest AUC: `{best['model']}`.** {verdict}",
        "- **τ** is the similarity above which a pair is called the same event at best F1; it seeds the "
        "clustering threshold. Thresholds are model-specific (E5 similarities are compressed near 1).",
        f"- **Cost on CI:** at {best['cpu_docs_per_s']:.1f} docs/s on this CPU, ~3k new articles/day take "
        f"~{cpu_min:.1f} CPU-minutes per day for `{best['model']}` (GitHub runners are slower than this PC).",
        "",
        "## Known biases",
        "- Pairs were sampled from **e5-small** nearest neighbours (plus 50 random pairs). Same-event pairs "
        "that e5-small ranks as dissimilar are under-represented, which can flatter e5-small's recall.",
        "- Silver labels come from one LLM; where it is wrong, every model is scored against the same error.",
        "- Only titles + ledes are embedded; body text beyond ~200 words is ignored by design.",
    ]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "embedding_bakeoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    items = event_pair_labels()
    ids = {it["pair"]["a_id"] for it in items} | {it["pair"]["b_id"] for it in items}
    docs = load_docs(ids)
    texts = {i: doc_text(d["title"], d["excerpt"]) for i, d in docs.items()}
    items = [it for it in items if it["pair"]["a_id"] in texts and it["pair"]["b_id"] in texts]
    print(f"{len(items)} labelled pairs ({sum(it['source'] == 'gold' for it in items)} gold), {len(texts)} articles")

    results = [evaluate(k, items, texts, rng) for k in args.models.split(",")]
    out = ROOT / "data" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "embedding_bakeoff.json").write_text(json.dumps(
        {"pairs": [it["pair"]["pair_id"] for it in items], "y": [it["y"] for it in items],
         "source": [it["source"] for it in items], "results": results}, default=float), encoding="utf-8")
    write_report(results, items)
    print("→ reports/embedding_bakeoff.md")


if __name__ == "__main__":
    main()
