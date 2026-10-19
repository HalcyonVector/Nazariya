"""Train and evaluate frame classifiers on silver labels (PROVISIONAL until human gold labels exist).

Task: multi-label over the 15 frames (primary + secondary = positives); the predicted primary frame
is the highest-scoring one. Inputs: headline + first INPUT_CHARS characters of the body.

Splits: by event (all articles about one event fall on one side), so the model can't score well by
memorising the event. The articles labelled by the reference labeller (gpt-oss) are always held out
and used as an independent test of agreement with a different, stronger model.

Models: TF-IDF + logistic regression baseline, then fine-tuned transformers (XLM-R base, MuRIL).

Usage:
    python -m nazariya.framing.train                       # all models → reports/framing_eval.md
    python -m nazariya.framing.train --models tfidf,xlmr
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import f1_score

from ..config import ROOT
from ..label.docs import read_jsonl
from . import FRAMES
from .silver import SAMPLE, load_items

INPUT_CHARS = 2000
BASES = {"xlmr": "FacebookAI/xlm-roberta-base", "muril": "google/muril-base-cased"}
MODELS_DIR = ROOT / "data" / "models"


def load_dataset(train_labeller: str, ref_labeller: str):
    sample = {s["article_id"]: s for s in read_jsonl(SAMPLE)}
    silver = {r["article_id"]: r for r in read_jsonl(ROOT / "labels" / f"framing_v1.{train_labeller}.jsonl")}
    ref = {r["article_id"]: r for r in read_jsonl(ROOT / "labels" / f"framing_v1.{ref_labeller}.jsonl")}
    ids = sorted(set(silver) | set(ref))
    items = load_items(ids)
    texts = {i: f"{it['title']}\n{' '.join(it['text'][:INPUT_CHARS].split())}" for i, it in items.items()}

    def multi_hot(r) -> np.ndarray:
        y = np.zeros(len(FRAMES), dtype=np.float32)
        for f in [r["primary"], *r["secondary"]]:
            y[FRAMES.index(f)] = 1
        return y

    def bucket(event_id: int) -> float:
        return int(hashlib.md5(str(event_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    split = {"train": [], "val": [], "ref": []}
    for aid in ids:
        if aid not in texts:
            continue
        if aid in ref:
            split["ref"].append((aid, texts[aid], multi_hot(ref[aid]), FRAMES.index(ref[aid]["primary"]),
                                 sample[aid]["lang"]))
        elif aid in silver:
            part = "val" if bucket(sample[aid]["event_id"]) < 0.15 else "train"
            r = silver[aid]
            split[part].append((aid, texts[aid], multi_hot(r), FRAMES.index(r["primary"]), sample[aid]["lang"]))
    return split


def score(probs: np.ndarray, data) -> dict:
    y = np.stack([d[2] for d in data])
    primary = np.array([d[3] for d in data])
    langs = np.array([d[4] for d in data])
    pred = (probs >= 0.5).astype(int)
    pred[np.arange(len(pred)), probs.argmax(1)] = 1  # always predict at least the top frame
    out = {"n": len(data),
           "primary_acc": float((probs.argmax(1) == primary).mean()),
           "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
           "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0))}
    for lang in ("en", "hi"):
        m = langs == lang
        if m.any():
            out[f"primary_acc_{lang}"] = float((probs[m].argmax(1) == primary[m]).mean())
    return out


def run_tfidf(split) -> tuple[dict, dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=200_000, sublinear_tf=True)
    X = vec.fit_transform([d[1] for d in split["train"]])
    Y = np.stack([d[2] for d in split["train"]])
    keep = Y.sum(0) > 0  # frames never seen in training can't be learned
    clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
    clf.fit(X, Y[:, keep])

    def predict(data):
        p = np.zeros((len(data), len(FRAMES)))
        p[:, keep] = clf.predict_proba(vec.transform([d[1] for d in data]))
        return p
    return score(predict(split["val"]), split["val"]), score(predict(split["ref"]), split["ref"])


def run_transformer(key: str, split, epochs: int, batch: int, lr: float) -> tuple[dict, dict]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASES[key])
    model = AutoModelForSequenceClassification.from_pretrained(
        BASES[key], num_labels=len(FRAMES), problem_type="multi_label_classification").to(dev)

    def encode(data):
        enc = tok([d[1] for d in data], truncation=True, max_length=256, padding=True, return_tensors="pt")
        return enc, torch.tensor(np.stack([d[2] for d in data]))

    def predict(data) -> np.ndarray:
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(data), 64):
                enc, _ = encode(data[i:i + 64])
                with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                    logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits
                outs.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.concatenate(outs)

    train = split["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    rng = np.random.default_rng(0)
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train))
        t0 = time.monotonic()
        for i in range(0, len(order), batch):
            enc, y = encode([train[j] for j in order[i:i + batch]])
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                loss = model(**{k: v.to(dev) for k, v in enc.items()}, labels=y.to(dev)).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        val = score(predict(split["val"]), split["val"])
        print(f"  {key} epoch {ep + 1}: val primary_acc {val['primary_acc']:.3f} macro_f1 {val['macro_f1']:.3f} "
              f"({time.monotonic() - t0:.0f}s)", flush=True)
        if val["macro_f1"] > best:
            best, best_state = val["macro_f1"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    out_dir = MODELS_DIR / f"framing-{key}"
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "frames.json").write_text(json.dumps(list(FRAMES)), encoding="utf-8")
    return score(predict(split["val"]), split["val"]), score(predict(split["ref"]), split["ref"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tfidf,xlmr,muril")
    ap.add_argument("--train-labeller", default="local_long")
    ap.add_argument("--ref-labeller", default="primary")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()

    split = load_dataset(args.train_labeller, args.ref_labeller)
    print({k: len(v) for k, v in split.items()})
    results = {}
    for key in args.models.split(","):
        print(f"== {key}")
        results[key] = run_tfidf(split) if key == "tfidf" else run_transformer(key, split, args.epochs, args.batch, args.lr)
        print(f"   val {results[key][0]}\n   ref {results[key][1]}")

    lines = [
        "# Frame classifier evaluation",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `python -m nazariya.framing.train`.",
        "",
        f"> **PROVISIONAL.** Trained on LLM silver labels (`{args.train_labeller}`); `val` = held-out events "
        f"labelled by the same LLM; `ref` = articles labelled by a different, stronger LLM (`{args.ref_labeller}`), "
        "never trained on. Human gold labels come at the end of the build.",
        "",
        f"Splits: train {len(split['train'])}, val {len(split['val'])} (by event), ref {len(split['ref'])}. "
        f"Input: headline + first {INPUT_CHARS} characters. 15 frames, multi-label (primary + secondary).",
        "",
        "| model | val primary acc | val macro-F1 | ref primary acc (en / hi) | ref macro-F1 | ref micro-F1 |",
        "|---|---|---|---|---|---|",
    ]
    for key, (val, ref) in results.items():
        lines.append(f"| {key} | {val['primary_acc']:.3f} | {val['macro_f1']:.3f} | {ref['primary_acc']:.3f} "
                     f"({ref.get('primary_acc_en', float('nan')):.2f} / {ref.get('primary_acc_hi', float('nan')):.2f}) | "
                     f"{ref['macro_f1']:.3f} | {ref['micro_f1']:.3f} |")
    (ROOT / "reports" / "framing_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("→ reports/framing_eval.md")


if __name__ == "__main__":
    main()
