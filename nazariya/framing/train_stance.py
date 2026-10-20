"""Train and evaluate the entity-stance classifier on silver labels (PROVISIONAL).

Each example is one (article, entity) pair: input "<entity name> | <headline + body start>",
label favourable / neutral / unfavourable / unclear (labels/FRAMING_CODEBOOK.md §2). Splits by
event as in train.py; the reference labeller's articles are held out as an independent test.

Usage:
    python -m nazariya.framing.train_stance                 # tfidf + xlmr → reports/stance_eval.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sqlalchemy import select

from ..config import ROOT
from ..db import entities, get_engine
from ..label.docs import read_jsonl
from . import STANCES
from .silver import SAMPLE, load_items
from .train import BASES, INPUT_CHARS, MODELS_DIR

MAIN = [STANCES.index(s) for s in ("favourable", "neutral", "unfavourable")]


def load_dataset(train_labeller: str, ref_labeller: str):
    sample = {s["article_id"]: s for s in read_jsonl(SAMPLE)}
    silver = {r["article_id"]: r for r in read_jsonl(ROOT / "labels" / f"framing_v1.{train_labeller}.jsonl")}
    ref = {r["article_id"]: r for r in read_jsonl(ROOT / "labels" / f"framing_v1.{ref_labeller}.jsonl")}
    items = load_items(sorted(set(silver) | set(ref)))
    qids = {s["qid"] for r in [*silver.values(), *ref.values()] for s in r["stances"]}
    with get_engine().connect() as conn:
        names = dict(conn.execute(select(entities.c.qid, entities.c.label_en).where(entities.c.qid.in_(qids))).all())

    def bucket(event_id: int) -> float:
        return int(hashlib.md5(str(event_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    def examples(rec, aid):
        it = items.get(aid)
        if not it:
            return []
        body = " ".join(it["text"][:INPUT_CHARS].split())
        return [(aid, f"{names.get(s['qid'], s['qid'])} | {it['title']}\n{body}", STANCES.index(s["stance"]),
                 sample[aid]["lang"], s["qid"]) for s in rec["stances"] if s["qid"] in names]

    split = {"train": [], "val": [], "ref": []}
    for aid, rec in ref.items():
        split["ref"] += examples(rec, aid)
    for aid, rec in silver.items():
        if aid in ref:
            continue
        split["val" if bucket(sample[aid]["event_id"]) < 0.15 else "train"] += examples(rec, aid)
    return split


def score(pred: np.ndarray, data) -> dict:
    y = np.array([d[2] for d in data])
    return {"n": len(data), "acc": float(accuracy_score(y, pred)),
            "macro_f1_3": float(f1_score(y, pred, labels=MAIN, average="macro", zero_division=0)),
            "dist": {STANCES[k]: int((y == k).sum()) for k in range(len(STANCES))}}


def run_tfidf(split):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=200_000, sublinear_tf=True)
    X = vec.fit_transform([d[1] for d in split["train"]])
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced").fit(X, [d[2] for d in split["train"]])
    pred = lambda data: clf.predict(vec.transform([d[1] for d in data]))
    return score(pred(split["val"]), split["val"]), score(pred(split["ref"]), split["ref"])


def run_transformer(key: str, split, epochs: int, batch: int, lr: float):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASES[key])
    model = AutoModelForSequenceClassification.from_pretrained(BASES[key], num_labels=len(STANCES)).to(dev)
    counts = np.bincount([d[2] for d in split["train"]], minlength=len(STANCES)).astype(np.float32)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(STANCES), device=dev)  # class balance
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    def encode(data):
        return tok([d[1] for d in data], truncation=True, max_length=256, padding=True, return_tensors="pt")

    def predict(data) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(data), 64):
                enc = encode(data[i:i + 64])
                with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                    out.append(model(**{k: v.to(dev) for k, v in enc.items()}).logits.float().argmax(1).cpu().numpy())
        return np.concatenate(out)

    train = split["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    rng = np.random.default_rng(0)
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        t0 = time.monotonic()
        order = rng.permutation(len(train))
        for i in range(0, len(order), batch):
            data = [train[j] for j in order[i:i + batch]]
            enc = encode(data)
            y = torch.tensor([d[2] for d in data], device=dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                loss = loss_fn(model(**{k: v.to(dev) for k, v in enc.items()}).logits.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        val = score(predict(split["val"]), split["val"])
        print(f"  {key} epoch {ep + 1}: val acc {val['acc']:.3f} macro-F1(3) {val['macro_f1_3']:.3f} "
              f"({time.monotonic() - t0:.0f}s)", flush=True)
        if val["macro_f1_3"] > best:
            best, best_state = val["macro_f1_3"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    out_dir = MODELS_DIR / f"stance-{key}"
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "labels.json").write_text(json.dumps(list(STANCES)), encoding="utf-8")
    # stable model version for the pipeline (file timestamps change on every CI download)
    (out_dir / "version.txt").write_text(datetime.now(timezone.utc).strftime("%Y%m%d%H%M"), encoding="utf-8")
    return score(predict(split["val"]), split["val"]), score(predict(split["ref"]), split["ref"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tfidf,xlmr")
    ap.add_argument("--train-labeller", default="local_long")
    ap.add_argument("--ref-labeller", default="primary")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    split = load_dataset(args.train_labeller, args.ref_labeller)
    print({k: len(v) for k, v in split.items()})
    results = {}
    for key in args.models.split(","):
        print(f"== {key}")
        results[key] = run_tfidf(split) if key == "tfidf" else run_transformer(key, split, args.epochs, args.batch, args.lr)
        print(f"   val {results[key][0]}\n   ref {results[key][1]}")
    lines = [
        "# Entity stance classifier evaluation",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `python -m nazariya.framing.train_stance`.",
        "",
        f"> **PROVISIONAL.** Trained on LLM silver labels (`{args.train_labeller}`); `ref` = pairs labelled by "
        f"`{args.ref_labeller}`, never trained on. Macro-F1 is over favourable/neutral/unfavourable.",
        "",
        f"Pairs: train {len(split['train'])}, val {len(split['val'])}, ref {len(split['ref'])}.",
        "",
        "| model | val acc | val macro-F1 | ref acc | ref macro-F1 |",
        "|---|---|---|---|---|",
    ]
    for key, (val, ref) in results.items():
        lines.append(f"| {key} | {val['acc']:.3f} | {val['macro_f1_3']:.3f} | {ref['acc']:.3f} | {ref['macro_f1_3']:.3f} |")
    lines += ["", f"Label distribution (train silver): {split and {s: sum(1 for d in split['train'] if d[2] == i) for i, s in enumerate(STANCES)}}"]
    (ROOT / "reports" / "stance_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("→ reports/stance_eval.md")


if __name__ == "__main__":
    main()
