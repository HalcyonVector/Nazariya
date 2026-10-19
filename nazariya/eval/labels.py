"""Load evaluation labels for event pairs, keeping track of where each label came from.

Binary task: same_event (positive) vs not same_event (same_story and unrelated collapse to
negative). `unsure` and repeated pairs are dropped. Human labels ("gold") win over LLM labels
("silver"); every metric is reported per source so provisional silver numbers are never
mistaken for gold ones.
"""

from __future__ import annotations

from pathlib import Path

from ..config import ROOT
from ..label.docs import read_jsonl, sibling

PAIRS = ROOT / "labels" / "event_pairs_v1.jsonl"


def event_pair_labels(pairs_path: Path = PAIRS) -> list[dict]:
    """[{pair, y (1 = same event), source: 'gold'|'silver'}] for every usable original pair."""
    pairs = [p for p in read_jsonl(pairs_path) if p["repeat_of"] is None]
    gold = {r["pair_id"]: r["label"] for r in read_jsonl(sibling(pairs_path, "labels.jsonl"))}
    silver = {r["pair_id"]: r["label"] for r in read_jsonl(sibling(pairs_path, "llm.jsonl"))}
    out = []
    for p in pairs:
        if p["pair_id"] in gold:
            label, source = gold[p["pair_id"]], "gold"
        elif p["pair_id"] in silver:
            label, source = silver[p["pair_id"]], "silver"
        else:
            continue
        if label == "unsure":
            continue
        out.append({"pair": p, "y": int(label == "same_event"), "source": source})
    return out
