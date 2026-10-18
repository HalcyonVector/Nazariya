"""LLM pre-labels for event pairs (silver labels), and selection of the pairs a human reviews.

Pre-labelling sends each pair (titles + first ~700 characters) to a labeller from config/llm.yaml
with a condensed version of labels/GUIDELINES.md and stores label, confidence, a short reason and
the provider/model that answered. It resumes where it stopped; when a host's daily quota is spent
the next host serving the same model takes over (e.g. Groq → Cerebras for gpt-oss-120b).

Labellers write to separate files: `primary` → labels/<pairs>.llm.jsonl (the silver set),
any other labeller → labels/<pairs>.llm-<name>.jsonl (second opinions, local models).

Selection picks ~120 pairs for human review: every low-confidence LLM label, extra cross-lingual
pairs, and a random slice of confident ones. A "blind" subset (incl. every repeated pair) is shown
without the LLM suggestion so we can measure how much seeing the suggestion sways the human.

Usage:
    python -m nazariya.label.prelabel                       # primary labeller (silver set)
    python -m nazariya.label.prelabel --labeller second_opinion
    python -m nazariya.label.prelabel --labeller local --only-review   # just the human-reviewed pairs
    python -m nazariya.label.prelabel --select              # write the review queue
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import ROOT
from ..llm import AllHostsExhausted, Labeller
from .docs import load_docs, read_jsonl, sibling

PROMPT_VERSION = "v1"
LLM_EXCERPT = 700
LABELS = ("same_event", "same_story", "unrelated", "unsure")

SYSTEM = """You label pairs of Indian news articles for a research dataset on news-event clustering.
Decide what each article is MAINLY about (headline and opening paragraph), then choose one label:

same_event  - both mainly report the same specific happening: same occurrence, same actors, about
              the same time. Different angle, framing, detail or language (English/Hindi) does NOT matter.
same_story  - same ongoing issue or saga, but they mainly report DIFFERENT happenings (different
              days' developments, a reaction that is itself new news, a follow-up, a later stage).
unrelated   - different issues, even if they share a person, party, place or broad theme.
unsure      - too little text to judge, extraction junk, or genuinely ambiguous.

Rules: a reaction quoted inside a report of event X is still X (same_event); a reaction that is
itself the headline news is a new event (same_story). Roundups/live blogs: judge the main subject.
General commentary on an issue vs. a report of one happening in it is same_story.

Respond with JSON only: {"label": "<one of the four>", "confidence": <0.0-1.0>,
"reason": "<at most 20 words, do not quote the articles>"}"""


def render(doc: dict | None, side: str) -> str:
    if not doc:
        return f"Article {side}: (missing)"
    excerpt = " ".join(doc["excerpt"][:LLM_EXCERPT].split())
    return (f"Article {side}\nOutlet: {doc['outlet']} | Published: {doc['published']} | Language: {doc['lang']}\n"
            f"Headline: {doc['title']}\nOpening: {excerpt}")


def output_path(pairs_path: Path, labeller: str) -> Path:
    return sibling(pairs_path, "llm.jsonl" if labeller == "primary" else f"llm-{labeller}.jsonl")


def prelabel(pairs_path: Path, labeller_name: str, limit: int | None = None, only_review: bool = False) -> None:
    labeller = Labeller(labeller_name)
    pairs = read_jsonl(pairs_path)
    if only_review:
        queue = set(json.loads(sibling(pairs_path, "review.json").read_text(encoding="utf-8"))["queue"])
        pairs = [p for p in pairs if p["pair_id"] in queue]
    out_path = output_path(pairs_path, labeller_name)
    done = {r["pair_id"] for r in read_jsonl(out_path)}
    todo = [p for p in pairs if p["pair_id"] not in done][:limit]
    hosts = ", ".join(f"{h.provider}/{h.model}" for h in labeller.hosts)
    print(f"{len(pairs)} pairs, {len(done)} done, {len(todo)} to go · labeller {labeller_name}: {hosts}")
    if not todo:
        return
    docs = load_docs({p["a_id"] for p in todo} | {p["b_id"] for p in todo})
    t0 = time.monotonic()
    with out_path.open("a", encoding="utf-8") as f:
        for n, p in enumerate(todo, 1):
            user = render(docs.get(p["a_id"]), "A") + "\n\n" + render(docs.get(p["b_id"]), "B")
            try:
                out, host = labeller.chat_json(SYSTEM, user)
                label = out.get("label")
                if label not in LABELS:
                    raise ValueError(f"bad label {label!r}")
            except AllHostsExhausted as e:
                print(f"stopping: {e}; re-run later to resume")
                break
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                print(f"  {p['pair_id']}: unusable response ({e}); skipping, re-run to retry", file=sys.stderr)
                continue
            rec = {"pair_id": p["pair_id"], "label": label,
                   "confidence": round(float(out.get("confidence") or 0), 2),
                   "reason": str(out.get("reason", ""))[:200], "provider": host.provider, "model": host.model,
                   "prompt_version": PROMPT_VERSION,
                   "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if n % 20 == 0 or n == len(todo):
                rate = n / (time.monotonic() - t0) * 60
                print(f"  {n}/{len(todo)}  ({rate:.1f} pairs/min)", flush=True)


def select_review(pairs_path: Path, target: int, blind: int, repeats: int, seed: int) -> None:
    rng = random.Random(seed)
    pairs = {p["pair_id"]: p for p in read_jsonl(pairs_path)}
    llm = {r["pair_id"]: r for r in read_jsonl(sibling(pairs_path, "llm.jsonl"))}
    human = {r["pair_id"] for r in read_jsonl(sibling(pairs_path, "labels.jsonl"))}
    originals = [pid for pid, p in pairs.items() if p["repeat_of"] is None]
    missing = [pid for pid in originals if pid not in llm and pid not in human]
    if missing:
        # e.g. Groq's daily token cap hit mid-run: review from what's labelled, finish the rest later
        print(f"note: {len(missing)} pairs have no LLM label yet; selecting from the labelled ones")

    chosen: list[str] = [pid for pid in originals if pid in human]  # already labelled (blind)

    def add(pids: list[str], k: int) -> None:
        rng.shuffle(pids)
        for pid in pids:
            if len(chosen) >= target or k <= 0:
                return
            if pid not in chosen:
                chosen.append(pid)
                k -= 1

    # 1. self-consistency: originals that have a swapped repeat; original and repeat are both blind
    repeat_of = {p["repeat_of"]: pid for pid, p in pairs.items() if p["repeat_of"]}
    rep_origs = [o for o in repeat_of if o in human]
    rep_origs += rng.sample([o for o in repeat_of if o in llm and o not in human],
                            k=min(repeats, sum(1 for o in repeat_of if o in llm and o not in human)))
    rep_origs = rep_origs[:repeats]
    add(rep_origs, len(rep_origs))
    # 2. the LLM itself is unsure
    low = [pid for pid in originals if pid in llm and (llm[pid]["label"] == "unsure" or llm[pid]["confidence"] < 0.75)]
    add(low, 50)
    # 3. LLM label contradicts embedding similarity: the most informative pairs to check
    surprising = [pid for pid in originals if pid in llm and (
        (llm[pid]["label"] == "same_event" and pairs[pid]["band"] in ("random", "nn1", "nn2"))
        or (llm[pid]["label"] == "unrelated" and pairs[pid]["band"] == "nn5"))]
    add(surprising, 25)
    # 4. extra cross-lingual pairs, 5. random fill
    cross = [pid for pid in originals if pid in llm and pairs[pid]["langs"] in ("en-hi", "hi-en")]
    add(cross, 25)
    add([pid for pid in originals if pid in llm], target)

    blind_ids = set(rep_origs) - human
    fresh = [pid for pid in chosen if pid not in human and pid not in blind_ids]
    blind_ids |= set(rng.sample(fresh, max(0, min(blind - len(blind_ids), len(fresh)))))
    rng.shuffle(chosen)
    rep = [repeat_of[o] for o in rep_origs]
    queue = chosen + rep  # repeats last, long after their originals
    out = sibling(pairs_path, "review.json")
    out.write_text(json.dumps({"queue": queue, "blind": sorted(blind_ids | set(rep)), "seed": seed}, indent=1),
                   encoding="utf-8")
    print(f"review queue: {len(queue)} pairs ({len(chosen)} originals incl. {len(human & set(chosen))} already "
          f"labelled, {len(rep)} blind repeats); {len(blind_ids)} originals shown without the LLM suggestion "
          f"→ {out.relative_to(ROOT)}")
    print(f"  low-confidence: {len(low)} · label/similarity contradictions: {len(surprising)} · "
          f"cross-lingual in queue: {sum(pairs[p]['langs'] in ('en-hi', 'hi-en') for p in chosen)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(ROOT / "labels" / "event_pairs_v1.jsonl"))
    ap.add_argument("--labeller", default="primary", help="labeller name from config/llm.yaml")
    ap.add_argument("--limit", type=int, help="pre-label at most N pairs (for testing)")
    ap.add_argument("--only-review", action="store_true", help="only pairs in the human review queue")
    ap.add_argument("--select", action="store_true", help="write the human review queue")
    ap.add_argument("--target", type=int, default=120)
    ap.add_argument("--blind", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.select:
        select_review(Path(args.pairs), args.target, args.blind, args.repeats, args.seed)
    else:
        prelabel(Path(args.pairs), args.labeller, args.limit, args.only_review)


if __name__ == "__main__":
    main()
