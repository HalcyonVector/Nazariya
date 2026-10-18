"""LLM pre-labels for event pairs (silver labels), and selection of the pairs a human reviews.

Pre-labelling sends each pair (titles + first ~700 characters) to an LLM on Groq with a condensed
version of labels/GUIDELINES.md and stores label, confidence and a short reason. It resumes where
it stopped and paces itself on Groq's rate-limit headers (free tier: ~8k tokens/minute).

Selection picks ~120 pairs for human review: every low-confidence LLM label, extra cross-lingual
pairs, and a random slice of confident ones. A "blind" subset is shown without the LLM suggestion
so we can measure how much seeing the suggestion sways the human (anchoring).

Usage:
    python -m nazariya.label.prelabel                 # pre-label (needs GROQ_API_KEY in .env)
    python -m nazariya.label.prelabel --select        # write the review queue
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from ..config import ROOT, load_dotenv
from .docs import load_docs, read_jsonl, sibling

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"
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


def parse_reset(value: str | None) -> float:
    """Groq reset headers look like '727ms', '7.5s', '1m26.4s'."""
    if not value:
        return 1.0
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", value):
        total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total or 1.0


class Groq:
    def __init__(self, key: str, model: str):
        self.client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=120)
        self.model = model
        self.remaining_tokens = 8000
        self.reset_tokens = 0.0

    def label(self, a: dict | None, b: dict | None) -> dict:
        if self.remaining_tokens < 2500:  # a request is ~1–1.5k tokens incl. reasoning
            time.sleep(self.reset_tokens + 0.5)
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": render(a, "A") + "\n\n" + render(b, "B")}],
            "temperature": 0,
            "max_tokens": 1500,
            "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
        }
        for attempt in range(6):
            r = self.client.post(GROQ_URL, json=body)
            self.remaining_tokens = int(r.headers.get("x-ratelimit-remaining-tokens", 8000))
            self.reset_tokens = parse_reset(r.headers.get("x-ratelimit-reset-tokens"))
            if r.status_code == 429:
                if "per day" in r.text.lower() or "tpd" in r.text.lower() or "rpd" in r.text.lower():
                    raise SystemExit(f"daily limit reached; re-run later to resume.\n{r.text[:300]}")
                time.sleep(float(r.headers.get("retry-after", 0)) or self.reset_tokens + 1)
                continue
            if r.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            out = json.loads(r.json()["choices"][0]["message"]["content"])
            label = out.get("label")
            if label not in LABELS:
                raise ValueError(f"bad label {label!r}")
            return {"label": label, "confidence": round(float(out.get("confidence", 0)), 2),
                    "reason": str(out.get("reason", ""))[:200]}
        raise RuntimeError("gave up after retries")


def prelabel(pairs_path: Path, model: str, limit: int | None = None) -> None:
    load_dotenv()
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY missing from .env")
    pairs = read_jsonl(pairs_path)
    out_path = sibling(pairs_path, "llm.jsonl")
    done = {r["pair_id"] for r in read_jsonl(out_path)}
    todo = [p for p in pairs if p["pair_id"] not in done][:limit]
    print(f"{len(pairs)} pairs, {len(done)} already pre-labelled, {len(todo)} to go with {model}")
    if not todo:
        return
    docs = load_docs({p["a_id"] for p in todo} | {p["b_id"] for p in todo})
    groq = Groq(key, model)
    t0 = time.monotonic()
    with out_path.open("a", encoding="utf-8") as f:
        for n, p in enumerate(todo, 1):
            try:
                res = groq.label(docs.get(p["a_id"]), docs.get(p["b_id"]))
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                print(f"  {p['pair_id']}: unparseable response ({e}); skipping, re-run to retry", file=sys.stderr)
                continue
            rec = {"pair_id": p["pair_id"], **res, "model": model, "prompt_version": PROMPT_VERSION,
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
        raise SystemExit(f"{len(missing)} pairs have no LLM label yet; finish pre-labelling first")

    chosen: list[str] = [pid for pid in originals if pid in human]  # already labelled (blind)

    def add(pids: list[str], k: int) -> None:
        rng.shuffle(pids)
        for pid in pids:
            if len(chosen) >= target or k <= 0:
                return
            if pid not in chosen:
                chosen.append(pid)
                k -= 1

    low = [pid for pid in originals if pid in llm and (llm[pid]["label"] == "unsure" or llm[pid]["confidence"] < 0.75)]
    add(low, 50)
    cross = [pid for pid in originals if pid in llm and pairs[pid]["langs"] in ("en-hi", "hi-en")]
    add(cross, 25)
    add([pid for pid in originals if pid in llm], target)

    fresh = [pid for pid in chosen if pid not in human]
    blind_ids = set(rng.sample(fresh, min(blind, len(fresh))))
    rng.shuffle(chosen)
    rep = [pid for pid, p in pairs.items() if p["repeat_of"] in set(chosen)][:repeats]
    queue = chosen + rep  # repeats last, long after their originals
    out = sibling(pairs_path, "review.json")
    out.write_text(json.dumps({"queue": queue, "blind": sorted(blind_ids | {r for r in rep if pairs[r]["repeat_of"] in blind_ids}),
                               "seed": seed}, indent=1), encoding="utf-8")
    print(f"review queue: {len(queue)} pairs ({len(chosen)} originals incl. {len(human & set(chosen))} already "
          f"labelled, {len(rep)} repeats); {len(blind_ids)} shown without the LLM suggestion → {out.relative_to(ROOT)}")
    print(f"  low-confidence LLM labels: {len(low)} (all included up to 50); cross-lingual in queue: "
          f"{sum(pairs[p]['langs'] in ('en-hi', 'hi-en') for p in chosen)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(ROOT / "labels" / "event_pairs_v1.jsonl"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, help="pre-label at most N pairs (for testing)")
    ap.add_argument("--select", action="store_true", help="write the human review queue")
    ap.add_argument("--target", type=int, default=120)
    ap.add_argument("--blind", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.select:
        select_review(Path(args.pairs), args.target, args.blind, args.repeats, args.seed)
    else:
        prelabel(Path(args.pairs), args.model, args.limit)


if __name__ == "__main__":
    main()
