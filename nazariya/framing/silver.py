"""Silver labels for framing and entity stance (LLM), per labels/FRAMING_CODEBOOK.md.

Sample: articles with body text from events covered by ≥ MIN_OUTLETS outlets (where comparing
framing across outlets matters), excluding wire copy, capped per event and per outlet. Stance
targets per article: its PER_ARTICLE_TARGETS most-mentioned people/parties (from entity links).

Files (ids, labels and short paraphrases only; committable):
    labels/framing_sample_v1.jsonl             the sample
    labels/framing_v1.<labeller>.jsonl         one labeller's labels

Usage:
    python -m nazariya.framing.silver --sample            # draw the sample (once)
    python -m nazariya.framing.silver --labeller local    # label (resumable)
    python -m nazariya.framing.silver --labeller primary --limit 150
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

from sqlalchemy import select

from ..config import ROOT
from ..db import article_events, article_text, articles, entities, events, get_engine, mentions
from ..label.docs import read_jsonl
from ..llm import AllHostsExhausted, Labeller
from . import BASES, FRAMES, STANCES

SAMPLE = ROOT / "labels" / "framing_sample_v1.jsonl"
MIN_OUTLETS = 5
PER_EVENT = 12
PER_OUTLET = 80
TARGET = 1500
PER_ARTICLE_TARGETS = 3
EXCERPT = 1500
PROMPT_VERSION = "v1"

SYSTEM = f"""You annotate Indian news articles (English or Hindi) for a media-framing research project.

FRAMES: the angle from which the article presents its subject. Choose one primary frame and up to two
secondary frames from: {", ".join(FRAMES)}.
economic = costs, prices, jobs, markets · capacity_resources = shortages/availability of resources ·
morality = right/wrong, religious or ethical duty · fairness_equality = discrimination, rights of groups ·
legality = courts, laws, constitutional questions · policy = whether a scheme/policy works or should change ·
crime_punishment = crimes, policing, arrests · security_defense = national/public security, military, terror ·
health_safety = health, accidents, disasters · quality_of_life = effects on daily living ·
cultural_identity = religion, caste, language, tradition, regional identity · public_opinion = protests, polls, public mood ·
political = party competition, elections, political strategy · external_reputation = foreign relations, India's standing abroad ·
other = none of these.
A politician speaking does not make it 'political'; it is political when the angle is party competition or power.

STANCE toward each listed entity: how the ARTICLE portrays it (favourable, neutral, unfavourable, unclear),
and the basis: 'wording' (the article's own voice, headline, word choice), 'quotes' (only via people quoted),
or 'events' (the news itself is good/bad for the entity, reported plainly). Good news reported plainly is
neutral with basis events. An attack quoted together with the target's response is closer to neutral.

EVIDENCE: a paraphrase of at most 15 words. Never copy sentences from the article.

Respond with JSON only:
{{"primary_frame": "<frame>", "secondary_frames": ["<frame>", ...],
 "frame_evidence": "<paraphrase>",
 "stances": [{{"id": "<entity id>", "stance": "<stance>", "basis": "<basis>", "evidence": "<paraphrase>"}}]}}"""


def draw_sample(seed: int) -> None:
    if SAMPLE.exists():
        raise SystemExit(f"{SAMPLE} exists; it defines the labelled set, so it is never redrawn")
    rng = random.Random(seed)
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(articles.c.id, articles.c.outlet_id, articles.c.lang, article_events.c.event_id)
            .join(article_events, article_events.c.article_id == articles.c.id)
            .join(events, events.c.id == article_events.c.event_id)
            .join(article_text, article_text.c.article_id == articles.c.id)
            .where(events.c.n_outlets >= MIN_OUTLETS, articles.c.status == "extracted",
                   articles.c.syndicated_of.is_(None), articles.c.agency.is_(None),
                   articles.c.lang.in_(["en", "hi"]))
        ).all()
    by_event: dict[int, list] = defaultdict(list)
    for r in rows:
        by_event[r.event_id].append(r)
    pool = []
    for ev_rows in by_event.values():
        rng.shuffle(ev_rows)
        pool += ev_rows[:PER_EVENT]
    rng.shuffle(pool)
    per_outlet: Counter = Counter()
    chosen = []
    for r in pool:
        if per_outlet[r.outlet_id] >= PER_OUTLET:
            continue
        per_outlet[r.outlet_id] += 1
        chosen.append({"article_id": r.id, "event_id": r.event_id, "outlet": r.outlet_id, "lang": r.lang})
        if len(chosen) >= TARGET:
            break
    SAMPLE.write_text("".join(json.dumps(c) + "\n" for c in chosen), encoding="utf-8")
    print(f"{len(chosen)} articles from {len({c['event_id'] for c in chosen})} events, "
          f"{len(per_outlet)} outlets, langs {Counter(c['lang'] for c in chosen)} → {SAMPLE.relative_to(ROOT)}")


def load_items(ids: list[int]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with get_engine().connect() as conn:
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            for r in conn.execute(
                select(articles.c.id, articles.c.title, article_text.c.text)
                .join(article_text, article_text.c.article_id == articles.c.id)
                .where(articles.c.id.in_(chunk))
            ):
                out[r.id] = {"title": r.title, "text": r.text, "targets": []}
            for r in conn.execute(
                select(mentions.c.article_id, mentions.c.qid, entities.c.label_en, entities.c.type, mentions.c.n)
                .join(entities, entities.c.qid == mentions.c.qid)
                .where(mentions.c.article_id.in_(chunk), entities.c.type.in_(["person", "party", "foreign_person"]))
                .order_by(mentions.c.article_id, mentions.c.n.desc())
            ):
                if r.article_id in out and len(out[r.article_id]["targets"]) < PER_ARTICLE_TARGETS:
                    out[r.article_id]["targets"].append({"id": r.qid, "label": r.label_en})
    return out


def validate(out: dict, target_ids: set[str]) -> dict:
    primary = out.get("primary_frame")
    if primary not in FRAMES:
        raise ValueError(f"bad primary frame {primary!r}")
    secondary = [f for f in (out.get("secondary_frames") or []) if f in FRAMES and f != primary][:2]
    stances = []
    for s in out.get("stances") or []:
        if s.get("id") in target_ids and s.get("stance") in STANCES:
            stances.append({"qid": s["id"], "stance": s["stance"],
                             "basis": s.get("basis") if s.get("basis") in BASES else None,
                             "evidence": str(s.get("evidence", ""))[:160]})
    return {"primary": primary, "secondary": secondary,
            "frame_evidence": str(out.get("frame_evidence", ""))[:160], "stances": stances}


def label(labeller_name: str, limit: int | None) -> None:
    sample = read_jsonl(SAMPLE)
    out_path = ROOT / "labels" / f"framing_v1.{labeller_name}.jsonl"
    done = {r["article_id"] for r in read_jsonl(out_path)}
    todo = [s for s in sample if s["article_id"] not in done][:limit]
    labeller = Labeller(labeller_name)
    print(f"{len(sample)} sampled, {len(done)} done, {len(todo)} to go · "
          + ", ".join(f"{h.provider}/{h.model}" for h in labeller.hosts))
    items = load_items([s["article_id"] for s in todo])
    t0 = time.monotonic()
    with out_path.open("a", encoding="utf-8") as f:
        for n, s in enumerate(todo, 1):
            it = items.get(s["article_id"])
            if not it:
                continue
            excerpt = " ".join(it["text"][:EXCERPT].split())
            targets = "\n".join(f"{t['id']}: {t['label']}" for t in it["targets"]) or "(none)"
            user = f"Headline: {it['title']}\nArticle: {excerpt}\n\nEntities for stance:\n{targets}"
            try:
                raw, host = labeller.chat_json(SYSTEM, user, max_tokens=900)
                rec = validate(raw, {t["id"] for t in it["targets"]})
            except AllHostsExhausted as e:
                print(f"stopping: {e}")
                break
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                print(f"  {s['article_id']}: unusable response ({e})", file=sys.stderr)
                continue
            f.write(json.dumps({"article_id": s["article_id"], **rec, "provider": host.provider,
                                "model": host.model, "prompt_version": PROMPT_VERSION,
                                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                               ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ({n / (time.monotonic() - t0) * 60:.1f}/min)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--labeller", default="local")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    if args.sample:
        draw_sample(args.seed)
    else:
        label(args.labeller, args.limit)


if __name__ == "__main__":
    main()
