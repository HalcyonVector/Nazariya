"""Evaluate the entity linker with LLM judgements (PROVISIONAL until the human labelling session).

For a sample of recent articles with body text (half English, half Hindi), the linker runs on the
headline + first EXCERPT characters, and an LLM judge sees the same text and:
  * says, for each linked entity, whether the article really refers to it → precision
  * lists Indian politicians / parties / institutions named in the text but not linked. Each missed
    name is then checked against the alias table: linkable (in the table, so a linker miss) or
    not in the table (a coverage gap) → recall against the table, and table coverage.

Usage:
    python -m nazariya.eval.entities                    # local judge (no API quota)
    python -m nazariya.eval.entities --labeller primary --n 100
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from ..config import ROOT
from ..db import article_text, articles, get_engine
from ..entities.linker import Linker
from ..llm import AllHostsExhausted, Labeller

EXCERPT = 1800

SYSTEM = """You check named-entity links in Indian news articles (English or Hindi).
You get an article excerpt and a list of entities that an automatic linker found in it.
1. For each listed entity, answer true if the excerpt really refers to that specific entity
   (a person, party, state, institution or country), false if the match is a different person,
   an ordinary word, or anything else.
2. List the Indian politicians, political parties and government institutions that ARE named in the
   excerpt but are NOT in the list. Use their usual English names. Do not list places, journalists,
   film stars, sportspeople or ordinary officials.
Respond with JSON only: {"verdicts": {"<id>": true|false, ...}, "missed": ["<name>", ...]}"""


def sample(n: int, seed: int) -> list:
    since = datetime.now(timezone.utc) - timedelta(days=7)
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(articles.c.id, articles.c.lang, articles.c.title, article_text.c.text)
            .join(article_text, article_text.c.article_id == articles.c.id)
            .where(articles.c.status == "extracted", articles.c.published_at >= since,
                   articles.c.lang.in_(["en", "hi"]))
        ).all()
    rng = random.Random(seed)
    out = []
    for lang in ("en", "hi"):
        pool = [r for r in rows if r.lang == lang]
        out += rng.sample(pool, min(n // 2, len(pool)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--labeller", default="local")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    linker = Linker()
    judge = Labeller(args.labeller)
    arts = sample(args.n, args.seed)
    per_lang = {l: Counter() for l in ("en", "hi")}
    examples = {"false_links": [], "missed_in_table": [], "not_in_table": Counter()}
    done = 0
    for a in arts:
        excerpt = (a.text or "")[:EXCERPT]
        links = linker.link(a.title, excerpt)
        listed = {q: f"{linker.entities[q]['label_en']} ({linker.entities[q].get('description', '')[:60]})" for q in links}
        user = (f"Headline: {a.title}\nExcerpt: {' '.join(excerpt.split())}\n\nLinked entities:\n"
                + ("\n".join(f"{q}: {d}" for q, d in listed.items()) or "(none)"))
        try:
            out, _ = judge.chat_json(SYSTEM, user, max_tokens=600)
        except AllHostsExhausted:
            break
        except Exception as e:  # unparseable response: skip the article
            print(f"  skip {a.id}: {e}")
            continue
        verdicts = out.get("verdicts") or {}
        c = per_lang[a.lang]
        for q in links:
            ok = verdicts.get(q)
            if ok is None:
                continue
            c["links_judged"] += 1
            c["links_correct"] += bool(ok)
            if not ok and len(examples["false_links"]) < 25:
                examples["false_links"].append(f"{linker.entities[q]['label_en']} ← “{a.title[:70]}”")
        for name in out.get("missed") or []:
            if not isinstance(name, str) or not name.strip():
                continue
            found = linker.link(name)
            if found and not (set(found) & set(links)):
                c["missed_in_table"] += 1
                if len(examples["missed_in_table"]) < 25:
                    examples["missed_in_table"].append(f"{name} ← “{a.title[:70]}”")
            elif not found:
                c["not_in_table"] += 1
                examples["not_in_table"][name.strip()] += 1
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(arts)}", flush=True)

    def metrics(c: Counter) -> dict:
        p = c["links_correct"] / c["links_judged"] if c["links_judged"] else float("nan")
        found = c["links_correct"]
        r = found / (found + c["missed_in_table"]) if found + c["missed_in_table"] else float("nan")
        cov = (found + c["missed_in_table"]) / (found + c["missed_in_table"] + c["not_in_table"]) \
            if found + c["missed_in_table"] + c["not_in_table"] else float("nan")
        keys = ("links_judged", "links_correct", "missed_in_table", "not_in_table")
        return {"precision": p, "recall_vs_table": r, "table_coverage": cov, **{k: c[k] for k in keys}}

    total = per_lang["en"] + per_lang["hi"]
    rows = {"English": metrics(per_lang["en"]), "Hindi": metrics(per_lang["hi"]), "All": metrics(total)}
    lines = [
        "# Entity linking evaluation",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `python -m nazariya.eval.entities` "
        f"· judge: `{args.labeller}` ({', '.join(f'{h.provider}/{h.model}' for h in judge.hosts)}) · "
        f"{done} articles · linker version `{linker.version}`.",
        "",
        "> **PROVISIONAL.** An LLM judges the links and lists misses; LLM judges make mistakes of their own. "
        "To be re-checked in the human labelling session.",
        "",
        "| | precision | recall vs. alias table | table coverage | links judged | missed (in table) | not in table |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in rows.items():
        lines.append(f"| {name} | {m['precision']:.3f} | {m['recall_vs_table']:.3f} | {m['table_coverage']:.3f} | "
                     f"{m['links_judged']} | {m['missed_in_table']} | {m['not_in_table']} |")
    lines += [
        "",
        "- **precision**: share of the linker's links the judge confirms.",
        "- **recall vs. alias table**: of correctly linkable mentions (confirmed links + names the judge found "
        "that the table could have linked), the share the linker caught.",
        "- **table coverage**: share of judge-listed political names the table contains at all.",
        "",
        "## Examples: links the judge rejected",
        *[f"- {x}" for x in examples["false_links"]],
        "",
        "## Examples: in the table but not linked",
        *[f"- {x}" for x in examples["missed_in_table"]],
        "",
        "## Most frequent names not in the table",
        *[f"- {n} ({k})" for n, k in examples["not_in_table"].most_common(30)],
    ]
    (ROOT / "reports" / "entities_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "data" / "eval").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "eval" / "entities_eval.json").write_text(json.dumps(rows, default=float, indent=1), encoding="utf-8")
    for name, m in rows.items():
        print(f"{name:8s} P={m['precision']:.3f} R(table)={m['recall_vs_table']:.3f} coverage={m['table_coverage']:.3f}")
    print("→ reports/entities_eval.md")


if __name__ == "__main__":
    main()
