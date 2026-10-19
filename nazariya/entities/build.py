"""Build the entity alias table (config/entities/entities.jsonl) from Wikidata (CC0).

Classes:
  person         Indian politicians: human, occupation politician, Indian citizen or member of an
                 Indian party, ≥ 8 Wikipedia
                 sitelinks (a notability proxy), alive or died after 1990
  party          political parties in India with ≥ 5 sitelinks, not dissolved before 2000
  state          states and union territories of India
  institution, organisation, country, foreign_person
                 hand-picked in config/entities/extra.yaml, resolved via Wikidata search

For each entity: English and Hindi labels and aliases, the English description and the sitelink
count (used as prominence to break ties between ambiguous names). The output is committed so
every link the pipeline makes can be traced to a reviewable row.

Usage:
    python -m nazariya.entities.build
"""

from __future__ import annotations

import json
import re
import time

import httpx
import yaml

from ..config import CONFIG

UA = {"User-Agent": "NazariyaBot/0.1 (https://github.com/HalcyonVector/Nazariya)"}
SPARQL = "https://query.wikidata.org/sparql"
API = "https://www.wikidata.org/w/api.php"
OUT = CONFIG / "entities" / "entities.jsonl"
LANGS = ("en", "hi")
# Since 2024 many person items keep their name under the language-neutral "mul" label instead of
# "en" (e.g. Rahul Gandhi); "mul" is used as the fallback for English names.
FETCH_LANGS = ("en", "hi", "mul")

QUERIES = {
    # citizenship OR party membership: some leading figures (e.g. Narendra Modi) have no P27 statement
    "person": """SELECT DISTINCT ?e ?sl WHERE {
        ?e wdt:P31 wd:Q5; wdt:P106 wd:Q82955; wikibase:sitelinks ?sl.
        { ?e wdt:P27 wd:Q668 } UNION { ?e wdt:P102 ?party . ?party wdt:P17 wd:Q668 }
        FILTER(?sl >= 8)
        FILTER NOT EXISTS { ?e wdt:P570 ?d . FILTER(YEAR(?d) < 1990) } }""",
    "party": """SELECT DISTINCT ?e ?sl WHERE {
        ?e wdt:P31/wdt:P279* wd:Q7278; wdt:P17 wd:Q668; wikibase:sitelinks ?sl.
        FILTER(?sl >= 5)
        FILTER NOT EXISTS { ?e wdt:P576 ?d . FILTER(YEAR(?d) < 2000) } }""",
    "state": """SELECT DISTINCT ?e ?sl WHERE {
        VALUES ?t { wd:Q12443800 wd:Q467745 } ?e wdt:P31 ?t; wikibase:sitelinks ?sl.
        FILTER NOT EXISTS { ?e wdt:P576 ?d } }""",
}


def sparql(client: httpx.Client, query: str) -> list[dict]:
    r = client.get(SPARQL, params={"query": query}, headers={**UA, "Accept": "application/sparql-results+json"})
    r.raise_for_status()
    return r.json()["results"]["bindings"]


def search(client: httpx.Client, name: str) -> list[dict]:
    r = client.get(API, params={"action": "wbsearchentities", "search": name, "language": "en",
                                "type": "item", "limit": 5, "format": "json"}, headers=UA)
    r.raise_for_status()
    return r.json().get("search", [])


def fetch_details(client: httpx.Client, qids: list[str]) -> dict[str, dict]:
    out = {}
    for i in range(0, len(qids), 50):
        r = client.get(API, params={"action": "wbgetentities", "ids": "|".join(qids[i:i + 50]),
                                    "props": "labels|aliases|descriptions|sitelinks",
                                    "languages": "|".join(FETCH_LANGS), "format": "json"}, headers=UA)
        r.raise_for_status()
        out.update(r.json()["entities"])
        time.sleep(0.2)
    return out


def main() -> None:
    client = httpx.Client(timeout=120)
    types: dict[str, str] = {}
    for etype, query in QUERIES.items():
        rows = sparql(client, query)
        for b in rows:
            types.setdefault(b["e"]["value"].rsplit("/", 1)[1], etype)
        print(f"{etype}: {len(rows)}")

    extra = yaml.safe_load((CONFIG / "entities" / "extra.yaml").read_text(encoding="utf-8"))
    for etype, items in extra.items():
        for it in items:
            hits = search(client, it["name"])
            ok = lambda d: re.search(it["expect"], d, re.I) and not (it.get("reject") and re.search(it["reject"], d, re.I))
            match = next((h for h in hits if ok(h.get("description", ""))), None)
            if match is None:
                raise SystemExit(f"no Wikidata match for {it['name']!r} with {it['expect']!r} in its description: "
                                 + "; ".join(f"{h['id']} {h.get('description', '')}" for h in hits))
            types[match["id"]] = etype
            print(f"  {etype:15s} {it['name']:40s} → {match['id']}  ({match.get('description', '')})")

    details = fetch_details(client, sorted(types))
    rows = []
    for qid, etype in types.items():
        d = details.get(qid, {})
        if "missing" in d:
            continue
        labels = {l: d.get("labels", {}).get(l, {}).get("value") for l in FETCH_LANGS}
        labels["en"] = labels["en"] or labels["mul"]
        if not labels["en"]:
            continue
        aliases = {l: {a["value"] for a in d.get("aliases", {}).get(l, [])} for l in FETCH_LANGS}
        aliases["en"] |= aliases.pop("mul")
        rows.append({
            "qid": qid, "type": etype, "label_en": labels["en"], "label_hi": labels["hi"],
            "description": d.get("descriptions", {}).get("en", {}).get("value", ""),
            "sitelinks": len(d.get("sitelinks", {})),
            "aliases": {l: sorted(aliases[l]) for l in LANGS},
        })
    extra_aliases = yaml.safe_load((CONFIG / "entities" / "aliases_extra.yaml").read_text(encoding="utf-8")) or {}
    by_label = {r["label_en"]: r for r in rows}
    for label, langs in extra_aliases.items():
        if label not in by_label:
            raise SystemExit(f"aliases_extra.yaml: no entity labelled {label!r}")
        for lang, names in langs.items():
            by_label[label]["aliases"][lang] = sorted(set(by_label[label]["aliases"][lang]) | set(names))
    rows.sort(key=lambda r: (r["type"], -r["sitelinks"], r["label_en"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with_hi = sum(1 for r in rows if r["label_hi"])
    print(f"wrote {len(rows)} entities ({with_hi} with a Hindi label) → {OUT.relative_to(CONFIG.parent)}")


if __name__ == "__main__":
    main()
