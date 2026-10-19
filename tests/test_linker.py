import json

import pytest
import yaml

from nazariya.entities.linker import Linker

ENTITIES = [
    {"qid": "Q1", "type": "person", "label_en": "Narendra Modi", "label_hi": "नरेन्द्र मोदी", "sitelinks": 170,
     "aliases": {"en": [], "hi": []}},
    {"qid": "Q2", "type": "person", "label_en": "Sushil Kumar Modi", "label_hi": None, "sitelinks": 15,
     "aliases": {"en": [], "hi": []}},
    {"qid": "Q3", "type": "person", "label_en": "Rahul Gandhi", "label_hi": "राहुल गाँधी", "sitelinks": 64,
     "aliases": {"en": [], "hi": []}},
    {"qid": "Q4", "type": "person", "label_en": "Sonia Gandhi", "label_hi": None, "sitelinks": 60,
     "aliases": {"en": [], "hi": []}},
    {"qid": "Q5", "type": "party", "label_en": "Aam Aadmi Party", "label_hi": "आम आदमी पार्टी", "sitelinks": 40,
     "aliases": {"en": ["AAP"], "hi": ["आप"]}},
    {"qid": "Q6", "type": "party", "label_en": "Indian National Congress", "label_hi": None, "sitelinks": 90,
     "aliases": {"en": ["Congress", "INC"], "hi": []}},
    {"qid": "Q7", "type": "party", "label_en": "All India Trinamool Congress", "label_hi": None, "sitelinks": 30,
     "aliases": {"en": ["Trinamool Congress", "TMC"], "hi": []}},
    {"qid": "Q8", "type": "country", "label_en": "Pakistan", "label_hi": None, "sitelinks": 300,
     "aliases": {"en": ["pk", "PAK"], "hi": []}},
    {"qid": "Q9", "type": "person", "label_en": "Harsh Kumar", "label_hi": None, "sitelinks": 9,
     "aliases": {"en": [], "hi": []}},
]
RULES = {"blocked_aliases": ["आप"], "common_surnames": ["kumar"], "prominence_ratio": 3.0,
         "acronym_types": ["party", "institution", "organisation"]}


@pytest.fixture(scope="module")
def linker(tmp_path_factory):
    d = tmp_path_factory.mktemp("ents")
    (d / "e.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in ENTITIES), encoding="utf-8")
    (d / "r.yaml").write_text(yaml.safe_dump(RULES, allow_unicode=True), encoding="utf-8")
    return Linker(d / "e.jsonl", d / "r.yaml", calibration_path=None)


def qids(result):
    return {q: v["n"] for q, v in result.items()}


def test_full_names_and_hindi_variants(linker):
    assert qids(linker.link("Rahul Gandhi in Wayanad")) == {"Q3": 1}
    # anusvara (गांधी) in text vs chandrabindu (गाँधी) in Wikidata
    assert qids(linker.link("राहुल गांधी ने कहा")) == {"Q3": 1}


def test_surname_resolved_by_prominence(linker):
    assert qids(linker.link("PM Modi speaks")) == {"Q1": 1}  # 170 vs 15 sitelinks
    assert qids(linker.link("मोदी ने कहा")) == {"Q1": 1}


def test_ambiguous_surname_needs_context(linker):
    assert qids(linker.link("Gandhi says no")) == {}  # 64 vs 60: no clear winner
    assert qids(linker.link("Gandhi says no", "Sonia Gandhi said on Monday")) == {"Q4": 2}


def test_longest_match_wins(linker):
    assert qids(linker.link("Trinamool Congress hits back")) == {"Q7": 1}
    assert qids(linker.link("Congress hits back")) == {"Q6": 1}


def test_acronyms_case_sensitive_and_type_limited(linker):
    assert qids(linker.link("AAP and TMC protest")) == {"Q5": 1, "Q7": 1}
    assert qids(linker.link("aap kaise hain")) == {}
    assert qids(linker.link("PAK team arrives")) == {}  # country codes aren't used


def test_blocked_and_common_names(linker):
    assert qids(linker.link("आप कैसे हैं")) == {}
    assert qids(linker.link("Kumar said")) == {}  # common surname never counts alone
    assert qids(linker.link("आम आदमी पार्टी की बैठक")) == {"Q5": 1}


def test_title_flag_and_counts(linker):
    r = linker.link("Modi in Kolkata", "Narendra Modi met officials. Modi later left.")
    assert r["Q1"]["n"] == 3 and r["Q1"]["in_title"]


def test_half_nasal_equals_anusvara():
    from nazariya.entities.linker import normalise
    assert normalise("नरेन्द्र") == normalise("नरेंद्र")
    assert normalise("गाँधी") == normalise("गांधी")
