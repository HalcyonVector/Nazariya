"""Dictionary entity linker over the Wikidata alias table (English + Hindi), CPU-fast and explainable.

Matching: text is tokenised (Devanagari-safe), Latin case-folded except for short all-caps
acronyms, Hindi normalised (chandrabindu → anusvara, nukta dropped), then scanned left to right
taking the longest alias that matches at each position.

Each alias maps to one or more entities. Resolution per article:
  1. aliases naming exactly one entity link directly (full names, unique aliases);
  2. ambiguous names (shared aliases, surnames like "Modi"/"मोदी") link to a candidate already
     named in full in the same article; otherwise to the most prominent candidate if it has at
     least `prominence_ratio` × the Wikipedia sitelinks of the next; otherwise they are dropped.
Very common surnames never count on their own; blocked aliases never count.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import yaml

from ..config import CONFIG

ENTITIES = CONFIG / "entities" / "entities.jsonl"
RULES = CONFIG / "entities" / "linking.yaml"
# single-word person names dropped because they are mostly ordinary words (nazariya.entities.calibrate)
CALIBRATION = CONFIG / "entities" / "alias_calibration.yaml"
SPLIT = re.compile(r"[\s.,;:!?\"()\[\]{}<>|/\\\-–—“”‘’'`…।॥·•]+")
MAX_ALIAS_TOKENS = 8
# Bump when matching logic changes, so stored mentions are rescanned (the version also hashes the
# alias table, rules and calibration files).
LOGIC_VERSION = b"2"
PAREN = re.compile(r"\s*\([^)]*\)")


# A nasal consonant + virama before another consonant is written interchangeably with anusvara:
# नरेन्द्र (Wikidata) and नरेंद्र (most news sites) are the same name.
HALF_NASAL = re.compile("[ङञणनम]्(?=[क-ह])")


def normalise(token: str) -> str:
    token = token.replace("ँ", "ं").replace("़", "")
    return HALF_NASAL.sub("ं", token).lower()


def tokenize(text: str) -> list[str]:
    return [t for t in SPLIT.split(text) if t]


def is_acronym(alias: str) -> bool:
    letters = alias.replace(".", "").replace(" ", "")
    return 2 <= len(letters) <= 4 and letters.isascii() and letters.isalpha() and letters.isupper()


@dataclass(frozen=True)
class Entry:
    qid: str
    derived: bool  # surname derived from the label, not an explicit Wikidata name


class Linker:
    def __init__(self, entities_path=ENTITIES, rules_path=RULES, calibration_path=CALIBRATION):
        rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        blocked = {" ".join(normalise(t) for t in tokenize(a)) for a in rules.get("blocked_aliases", [])}
        common = {normalise(s) for s in rules.get("common_surnames", [])}
        acronym_types = set(rules.get("acronym_types", []))
        self.ratio = float(rules.get("prominence_ratio", 3.0))
        raw = entities_path.read_bytes()
        calib = calibration_path.read_bytes() if calibration_path and calibration_path.exists() else b""
        self.version = hashlib.sha1(LOGIC_VERSION + raw + rules_path.read_bytes() + calib).hexdigest()[:10]
        cal = (yaml.safe_load(calib) or {}) if calib else {}
        dropped = {(d["alias"], d["qid"]) for d in cal.get("dropped", [])}
        # With a calibration, a single-word person name needs positive evidence (seen often enough
        # and usually alongside the full name); rare single words such as given names are not trusted.
        confirmed = {(d["alias"], d["qid"]) for d in cal.get("kept", [])} if calib else None
        self.entities = {e["qid"]: e for e in map(json.loads, raw.decode("utf-8").splitlines()) if e}
        for e in self.entities.values():  # "कार्तिक (अभिनेता)" → "कार्तिक": drop Wikipedia-style notes
            for k in ("label_en", "label_hi"):
                if e.get(k):
                    e[k] = PAREN.sub("", e[k]).strip() or e[k]
        persons = {q for q, e in self.entities.items() if e["type"] in ("person", "foreign_person")}

        def single_ok(word: str, qid: str) -> bool:
            if (word, qid) in dropped:
                return False
            return confirmed is None or qid not in persons or (word, qid) in confirmed

        # folded alias tokens → entries; case-sensitive acronyms kept separately
        self.index: dict[tuple[str, ...], set[Entry]] = defaultdict(set)
        self.acronyms: dict[str, set[str]] = defaultdict(set)
        for qid, e in self.entities.items():
            names = {e["label_en"], *(e["aliases"]["en"])}
            if e.get("label_hi"):
                names.add(e["label_hi"])
            names |= set(e["aliases"]["hi"])
            for name in names:
                toks = tokenize(PAREN.sub("", name))
                if not toks or len(toks) > MAX_ALIAS_TOKENS:
                    continue
                if is_acronym(name):
                    if e["type"] in acronym_types:
                        self.acronyms[name.replace(".", "").replace(" ", "")].add(qid)
                    continue
                key = tuple(normalise(t) for t in toks)
                if " ".join(key) in blocked or len("".join(key)) <= 3:
                    continue
                if len(key) == 1 and not single_ok(key[0], qid):
                    continue
                self.index[key].add(Entry(qid, False))
            if e["type"] in ("person", "foreign_person"):
                for label in (e["label_en"], e.get("label_hi")):
                    toks = tokenize(label or "")
                    if len(toks) >= 2:
                        sur = normalise(toks[-1])
                        if len(sur) >= 4 and sur not in common and sur not in blocked and single_ok(sur, qid):
                            self.index[(sur,)].add(Entry(qid, True))

    def _prominent(self, qids: set[str]) -> str | None:
        ranked = sorted(qids, key=lambda q: -self.entities[q]["sitelinks"])
        if len(ranked) == 1:
            return ranked[0]
        top, nxt = self.entities[ranked[0]]["sitelinks"], self.entities[ranked[1]]["sitelinks"]
        return ranked[0] if top >= self.ratio * max(nxt, 1) else None

    def link(self, title: str, body: str = "") -> dict[str, dict]:
        """{qid: {"n": mentions, "in_title": bool}} for one article."""
        hits: list[tuple[set[Entry] | set[str], bool, bool]] = []  # (candidates, in_title, is_acronym)
        for part, in_title in ((title or "", True), (body or "", False)):
            toks = tokenize(part)
            folded = [normalise(t) for t in toks]
            i = 0
            while i < len(toks):
                if toks[i] in self.acronyms:
                    hits.append((self.acronyms[toks[i]], in_title, True))
                    i += 1
                    continue
                for n in range(min(MAX_ALIAS_TOKENS, len(toks) - i), 0, -1):
                    entries = self.index.get(tuple(folded[i:i + n]))
                    if entries:
                        hits.append((entries, in_title, False))
                        i += n
                        break
                else:
                    i += 1

        def candidates(h) -> set[str]:
            c, _, acro = h
            return set(c) if acro else {e.qid for e in c}

        def direct(h) -> bool:
            c, _, acro = h
            return len(candidates(h)) == 1 and (acro or not any(e.derived for e in c))

        named = {next(iter(candidates(h))) for h in hits if direct(h)}
        out: dict[str, dict] = {}
        for h in hits:
            cands = candidates(h)
            if direct(h):
                qid = next(iter(cands))
            else:
                in_article = cands & named
                qid = next(iter(in_article)) if len(in_article) == 1 else (None if in_article else self._prominent(cands))
            if qid is None:
                continue
            rec = out.setdefault(qid, {"n": 0, "in_title": False})
            rec["n"] += 1
            rec["in_title"] |= h[1]
        return out

    def label(self, qid: str, lang: str = "en") -> str:
        e = self.entities[qid]
        return (e.get("label_hi") if lang == "hi" else None) or e["label_en"]
