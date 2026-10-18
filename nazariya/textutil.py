"""Pure text/URL helpers: canonical URLs, language, sections, SimHash, agency bylines, dates."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import datetime, timedelta
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
SCRIPTS = {
    "hi": (0x0900, 0x097F),  # Devanagari (also Marathi; refine with a LID model if added)
    "bn": (0x0980, 0x09FF),
    "pa": (0x0A00, 0x0A7F),
    "gu": (0x0A80, 0x0AFF),
    "ta": (0x0B80, 0x0BFF),
    "te": (0x0C00, 0x0C7F),
    "kn": (0x0C80, 0x0CFF),
    "ml": (0x0D00, 0x0D7F),
}

TRACKING_PARAMS = {"fbclid", "gclid", "ref", "ref_src", "cmp", "from", "src", "sref", "ito", "pfrom", "amp", "utm"}


def rewrite_host(url: str, rewrites: dict[str, str]) -> str:
    """Fix links whose host is wrong in the outlet's own feed (e.g. a dead mirror domain)."""
    p = urlsplit(url)
    return p._replace(netloc=rewrites[p.netloc]).geturl() if p.netloc in rewrites else url


def canonical_url(url: str, rewrites: dict[str, str] | None = None) -> str:
    """Normalise so the same article reached via AMP / tracking links maps to one key."""
    p = urlsplit(rewrite_host(url.strip(), rewrites or {}))
    host = p.netloc.lower().removeprefix("amp.")
    path = p.path or "/"
    path = path.replace("/amp_articleshow/", "/articleshow/")  # Times Internet AMP
    path = re.sub(r"^/amp(?=/)", "", path)
    path = re.sub(r"/amp/?$", "", path)
    path = re.sub(r"\.amp(?=$|\.)", "", path)
    if len(path) > 1:
        path = path.rstrip("/")
    query = [
        (k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS
    ]
    return urlunsplit(("https", host, path, urlencode(sorted(query)), ""))


def in_scope(url: str, homepage: str, extra_hosts: list[str] | tuple[str, ...] = ()) -> bool:
    """Does url belong to this outlet? Same host (ignoring www.) and, when the homepage has a
    path like bbc.com/hindi, under that path. Shared sitemaps (BBC, Zee) list every edition."""
    u, h = urlsplit(url), urlsplit(homepage)
    host = u.netloc.lower().removeprefix("www.")
    hosts = {h.netloc.lower().removeprefix("www.")} | {x.removeprefix("www.") for x in extra_hosts}
    if host not in hosts:
        return False
    prefix = h.path.rstrip("/")
    return not prefix or u.path == prefix or u.path.startswith(prefix + "/")


def path_segments(url: str) -> list[str]:
    return [s.lower() for s in urlsplit(url).path.split("/") if s]


def section_of(url: str) -> str | None:
    """First non-numeric path segment, e.g. /news/india/... → 'news'."""
    for seg in path_segments(url):
        if not seg.isdigit() and "." not in seg:
            return seg[:80]
    return None


def detect_lang(text: str, default: str = "en") -> str:
    """Script-based language guess; good enough to separate en/hi/regional scripts."""
    counts: Counter[str] = Counter()
    latin = 0
    for ch in text:
        o = ord(ch)
        if ch.isascii():
            latin += ch.isalpha()
            continue
        for lang, (lo, hi) in SCRIPTS.items():
            if lo <= o <= hi:
                counts[lang] += 1
                break
    if not counts and not latin:
        return default
    lang, n = counts.most_common(1)[0] if counts else (default, 0)
    return lang if n >= latin * 0.5 else "en"


# Punctuation incl. Devanagari danda; splitting on \W would break Hindi words at vowel signs.
TOKEN_SPLIT = re.compile(r"[\s.,;:!?\"'()\[\]{}<>|/\\\-–—“”‘’…।॥]+")
MASK64 = (1 << 64) - 1


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN_SPLIT.split(text.lower()) if t]


def simhash(text: str, ngram: int = 3) -> int:
    """64-bit SimHash over word n-gram shingles (Charikar). Near-duplicates differ in few bits."""
    toks = tokens(text)
    shingles = [" ".join(toks[i:i + ngram]) for i in range(max(1, len(toks) - ngram + 1))]
    weights = [0] * 64
    for sh in shingles:
        h = int.from_bytes(hashlib.blake2b(sh.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if (h >> bit) & 1 else -1
    return sum(1 << bit for bit in range(64) if weights[bit] > 0)


def hamming(a: int, b: int) -> int:
    return ((a ^ b) & MASK64).bit_count()


def to_signed64(x: int) -> int:
    """Postgres BIGINT is signed; store the unsigned hash in its two's-complement form."""
    return x - (1 << 64) if x >= 1 << 63 else x


def to_unsigned64(x: int) -> int:
    return x & MASK64


AGENCY_NAMES = {
    "pti": "PTI", "ani": "ANI", "ians": "IANS", "uni": "UNI", "reuters": "Reuters", "afp": "AFP",
    "भाषा": "PTI", "पीटीआई": "PTI", "एएनआई": "ANI", "आईएएनएस": "IANS", "वार्ता": "UNI",
}
_AG = "|".join(sorted(map(re.escape, AGENCY_NAMES), key=len, reverse=True))
AGENCY_PATTERNS = [
    # "New Delhi, Oct 17 (PTI)" / "नई दिल्ली, 17 अक्टूबर (भाषा)"
    re.compile(rf"^[^\n()]{{0,60}}\(\s*({_AG})\s*\)", re.IGNORECASE),
    # "(With inputs from PTI)" / "(With PTI inputs)" / "-- PTI" at the end
    re.compile(rf"\(\s*with\s+(?:inputs\s+from\s+)?(?:agency\s+)?({_AG})(?:\s+inputs)?\s*\)", re.IGNORECASE),
    re.compile(rf"(?:^|\n)\s*[-–—]+\s*({_AG})\s*$", re.IGNORECASE),
    re.compile(rf"\(\s*({_AG})\s*\)\s*$", re.IGNORECASE),
]
SYNDICATED_NOTE = re.compile(r"(auto-generated from a syndicated feed|published from a wire agency feed)", re.IGNORECASE)


def detect_agency(text: str) -> str | None:
    """Wire-agency byline in the opening or closing lines, e.g. PTI/ANI/IANS (incl. Hindi forms)."""
    head, tail = text[:300], text[-300:]
    for pat in AGENCY_PATTERNS:
        m = pat.search(head) or pat.search(tail)
        if m:
            return AGENCY_NAMES[m.group(1).lower()]
    if SYNDICATED_NOTE.search(tail) or SYNDICATED_NOTE.search(head):
        return "wire"
    return None


IST_OFFSET = timedelta(hours=5, minutes=30)
FUTURE_TOLERANCE = timedelta(minutes=15)


def fix_published(dt: datetime | None, now: datetime) -> tuple[datetime | None, bool]:
    """Repair timestamps in the future: usually IST wall time mislabelled as UTC."""
    if dt is None or dt <= now + FUTURE_TOLERANCE:
        return dt, False
    shifted = dt - IST_OFFSET
    return (shifted if shifted <= now + FUTURE_TOLERANCE else now), True
