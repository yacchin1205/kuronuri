import re
from dataclasses import dataclass
from functools import cache

import spacy

from .extract import PageText

_WIDE_TO_ASCII = str.maketrans(
    "０１２３４５６７８９－‐−ー＠．",
    "0123456789----@.",
)

PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "phone": re.compile(r"(?<!\d)\(?0\d{1,4}\)?-?\d{1,4}-\d{3,4}(?!\d)"),
    "postal": re.compile(r"(?:〒\s?)?(?<![\d-])\d{3}-\d{4}(?![\d-])"),
    "mynumber": re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)"),
}

# GiNZA labels (Sekine's extended NE hierarchy) worth proposing for review.
NER_LABELS = {
    "Person",
    "Company",
    "Corporation_Other",
    "Organization_Other",
    "School",
    "Postal_Address",
    "City",
    "Province",
    "Phone_Number",
    "Email",
    "Disease",
}


@dataclass
class Candidate:
    page: int
    start: int
    end: int
    text: str
    kind: str
    source: str
    context: str  # excerpt with the candidate marked as 【candidate】


@cache
def _nlp():
    return spacy.load("ja_ginza")


def _candidate(page: PageText, start: int, end: int, kind: str, source: str,
               context_width: int) -> Candidate | None:
    """Trim surrounding whitespace; whitespace-only spans are not candidates."""
    raw = page.text[start:end]
    start += len(raw) - len(raw.lstrip())
    end -= len(raw) - len(raw.rstrip())
    if end <= start:
        return None
    text = page.text[start:end]
    context = (page.text[max(0, start - context_width) : start] + "【" + text + "】"
               + page.text[end : end + context_width]).replace("\n", " ")
    return Candidate(page.number, start, end, text, kind, source, context)


def detect_regex(page: PageText, context_width: int) -> list[Candidate]:
    normalized = page.text.translate(_WIDE_TO_ASCII)
    return [
        c for kind, pattern in PATTERNS.items() for m in pattern.finditer(normalized)
        if (c := _candidate(page, *m.span(), kind, "regex", context_width))
    ]


def detect_ner(page: PageText, context_width: int) -> list[Candidate]:
    return [
        c for ent in _nlp()(page.text).ents
        if ent.label_ in NER_LABELS
        and (c := _candidate(page, ent.start_char, ent.end_char, ent.label_, "ner", context_width))
    ]


def detect(page: PageText, context_width: int = 60) -> list[Candidate]:
    found = detect_regex(page, context_width) + detect_ner(page, context_width)
    unique = {(c.start, c.end): c for c in sorted(found, key=lambda c: c.source == "ner")}
    return sorted(unique.values(), key=lambda c: c.start)
