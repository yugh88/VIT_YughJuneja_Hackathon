"""Stage 1 - normalise: clean text and split it into sentences.

Pure functions, no models. Connectors already strip HTML; this stage makes text
from every source look alike (Unicode NFKC, no URLs or @handles, collapsed
whitespace, bounded length) and splits it into sentences in any script.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

MAX_CHARS = 2_000  # ~512 tokens; titles and first paragraphs only
_URL = re.compile(r"https?://\S+|www\.\S+")
_HANDLE = re.compile(r"(?<!\w)@\w{2,30}")
_WS = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]{1,200}>")
# Sentence ends: Latin/Cyrillic . ! ?, CJK 。！？, Devanagari danda ।, Arabic ؟ ۔
_SENT_END = re.compile(r"(?<=[.!?。！？।؟۔])\s+|(?<=[。！？])")
# Publisher suffixes Google News appends: "Headline - Reuters"
_PUBLISHER_SUFFIX = re.compile(r"\s+[-|–—]\s+[\w .&'’]{2,40}$")


@dataclass(frozen=True)
class CleanText:
    text: str
    sentences: list[str]


def clean(text: str, max_chars: int = MAX_CHARS) -> str:
    t = html.unescape(_TAG.sub(" ", text or ""))
    t = unicodedata.normalize("NFKC", t)
    t = _URL.sub(" ", t)
    t = _HANDLE.sub(" ", t)
    t = "".join(ch for ch in t if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    t = _WS.sub(" ", t).strip()
    return t[:max_chars]


def strip_publisher(title: str) -> str:
    """'Nvidia jumps on record sales - Reuters' -> 'Nvidia jumps on record sales'."""
    return _PUBLISHER_SUFFIX.sub("", title).strip() or title


def split_sentences(text: str, min_len: int = 3) -> list[str]:
    parts = [p.strip() for p in _SENT_END.split(text) if p and p.strip()]
    return [p for p in parts if len(p) >= min_len] or ([text] if text else [])


def normalise(title: str, text: str = "") -> CleanText:
    """Title + body as one cleaned string, plus its sentences (title is sentence 0)."""
    t = strip_publisher(clean(title)) if title else ""
    b = clean(text)
    if t and b.startswith(t):
        b = b[len(t) :].strip(" .-:")
    joined = f"{t}. {b}" if t and b else (t or b)
    sents = ([t] if t else []) + (split_sentences(b) if b else [])
    return CleanText(text=joined[:MAX_CHARS], sentences=sents[:12])
