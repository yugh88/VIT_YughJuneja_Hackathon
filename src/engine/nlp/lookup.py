"""Company lookup for the "Try it" box: is the text a headline to score, or a question about a
company or country ("tata", "latest on Reliance", "tata ki recent news btao", "infosys ka kya hua")?

Rules only, no model and no API: a small intent word list (English, Hindi, Hinglish), filler words
removed, then the remainder matched against every gazetteer name (all languages), English short
forms and tickers, case-insensitively and with a little spelling tolerance for romanised names.
A bare family name that starts several entities ("tata", "adani", "bajaj") is reported as
ambiguous so the UI can ask "Did you mean ...".
"""

from __future__ import annotations

import difflib
import gzip
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from engine.config import get_settings

# Question intent (Latin script, incl. romanised Hindi) and Devanagari equivalents.
INTENT_WORDS = {
    "news", "latest", "update", "updates", "recent", "happened", "happening", "whats", "status", "headlines",
    "batao", "btao", "bataiye", "bataye", "bata", "batana", "khabar", "khabren", "khabrein", "samachar", "taaza", "taza",
    "खबर", "ख़बर", "खबरें", "ताज़ा", "ताजा", "बताओ", "बताइए", "बताएं", "समाचार", "अपडेट",
}  # fmt: skip
INTENT_PHRASES = re.compile(
    r"\b(what happened|what is happening|what'?s up|kya hua|kya chal raha|how is|any news)\b|क्या हुआ|"
    r"ニュース|最新|どうなった|新闻|新聞|消息|动态",
    re.I,
)
# Words that carry no company meaning in a lookup question.
FILLER = {
    "the", "a", "an", "on", "of", "about", "for", "in", "me", "tell", "show", "give", "any", "what", "is", "are", "was",
    "with", "today", "now", "please", "pls", "plz", "and", "to", "from", "share", "shares", "stock", "stocks", "price",
    "company", "ki", "ka", "ke", "ko", "kya", "hua", "hai", "hain", "mein", "se", "par", "aaj", "abhi", "koi",
    "kuch", "wala", "wali", "chal", "raha", "rahi", "jankari", "की", "का", "के", "को", "में", "है", "हैं", "से", "पर", "आज", "कोई",
    "कुछ", "क्या", "हुआ", "बारे",
}  # fmt: skip
MAX_LOOKUP_WORDS = 3  # without intent words, longer text is treated as a headline
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
MIN_COVERAGE = 0.6  # ... and the matched name must cover most of what is left


def _norm(s: str) -> str:
    """Lower-case; keep letters, combining marks (Devanagari matras), digits, '&' and apostrophes."""
    s = unicodedata.normalize("NFKC", s).lower().replace("’", "'")
    s = "".join(c if unicodedata.category(c)[0] in "LMN" or c in "&'" else " " for c in s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass(frozen=True)
class Alias:
    entity: str
    text: str  # normalised
    words: tuple[str, ...]
    kind: str  # name | alias | ticker


@lru_cache(maxsize=1)
def _index() -> tuple[list[Alias], dict[str, dict[str, Any]]]:
    with gzip.open(get_settings().reference_dir / "gazetteer.json.gz") as fh:
        data = json.loads(fh.read())
    aliases: list[Alias] = []
    info: dict[str, dict[str, Any]] = {}
    for e in data["entities"]:
        info[e["id"]] = {k: e.get(k) for k in ("id", "name", "kind", "ticker", "country", "sector")}
        seen: set[str] = set()

        def add(text: str, kind: str, ent: str = e["id"], seen: set[str] = seen) -> None:
            n = _norm(text)
            if len(n) < 2 or n in seen:
                return
            seen.add(n)
            aliases.append(Alias(ent, n, tuple(n.split()), kind))

        add(e["name"], "name")
        if e.get("ticker"):
            add(str(e["ticker"]).split(".")[0].replace("-", " "), "ticker")
        for lang, names in (e.get("names") or {}).items():
            for a in names:
                # short all-caps English forms (TCS, RIL) are kept; other 2-3 letter aliases are too noisy
                if len(_norm(a)) <= 3 and not (lang == "en" and a.isupper()):
                    continue
                add(a, "alias")
    return aliases, info


def _intent(text: str, words: list[str]) -> bool:
    return bool(INTENT_PHRASES.search(text)) or any(w in INTENT_WORDS for w in words)


def lookup(text: str, signal_counts: dict[str, int] | None = None) -> dict[str, Any]:
    """Classify the text and, in company mode, return the matching entities."""
    words = _norm(text).split()
    intent = _intent(text, words)
    rest = [w for w in words if w not in INTENT_WORDS and w not in FILLER]
    q = " ".join(rest)
    aliases, info = _index()
    best: dict[str, tuple[float, str, float]] = {}  # entity -> (score, how, coverage of the query)

    def offer(ent: str, score: float, how: str, cover: float) -> None:
        if score > best.get(ent, (0.0, "", 0.0))[0]:
            best[ent] = (score, how, cover)

    if q:
        for a in aliases:
            if a.text == q:
                offer(a.entity, 3.0, "exact" if a.kind != "alias" else "exact alias", 1.0)
            elif len(q) >= 3 and a.text.startswith(q + " "):
                offer(a.entity, 2.0, "starts with", 1.0)  # "tata" -> Tata Motors, Tata Steel, ...
            elif len(a.text) >= 3 and (
                (_CJK.search(a.text) and a.text in q)  # no word boundaries in CJK script
                or re.search(rf"(?<!\w){re.escape(a.text)}(?!\w)", q)
            ):
                offer(a.entity, 2.5, "contains", len(a.text) / len(q))  # "infosys results" -> Infosys
        if not best and len(q) >= 5:  # romanised / misspelt names: "infosis", "relaince"
            texts = {a.text: a for a in aliases if a.kind != "ticker" or len(a.text) >= 5}
            for m in difflib.get_close_matches(q, list(texts), n=6, cutoff=0.75):
                offer(texts[m].entity, 1.5, "similar spelling", 1.0)
        # A one-word family name that is only an alias of one firm but starts others ("bajaj") is ambiguous.
        if len(rest) == 1 and any(h == "exact alias" for _, h, _ in best.values()):
            for k, (_sc, h, cv) in list(best.items()):
                if h == "starts with":
                    best[k] = (3.0, h, cv)
    counts = signal_counts or {}
    ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], -counts.get(kv[0], 0), info[kv[0]]["name"]))
    top = ranked[0][1][0] if ranked else 0.0
    tied = [k for k, (sc, _, _) in ranked if sc == top]
    cover = max((cv for k, (_, _, cv) in ranked if k in tied), default=0.0)
    company = (
        bool(ranked) and top >= 1.5 and (intent or (len(rest) <= MAX_LOOKUP_WORDS and cover >= MIN_COVERAGE))
    )
    return {
        "mode": "company" if company else "headline",
        "intent": intent,
        "query": q,
        "resolved": tied[0] if company and len(tied) == 1 else None,
        "ambiguous": company and len(tied) > 1,
        "matches": [
            {**info[k], "match": how, "score": sc, "signals": counts.get(k, 0)}
            for k, (sc, how, _) in ranked[:8]
        ],
    }


@lru_cache(maxsize=1)
def ticker_names() -> dict[str, str]:
    """Upper-case ticker root (no exchange suffix) -> entity name."""
    _, info = _index()
    out: dict[str, str] = {}
    for v in info.values():
        if v.get("ticker"):
            out.setdefault(str(v["ticker"]).split(".")[0].upper(), str(v["name"]))
    return out
