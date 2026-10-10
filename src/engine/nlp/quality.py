"""News-quality rules applied before and after the models (DECISIONS D71).

* Promotional social posts are dropped: more than 4 cashtags, or cashtags + hashtags over 40% of
  the words, or a promo phrase ("get alerts", "click link", "join our", ...).
* Short texts are not language-guessed: under 5 real words once cashtags, hashtags, mentions,
  numbers and URLs are removed, the source's default language is used instead.
* Price-move headlines ("X soars / plunges ...") take the class of the stated cause, else
  Earnings when earnings words appear, else stay Other with the Impact Score capped at 5.
"""

from __future__ import annotations

import re

MAX_CASHTAGS = 4
MAX_TAG_SHARE = 0.40
MIN_REAL_WORDS = 5
SOCIAL_IMPACT_CAP = 5

PROMO = re.compile(
    r"get (instant |real[- ]time |free )?alerts|click (the |this |our )?link\b(?!s)|click here|join (our|us)\b|"
    r"chat with fellow members|free trading (analysis|signals|alerts)|sign up (now|today|free)|"
    r"subscribe (now|today)|dm (me|us) for|link in (bio|profile)|use (my|our) (code|link)|"
    r"(discord|telegram|whatsapp) (group|channel|chat)",
    re.I,
)
_CASHTAG = re.compile(r"(?<![\w$])\$[A-Za-z][A-Za-z.]{0,9}\b")
_HASHTAG = re.compile(r"(?<!\w)#\w+")
_NOISE = re.compile(
    r"https?://\S+|www\.\S+|(?<![\w$])\$[A-Za-z][A-Za-z.]{0,9}\b|(?<!\w)[#@]\w+|\b\d[\d.,:%/-]*\b"
)

# Default language per source when a text is too short to identify (replayed datasets are English).
SOURCE_DEFAULT_LANG = {
    "tweets": "en",
    "reddit": "en",
    "hackernews": "en",
    "edgar_8k": "en",
    "central_banks": "en",
}


def tag_counts(text: str) -> tuple[int, int, int]:
    """(cashtags, hashtags, words) of a post."""
    words = text.split()
    return len(_CASHTAG.findall(text)), len(_HASHTAG.findall(text)), len(words)


def is_promotional(text: str) -> str | None:
    """Reason a social post is promotional spam, or None."""
    cash, tags, words = tag_counts(text)
    if cash > MAX_CASHTAGS:
        return f"{cash} cashtags"
    if words and (cash + tags) / words > MAX_TAG_SHARE:
        return f"tags are {round(100 * (cash + tags) / words)}% of words"
    m = PROMO.search(text)
    return f"promo phrase “{m.group(0)}”" if m else None


_NO_SPACE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\u0e00-\u0e7f]")


def real_word_count(text: str) -> int:
    """Words left after removing URLs, cashtags, hashtags, mentions and numbers (scripts written
    without spaces count about one word per two characters)."""
    rest = _NOISE.sub(" ", text)
    cjk = len(_NO_SPACE.findall(rest))
    spaced = [w for w in _NO_SPACE.sub(" ", rest).split() if any(c.isalpha() for c in w)]
    return len(spaced) + cjk // 2


# ------------------------------------------------------------------ price-move headlines
MOVE = re.compile(
    r"\b(sky-?rocket\w*|soar\w*|surg(e|es|ed|ing)|jump(s|ed|ing)?|rall(y|ies|ied|ying)|spik(e|es|ed|ing)|rocket(s|ed|ing)?|"
    r"plung\w*|tumbl\w*|slump\w*|sink(s|ing)?|sank|crash(es|ed|ing)?|plummet\w*|nosedive\w*|tank(s|ed|ing)?|"
    r"sinks?|skids?|slides?|slid|dives?|dived|dove)\b",
    re.I,
)
CAUSE = re.compile(
    r"\b(after|following|amid|due to|because of|thanks to|on (news|reports?|hopes?|fears?|concerns?) (of|that|about)|"
    r"on|as|over)\s+(?P<cause>.{6,})$",
    re.I,
)
EARNINGS = re.compile(
    r"\b(earnings|results|quarter(ly)?|q[1-4]|profit|loss(es)?|revenue|sales|guidance|forecast|outlook|eps|"
    r"beats?|miss(es|ed)?|dividend)\b",
    re.I,
)


_NOT_STOCK = re.compile(
    r"\b(profit|profits|revenue|revenues|sales|earnings|income|margin|margins|inflation|prices?|yields?|output|demand|"
    r"exports?|imports?|orders|deficit|debt|costs?|rates?|production|unemployment|gdp|traffic|deliveries)\b",
    re.I,
)


def price_move(text: str, subjects: list[tuple[int, int]] | None = None) -> tuple[bool, str | None, bool]:
    """(the stock itself moved, stated cause clause or None, has earnings words).

    The move verb must follow the stock's subject (a linked company mention, a cashtag, or
    "shares"/"stock") within two words, with no other subject such as "profit" or "inflation"."""
    starts = [m.end() for m in re.finditer(r"\b(shares?|stocks?|ADRs?)\b|(?<![\w$])\$[A-Za-z]{1,6}\b", text)]
    starts += [b for _, b in subjects or []]
    for m in MOVE.finditer(text):
        for end in starts:
            gap = text[end : m.start()]
            if end <= m.start() and len(gap.split()) <= 2 and not _NOT_STOCK.search(gap) and "," not in gap:
                c = CAUSE.search(text[m.end() :])
                cause = c.group("cause").strip() if c else None
                return True, cause, bool(EARNINGS.search(text))
    return False, None, bool(EARNINGS.search(text))
