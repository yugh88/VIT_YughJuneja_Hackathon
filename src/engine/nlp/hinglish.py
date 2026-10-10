"""Romanised Hindi ("Hinglish"): detect it and transliterate to Devanagari for sentiment (DECISIONS D73c).

fastText reads "Tata Motors ka munafa doguna hua" as Malay at low confidence. Two or more distinctly
Hindi words in Latin script mark the text as Hindi; sentiment then scores a Devanagari transliteration
(ITRANS, lower-cased) with capitalised words - company and place names - left in Latin script.
Entity linking still runs on the original text.
"""

from __future__ import annotations

import re

# Distinctly Hindi words (no common Spanish, French, Italian or Portuguese words such as "se" or "par").
_HI = {
    "hai", "hain", "ka", "ki", "ke", "ko", "mein", "kya", "hua", "hui", "hue", "nahi", "nahin", "aur", "kiya",
    "kiye", "gaya", "gayi", "badha", "badhi", "badhe", "ghata", "girawat", "tezi", "munafa", "kamai", "mila",
    "mili", "laga", "lagi", "rahi", "raha", "rahe", "hoga", "hogi", "diya", "naya", "nayi", "naye", "bada",
    "badi", "bhari", "doguna", "sath", "liye", "abhi", "btao", "batao", "khabar",
}  # fmt: skip
_LATIN_WORD = re.compile(r"[A-Za-z]+")


def is_hinglish(text: str) -> bool:
    if re.search(r"[ऀ-ॿ]", text):
        return False
    words = {w.lower() for w in _LATIN_WORD.findall(text)}
    return len(words & _HI) >= 2


def to_devanagari(text: str) -> str:
    from indic_transliteration import sanscript

    return " ".join(
        w if w[:1].isupper() else sanscript.transliterate(w.lower(), sanscript.ITRANS, sanscript.DEVANAGARI)
        for w in text.split()
    )
