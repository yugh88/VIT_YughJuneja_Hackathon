"""Stage 4 - entity linking.

Aho-Corasick over the multilingual Wikidata alias gazetteer (one pass over the
text for ~13k names in 20+ languages), plus cashtags (``$AAPL``) and bare tickers.
Rules that keep precision high:

* longest match wins, so "Bank of China" is not also "China";
* aliases written in scripts that use spaces must sit on word boundaries;
* short all-caps aliases (IBM, UBS, US) match case-sensitively;
* ambiguous names ("Apple", "Shell", "Target") need a finance context word in the
  same sentence;
* when nothing matches, GDELT's geocoded location or a feed's ticker tag can
  supply a fallback entity.
"""

from __future__ import annotations

import gzip
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import ahocorasick
import yaml

from engine.config import get_settings

COMMON_WORDS = {
    "apple", "amazon", "target", "shell", "visa", "oracle", "meta", "ford", "titan", "trent", "eternal",
    "booking", "intuit", "progressive", "discover", "southern", "general", "united", "american", "chinese",
    "union", "pacific", "republic", "signature", "deere", "linde", "nike", "intel", "uber", "reliance",
}  # fmt: skip
ALIAS_BLOCKLIST = {
    "水果店", "staaten", "merica", "murica", "bital", "the company", "company", "group", "bank",
    "holding", "holdings", "corporation", "inc.", "the bank", "america",
    # acronyms that mostly mean something else in headlines (audited on 30k GDELT titles)
    "abc", "usb", "pmi", "crm", "cat", "sa", "ri", "boc", "td", "cm", "kb", "fab", "snb",
}  # fmt: skip
TWO_LETTER_OK = {"US", "UK", "EU"}
FINANCE_CONTEXT = re.compile(
    r"\b(shares?|shareholders?|annual meeting|agm|board|stocks?|earnings|revenue|profit|sales|ceo|cfo|investors?|market|nasdaq|nyse|quarter|q[1-4]|"
    r"inc\.?|corp\.?|company|iphone|ipad|mac|aws|prime|guidance|dividend|valuation|analysts?|downgrade|"
    r"upgrade|results|layoffs?|acquisition|deal|antitrust|lawsuit|plc|refinery|oil|gas|retailer|stores?|"
    r"payments?|card|cloud|database|software|aktie|actions?|acciones|ações|bourse|börse)\b|\$[A-Z]",
    re.I,
)
BARE_TICKER_STOP = {"CAT", "LOW", "NOW", "COST", "DIS", "MET", "ALL", "IT", "ON", "SO", "KO", "GE", "GM", "MA",
                    "MS", "HD", "PM", "MO", "DE", "GD", "LT", "BEL", "ITC", "TCS", "CEO", "USA", "IPO", "ETF", "USB", "CRM", "PEP", "CVS", "AMT"}  # fmt: skip
_CASHTAG = re.compile(r"\$([A-Za-z]{1,10}(?:[.\-][A-Za-z]{1,2})?)\b")
_BARE = re.compile(r"(?<![$\w])([A-Z]{3,5})\b")
NO_BOUNDARY_SCRIPTS = ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "THAI")

# FIPS 10-4 (GDELT) -> ISO 3166-1 alpha-2 for the sovereigns in the gazetteer.
FIPS_TO_ISO = {
    "US": "US", "CA": "CA", "MX": "MX", "BR": "BR", "AR": "AR", "CI": "CL", "CO": "CO", "PE": "PE", "VE": "VE",
    "UK": "GB", "GM": "DE", "FR": "FR", "IT": "IT", "SP": "ES", "NL": "NL", "SZ": "CH", "SW": "SE", "NO": "NO",
    "PL": "PL", "GR": "GR", "PO": "PT", "EI": "IE", "BE": "BE", "AU": "AT", "FI": "FI", "HU": "HU", "RO": "RO",
    "UP": "UA", "RS": "RU", "BO": "BY", "TU": "TR", "IS": "IL", "IR": "IR", "IZ": "IQ", "SA": "SA", "AE": "AE",
    "QA": "QA", "EG": "EG", "LE": "LB", "SY": "SY", "YM": "YE", "WE": "PS", "GZ": "PS", "CH": "CN", "TW": "TW",
    "HK": "HK", "JA": "JP", "KS": "KR", "KN": "KP", "IN": "IN", "PK": "PK", "BG": "BD", "CE": "LK", "ID": "ID",
    "MY": "MY", "SN": "SG", "TH": "TH", "VM": "VN", "RP": "PH", "AS": "AU", "NZ": "NZ", "SF": "ZA", "NI": "NG",
    "KE": "KE", "ET": "ET", "GH": "GH", "SU": "SD",
}  # fmt: skip


@dataclass(frozen=True)
class EntityInfo:
    id: str
    kind: str  # equity | bank | corporate | sovereign
    name: str
    ticker: str | None
    country: str
    sector: str
    region: str | None


@dataclass
class Mention:
    entity: EntityInfo
    alias: str
    start: int
    end: int
    method: str  # alias | cashtag | ticker | tag | geo


@dataclass
class LinkResult:
    mentions: list[Mention] = field(default_factory=list)

    def entities(self) -> list[EntityInfo]:
        seen: dict[str, EntityInfo] = {}
        for m in self.mentions:
            seen.setdefault(m.entity.id, m.entity)
        return list(seen.values())


def _lower_same_len(text: str) -> str:
    """casefold without changing string length, so match offsets map back to the original."""
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text)


def _needs_boundary(alias: str) -> bool:
    ch = alias[0]
    name = unicodedata.name(ch, "")
    return not name.startswith(NO_BOUNDARY_SCRIPTS)


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


class EntityLinker:
    def __init__(self, gazetteer_path: Path | None = None) -> None:
        path = gazetteer_path or get_settings().reference_dir / "gazetteer.json.gz"
        doc = json.loads(
            gzip.decompress(path.read_bytes()).decode("utf-8")
            if path.suffix == ".gz"
            else path.read_text("utf-8")
        )
        self.entities: dict[str, EntityInfo] = {}
        self.by_ticker: dict[str, EntityInfo] = {}
        self.by_country: dict[str, EntityInfo] = {}
        self._auto = ahocorasick.Automaton()
        # key (lowercased alias) -> list of (entity_id, original alias, case_sensitive, ambiguous)
        table: dict[str, list[tuple[str, str, bool, bool]]] = {}
        extra_path = path.parent / "aliases_extra.yaml"
        extra = yaml.safe_load(extra_path.read_text(encoding="utf-8")) if extra_path.exists() else {}
        for e in doc["entities"]:
            if e["id"] in extra:
                e["names"].setdefault("extra", []).extend(extra[e["id"]])
            info = EntityInfo(
                e["id"], e["kind"], e["name"], e.get("ticker"), e["country"], e["sector"], e.get("region")
            )
            self.entities[info.id] = info
            if info.ticker:
                self.by_ticker[info.ticker.upper()] = info
                self.by_ticker[info.id.split(".")[0].upper()] = info
            if info.kind == "sovereign":
                self.by_country[info.country] = info
            for _lang, names in e["names"].items():
                for alias in names:
                    for key, entry in self._alias_entries(alias, info, e.get("ambiguous", False)):
                        table.setdefault(key, []).append(entry)
        for key, entries in table.items():
            self._auto.add_word(key, (key, entries))
        self._auto.make_automaton()
        self.alias_count = len(table)

    @staticmethod
    def _alias_entries(
        alias: str, info: EntityInfo, ambiguous_entity: bool
    ) -> Iterable[tuple[str, tuple[str, str, bool, bool]]]:
        a = alias.strip()
        if not a or "(" in a or a.lower() in ALIAS_BLOCKLIST:
            return []
        spaced = _needs_boundary(a)
        if spaced:
            if len(a) < 2 or (len(a) == 2 and a not in TWO_LETTER_OK):
                return []
            if len(a) < 3 and not a.isupper():
                return []
        elif len(a) < 2:
            return []
        # Short Latin aliases match case-sensitively: "US" not "us", "Fed" not "fed up".
        case_sensitive = spaced and len(a) <= 4 and any(c.isupper() for c in a)
        ambiguous = spaced and (
            a.lower() in COMMON_WORDS or (ambiguous_entity and a.lower() == info.name.lower())
        )
        return [(_lower_same_len(a), (info.id, a, case_sensitive, ambiguous))]

    # ------------------------------------------------------------------ matching
    def _alias_mentions(self, text: str) -> list[Mention]:
        low = _lower_same_len(text)
        raw: list[Mention] = []
        for end_idx, (key, entries) in self._auto.iter(low):
            start = end_idx - len(key) + 1
            end = end_idx + 1
            for eid, alias, case_sensitive, ambiguous in entries:
                if _needs_boundary(alias) and (
                    (start > 0 and _is_word_char(text[start - 1]))
                    or (end < len(text) and _is_word_char(text[end]))
                ):
                    continue
                if case_sensitive and text[start:end] != alias:
                    continue
                if ambiguous and not self._has_context(text, start, end):
                    continue
                raw.append(Mention(self.entities[eid], text[start:end], start, end, "alias"))
                break
        # longest match wins on overlaps
        raw.sort(key=lambda m: (-(m.end - m.start), m.start))
        kept: list[Mention] = []
        for m in raw:
            if all(m.end <= k.start or m.start >= k.end for k in kept):
                kept.append(m)
        return sorted(kept, key=lambda m: m.start)

    @staticmethod
    def _has_context(text: str, start: int, end: int) -> bool:
        # the sentence containing the mention
        left = max(text.rfind(p, 0, start) for p in ".!?。\n")
        rights = [i for i in (text.find(p, end) for p in ".!?。\n") if i != -1]
        sentence = text[left + 1 : min(rights) if rights else len(text)]
        return bool(FINANCE_CONTEXT.search(sentence))

    def _ticker_mentions(self, text: str) -> list[Mention]:
        out = []
        for m in _CASHTAG.finditer(text):
            info = self.by_ticker.get(m.group(1).upper().replace("-", "."))
            if info:
                out.append(Mention(info, m.group(0), m.start(), m.end(), "cashtag"))
        letters = [c for c in text if c.isalpha()]
        mostly_caps = letters and sum(c.isupper() for c in letters) / len(letters) > 0.6
        if not mostly_caps:
            for m in _BARE.finditer(text):
                tok = m.group(1)
                if tok in BARE_TICKER_STOP:
                    continue
                info = self.by_ticker.get(tok)
                if info and info.kind == "equity" and info.ticker == tok and len(tok) >= 3:
                    out.append(Mention(info, tok, m.start(), m.end(), "ticker"))
        return out

    def link(self, text: str, meta: dict[str, Any] | None = None) -> LinkResult:
        meta = meta or {}
        mentions = self._alias_mentions(text)
        covered = [(m.start, m.end) for m in mentions]
        for tm in self._ticker_mentions(text):
            if all(tm.end <= s or tm.start >= e for s, e in covered):
                mentions.append(tm)
                covered.append((tm.start, tm.end))
        if not mentions:
            mentions += self._fallback(meta)
        return LinkResult(sorted(mentions, key=lambda m: m.start))

    def _fallback(self, meta: dict[str, Any]) -> list[Mention]:
        out: list[Mention] = []
        for tag in meta.get("tags", []) or []:  # per-ticker news feeds
            info = self.by_ticker.get(str(tag).upper())
            if info:
                out.append(Mention(info, str(tag), 0, 0, "tag"))
        if out:
            return out
        fips = (meta.get("gdelt_event") or {}).get("action_fips")
        if not fips:
            locs = meta.get("gdelt_locations") or []
            fips = next((loc.get("fips") for loc in locs if loc.get("type") in (1, 4, 5)), None) or (
                locs[0].get("fips") if locs else None
            )
        iso = FIPS_TO_ISO.get(fips or "")
        if iso and iso in self.by_country:
            out.append(Mention(self.by_country[iso], fips or iso, 0, 0, "geo"))
        return out


@lru_cache(maxsize=1)
def get_linker() -> EntityLinker:
    return EntityLinker()
