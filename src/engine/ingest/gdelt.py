"""GDELT 2.0 connectors (keyless).

* :class:`GdeltGkgSource` reads the 15-minute GKG drop (English and the
  machine-translated "translingual" stream), joins it to the same window's Events
  export by URL to pick up CAMEO codes, Goldstein scale and mention counts, and
  keeps only finance/risk-relevant articles.
* :class:`GdeltDocSource` queries the DOC 2.0 API, which returns article titles in
  their original language - our main multilingual live feed.

Column positions follow the GDELT 2.0 codebooks (GKG 2.1: 27 columns, Events: 61).
"""

from __future__ import annotations

import csv
import html
import io
import logging
import re
import sys
import zipfile
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx

from engine.ingest.base import Source
from engine.ingest.text import domain_of, parse_dt
from engine.schema import RawDoc, SourceType

log = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)

GDELT_BASE = "https://data.gdeltproject.org/gdeltv2"

# GKG 2.1 column indices
G_DATE, G_SOURCE, G_URL = 1, 3, 4
G_THEMES, G_LOCATIONS, G_ORGS, G_TONE, G_TRANSLATION, G_EXTRAS = 7, 9, 13, 15, 25, 26
# Events 2.0 column indices
E_ROOT, E_CODE, E_QUAD, E_GOLDSTEIN, E_MENTIONS, E_SOURCES, E_ARTICLES, E_TONE = (
    28,
    26,
    29,
    30,
    31,
    32,
    33,
    34,
)
E_GEO_NAME, E_GEO_CC, E_GEO_LAT, E_GEO_LON, E_URL = 52, 53, 56, 57, 60

# Themes that mark an article as relevant to markets, credit or macro/geopolitical risk.
RELEVANT_THEME_PREFIXES = (
    "ECON_",
    "EPU_",
    "WB_1104",
    "WB_318",
    "WB_1920",
    "WB_442",
    "WB_2443",
    "WB_1235",  # macro, finance
    "TAX_FNCACT_CEO",
    "TAX_FNCACT_BANKER",
    "TAX_FNCACT_INVESTOR",
    "SANCTIONS",
    "ARMEDCONFLICT",
    "MILITARY",
    "TERROR",
    "CYBER_ATTACK",
    "STRIKE",
    "ENV_OIL",
    "ENV_GAS",
    "ENV_CLIMATECHANGE",
    "NATURAL_DISASTER",
    "MANMADE_DISASTER",
    "CORRUPTION",
    "BAN",
    "TRADE_DISPUTE",
    "TAX_POLICY",
    "UNEMPLOYMENT",
    "FUELPRICES",
    "BANKRUPTCY",
    "MERGER",
    "ACQUISITION",
    "INFLATION",
)

_TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)
_SRCLANG = re.compile(r"srclc:([a-z]{3})")

# ISO 639-3 (GDELT translation info) -> ISO 639-1 for the languages we expect most.
ISO3_TO_1 = {
    "eng": "en",
    "fra": "fr",
    "deu": "de",
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "rus": "ru",
    "zho": "zh",
    "jpn": "ja",
    "kor": "ko",
    "ara": "ar",
    "hin": "hi",
    "tur": "tr",
    "ukr": "uk",
    "pol": "pl",
    "nld": "nl",
    "swe": "sv",
    "ind": "id",
    "vie": "vi",
    "tha": "th",
    "fas": "fa",
    "heb": "he",
    "ben": "bn",
    "urd": "ur",
    "ell": "el",
    "ces": "cs",
    "ron": "ro",
    "hun": "hu",
    "bul": "bg",
    "srp": "sr",
    "hrv": "hr",
    "bos": "bs",
    "slk": "sk",
    "slv": "sl",
    "lit": "lt",
    "lav": "lv",
    "est": "et",
    "fin": "fi",
    "dan": "da",
    "nor": "no",
    "nob": "no",
    "cat": "ca",
    "msa": "ms",
    "zsm": "ms",
    "tam": "ta",
    "tel": "te",
    "mar": "mr",
    "guj": "gu",
    "kan": "kn",
    "mal": "ml",
    "pan": "pa",
    "sqi": "sq",
    "mkd": "mk",
    "aze": "az",
    "kat": "ka",
    "hye": "hy",
    "swa": "sw",
    "afr": "af",
    "isl": "is",
    "glg": "gl",
    "eus": "eu",
    "kaz": "kk",
    "uzb": "uz",
    "mon": "mn",
    "nep": "ne",
    "sin": "si",
    "khm": "km",
    "lao": "lo",
    "mya": "my",
    "amh": "am",
    "som": "so",
    "hau": "ha",
    "yor": "yo",
    "tgl": "tl",
    "fil": "tl",
    "bel": "be",
    "pus": "ps",
}


def _relevant(themes: str) -> bool:
    return any(t.startswith(RELEVANT_THEME_PREFIXES) for t in themes.split(";") if t)


def parse_tone(tone: str) -> dict[str, float]:
    keys = (
        "tone",
        "positive",
        "negative",
        "polarity",
        "activity_density",
        "self_group_density",
        "word_count",
    )
    out: dict[str, float] = {}
    for k, v in zip(keys, tone.split(","), strict=False):
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def parse_locations(v1: str, limit: int = 5) -> list[dict[str, Any]]:
    """V1LOCATIONS: type#name#FIPS-country#adm1#lat#lon#featureid; separated by ';'."""
    locs = []
    for chunk in v1.split(";"):
        f = chunk.split("#")
        if len(f) < 6 or not f[2]:
            continue
        try:
            lat, lon = float(f[4]), float(f[5])
        except ValueError:
            lat = lon = None  # type: ignore[assignment]
        locs.append(
            {"type": int(f[0]) if f[0].isdigit() else 0, "name": f[1], "fips": f[2], "lat": lat, "lon": lon}
        )
        if len(locs) >= limit:
            break
    return locs


def parse_events(raw: bytes) -> dict[str, dict[str, Any]]:
    """Map SOURCEURL -> strongest event (largest |Goldstein| x mentions) in an Events export."""
    best: dict[str, dict[str, Any]] = {}
    reader = csv.reader(io.StringIO(raw.decode("utf-8", errors="replace")), delimiter="\t")
    for r in reader:
        if len(r) < 61:
            continue
        try:
            ev: dict[str, Any] = {
                "cameo_root": r[E_ROOT],
                "cameo_code": r[E_CODE],
                "quad_class": int(r[E_QUAD] or 0),
                "goldstein": float(r[E_GOLDSTEIN] or 0),
                "num_mentions": int(r[E_MENTIONS] or 0),
                "num_sources": int(r[E_SOURCES] or 0),
                "num_articles": int(r[E_ARTICLES] or 0),
                "avg_tone": float(r[E_TONE] or 0),
                "action_geo": r[E_GEO_NAME],
                "action_fips": r[E_GEO_CC],
                "lat": float(r[E_GEO_LAT]) if r[E_GEO_LAT] else None,
                "lon": float(r[E_GEO_LON]) if r[E_GEO_LON] else None,
            }
        except ValueError:
            continue
        url = r[E_URL]
        score = abs(ev["goldstein"]) * max(ev["num_mentions"], 1)
        cur = best.get(url)
        if cur is None or score > cur["_score"]:
            ev["_score"] = score
            best[url] = ev
    for ev in best.values():
        ev.pop("_score", None)
    return best


def parse_gkg(
    raw: bytes,
    events: dict[str, dict[str, Any]] | None = None,
    translated: bool = False,
    relevant_only: bool = True,
    max_docs: int | None = None,
) -> list[RawDoc]:
    """Turn one GKG 2.1 file into RawDocs. Pure function: unit-tested on a fixture."""
    events = events or {}
    docs: list[RawDoc] = []
    reader = csv.reader(io.StringIO(raw.decode("utf-8", errors="replace")), delimiter="\t")
    for r in reader:
        if len(r) < 27:
            continue
        url, themes = r[G_URL], r[G_THEMES]
        if not url.startswith("http"):
            continue
        if relevant_only and not _relevant(themes):
            continue
        m = _TITLE.search(r[G_EXTRAS])
        title = html.unescape(m.group(1)).strip() if m else ""
        if len(title) < 8:  # drops "Home", "News" and other non-titles; CJK titles can be short
            continue
        lang = "en"
        if translated:
            lm = _SRCLANG.search(r[G_TRANSLATION])
            lang = ISO3_TO_1.get(lm.group(1), lm.group(1)) if lm else "und"
        published = parse_dt(r[G_DATE]) or datetime.now(UTC)
        top_themes = [t for t in themes.split(";") if t][:12]
        meta: dict[str, Any] = {
            "gdelt_tone": parse_tone(r[G_TONE]),
            "gdelt_themes": top_themes,
            "gdelt_locations": parse_locations(r[G_LOCATIONS]),
            "gdelt_orgs": [o for o in r[G_ORGS].split(";") if o][:10],
            "gdelt_translated": translated,
        }
        if url in events:
            meta["gdelt_event"] = events[url]
        docs.append(
            RawDoc(
                doc_id=RawDoc.make_id("gdelt", url),
                source="gdelt_gkg",
                source_type=SourceType.NEWS,
                outlet=r[G_SOURCE] or domain_of(url),
                url=url,
                title=title,
                text="",
                # Titles in the translingual stream keep their original language; the
                # English GKG fields (themes, tone) are GDELT's translation.
                language=lang,
                published_at=published,
                meta=meta,
            )
        )
        if max_docs and len(docs) >= max_docs:
            break
    return docs


def _unzip(blob: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        return zf.read(zf.namelist()[0])


class GdeltGkgSource(Source):
    source_type = SourceType.NEWS
    poll_interval_s = 300.0

    def __init__(
        self, name: str = "gdelt_gkg", translation: bool = True, max_docs_per_file: int = 600, **kw: Any
    ):
        super().__init__(name, **kw)
        self.translation = translation
        self.max_docs = max_docs_per_file
        self._last: dict[str, str] = {}

    @staticmethod
    def _latest(manifest: str, suffix: str) -> str | None:
        for line in manifest.splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[2].endswith(suffix):
                return parts[2]
        return None

    async def _stream(self, client: httpx.AsyncClient, manifest_name: str, translated: bool) -> list[RawDoc]:
        manifest = (await client.get(f"{GDELT_BASE}/{manifest_name}")).raise_for_status().text
        gkg_url = self._latest(manifest, ".gkg.csv.zip")
        if not gkg_url or self._last.get(manifest_name) == gkg_url:
            return []
        ev_url = self._latest(manifest, ".export.CSV.zip")
        events: dict[str, dict[str, Any]] = {}
        if ev_url:
            try:
                events = parse_events(_unzip((await client.get(ev_url)).raise_for_status().content))
            except (httpx.HTTPError, zipfile.BadZipFile) as exc:
                log.info("gdelt events join skipped: %s", exc)
        gkg = _unzip((await client.get(gkg_url, timeout=120)).raise_for_status().content)
        self._last[manifest_name] = gkg_url
        return parse_gkg(gkg, events, translated=translated, max_docs=self.max_docs)

    async def fetch_window(self, client: httpx.AsyncClient, stamp: str) -> list[RawDoc]:
        """Fetch one historical 15-minute window, e.g. ``stamp='20220224043000'``.

        Used by the recorder to backfill replay twins and to build the
        24 Feb 2022 scenario replay. Missing files (GDELT has a few gaps) yield [].
        """
        docs: list[RawDoc] = []
        streams = [("", False)] + ([(".translation", True)] if self.translation else [])
        for infix, translated in streams:
            gkg_url = f"{GDELT_BASE}/{stamp}{infix}.gkg.csv.zip"
            ev_url = f"{GDELT_BASE}/{stamp}{infix}.export.CSV.zip"
            try:
                gkg_resp = await client.get(gkg_url, timeout=120)
                if gkg_resp.status_code == 404:
                    continue
                gkg = _unzip(gkg_resp.raise_for_status().content)
                ev_resp = await client.get(ev_url, timeout=60)
                events = parse_events(_unzip(ev_resp.content)) if ev_resp.status_code == 200 else {}
            except zipfile.BadZipFile:
                continue
            docs += parse_gkg(gkg, events, translated=translated, max_docs=self.max_docs)
        return docs

    async def fetch(self, client: httpx.AsyncClient) -> list[RawDoc]:
        docs = await self._stream(client, "lastupdate.txt", translated=False)
        if self.translation:
            docs += await self._stream(client, "lastupdate-translation.txt", translated=True)
        return docs


class GdeltDocSource(Source):
    """DOC 2.0 article search. GDELT asks for at most one request per 5 seconds, so
    queries run sequentially with a pause and the poll interval is long."""

    source_type = SourceType.NEWS
    poll_interval_s = 900.0
    API = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(
        self,
        name: str = "gdelt_doc",
        queries: list[str] | None = None,
        max_records: int = 75,
        timespan: str = "30min",
        pause_s: float = 6.0,
        **kw: Any,
    ):
        super().__init__(name, **kw)
        self.queries = queries or ["theme:ECON_STOCKMARKET"]
        self.max_records = max_records
        self.timespan = timespan
        self.pause_s = pause_s
        self._cooldown_until = 0.0
        self.rate_limited = 0

    @staticmethod
    def parse(payload: dict[str, Any], query: str) -> list[RawDoc]:
        docs = []
        for a in payload.get("articles", []) or []:
            url, title = a.get("url", ""), (a.get("title") or "").strip()
            if not url or not title:
                continue
            docs.append(
                RawDoc(
                    doc_id=RawDoc.make_id("gdelt", url),
                    source="gdelt_doc",
                    source_type=SourceType.NEWS,
                    outlet=a.get("domain") or domain_of(url),
                    url=url,
                    title=title,
                    language=None,  # DOC API gives a language *name*; the NLP stage detects the code
                    published_at=parse_dt(a.get("seendate")) or datetime.now(UTC),
                    meta={
                        "gdelt_query": query,
                        "language_name": a.get("language"),
                        "source_country": a.get("sourcecountry"),
                    },
                )
            )
        return docs

    cooldown_s = 900.0  # after HTTP 429, back off this long (or Retry-After, if longer)

    async def fetch(self, client: httpx.AsyncClient) -> list[RawDoc]:
        import asyncio
        import time

        if time.monotonic() < self._cooldown_until:
            return []  # rate-limited earlier: stay quiet instead of hammering the API
        out: list[RawDoc] = []
        for i, q in enumerate(self.queries):
            if i:
                await asyncio.sleep(self.pause_s)
            url = (
                f"{self.API}?query={quote(q)}&mode=artlist&format=json&sort=datedesc"
                f"&maxrecords={self.max_records}&timespan={self.timespan}"
            )
            resp = await client.get(url, timeout=60)
            if resp.status_code == 429:
                retry = resp.headers.get("retry-after", "")
                wait = max(self.cooldown_s, float(retry) if retry.isdigit() else 0.0)
                self._cooldown_until = time.monotonic() + wait
                self.rate_limited += 1
                log.warning("GDELT DOC API rate-limited (429); backing off %.0f s", wait)
                return out
            resp.raise_for_status()
            if "json" not in resp.headers.get("content-type", "") and not resp.text.lstrip().startswith("{"):
                raise RuntimeError(f"GDELT DOC non-JSON reply: {resp.text[:120]}")
            out += self.parse(resp.json(), q)
        return out
