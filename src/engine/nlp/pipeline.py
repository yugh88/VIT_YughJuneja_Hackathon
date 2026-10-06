"""The seven-stage pipeline: RawDoc -> Signals (one per linked entity).

    normalise -> language ID -> de-duplicate -> entity linking
              -> entity-level sentiment -> event classification -> impact

Documents are processed in batches so each model runs once per batch. Each stage
is a separate module with typed inputs/outputs and can be swapped or tested alone.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from engine.config import REPO_ROOT
from engine.impact import heuristic
from engine.impact.features import (
    ActivityStats,
    EarningsCalendar,
    EntityActivity,
    PriceBook,
    build_feature_row,
    country_info,
    load_market_map,
    session_day,
    yf_ticker,
)
from engine.impact.model import ImpactOutput, ImpactScorer
from engine.nlp.dedupe import Deduplicator
from engine.nlp.entities import EntityInfo, EntityLinker, LinkResult, Mention, get_linker
from engine.nlp.events import EventModel, EventResult
from engine.nlp.langid import LangResult, LanguageIdentifier
from engine.nlp.normalise import CleanText, normalise
from engine.nlp.sentiment import SentimentModel, SentimentResult
from engine.schema import (
    DEFAULT_CREDIBILITY,
    Attention,
    EntityRef,
    Event,
    EventClass,
    Evidence,
    Impact,
    RawDoc,
    Sentiment,
    Signal,
    SourceRef,
    SourceType,
    utcnow,
)

log = logging.getLogger(__name__)
ONNX_DIR = REPO_ROOT / "models" / "onnx"
MIN_BASELINE_WINDOWS = 8  # burst z needs this much history before it counts as corroboration
SOVEREIGN_EVENTS = {
    EventClass.GEOPOLITICAL.value,
    EventClass.MACROECONOMIC.value,
    EventClass.CREDIT_EVENT.value,
}
WIRE_OUTLETS = {"reuters.com", "apnews.com", "bloomberg.com", "afp.com", "sec.gov", "federalreserve.gov",
                "ecb.europa.eu", "rbi.org.in", "ft.com", "wsj.com"}  # fmt: skip


@dataclass
class DocTrace:
    """Everything the pipeline learned about one document (used by /v1/analyze)."""

    doc: RawDoc
    clean: CleanText
    lang: LangResult
    duplicate_of: str | None = None
    syndication_count: int = 1
    links: LinkResult = field(default_factory=LinkResult)
    event: EventResult | None = None
    sentiments: dict[str, SentimentResult] = field(default_factory=dict)
    sentences: dict[str, str] = field(default_factory=dict)
    signals: list[Signal] = field(default_factory=list)
    dropped: str | None = None  # reason no signal was produced


@dataclass
class PipelineStats:
    docs: int = 0
    duplicates: int = 0
    unlinked: int = 0
    signals: int = 0
    by_language: dict[str, int] = field(default_factory=dict)


# CAMEO roots that signal state-level conflict: 15 force posture, 16 reduce relations,
# 17 coerce (sanctions, blockades), 19 fight, 20 mass violence. 18 (assault) is left
# out: GDELT uses it for domestic crime, which is not sovereign risk.
GEO_CONFLICT_ROOTS = {"15", "16", "17", "19", "20"}


def geo_fallback_ok(doc: RawDoc, ev: EventResult) -> bool:
    """May a GDELT location alone make this a sovereign signal?

    Only with corroboration: a conflict-coded GDELT event with a strong Goldstein
    score for geopolitics, or a confident macro/credit call on an article GDELT
    itself tagged with economic themes and that several outlets carried.
    """
    if ev.primary not in SOVEREIGN_EVENTS:
        return False
    gd = doc.meta.get("gdelt_event") or {}
    if ev.primary == EventClass.GEOPOLITICAL.value:
        return (
            gd.get("cameo_root") in GEO_CONFLICT_ROOTS
            and abs(gd.get("goldstein", 0)) >= 5
            and gd.get("num_sources", 0) >= 2
        )
    econ = any(t.startswith(("ECON_", "EPU_")) for t in doc.meta.get("gdelt_themes", []))
    return econ and ev.confidence >= 0.9 and (gd.get("num_sources", 0) >= 2 or gd.get("num_mentions", 0) >= 5)


def credibility(doc: RawDoc) -> float:
    if doc.source_type is SourceType.NEWS and doc.outlet in WIRE_OUTLETS:
        return 1.0
    if doc.source_type is SourceType.NEWS and doc.source == "gdelt_gkg":
        return 0.7  # long tail of outlets; GDELT carries everything from wires to blogs
    return DEFAULT_CREDIBILITY.get(doc.source_type, 0.5)


def _sentence_for(clean: CleanText, mentions: list[Mention]) -> str:
    """The sentences that mention the entity (title if the link came from metadata)."""
    if not clean.sentences:
        return clean.text
    spans, pos = [], 0
    for s in clean.sentences:
        i = clean.text.find(s, pos)
        spans.append((i if i >= 0 else pos, (i if i >= 0 else pos) + len(s)))
        pos = spans[-1][1]
    picked = []
    for m in mentions:
        if m.method in {"tag", "geo"}:
            continue
        for k, (a, b) in enumerate(spans):
            if a <= m.start < b + 2 and clean.sentences[k] not in picked:
                picked.append(clean.sentences[k])
    return " ".join(picked[:3]) if picked else clean.sentences[0]


class Pipeline:
    def __init__(
        self,
        onnx_dir: Path = ONNX_DIR,
        linker: EntityLinker | None = None,
        zero_shot: bool = True,
        dedupe: bool = True,
        time_scale: float = 1.0,
        impact_model: Path | None = None,
    ) -> None:
        self.langid = LanguageIdentifier()
        self.dedup = Deduplicator() if dedupe else None
        self.linker = linker or get_linker()
        self.sentiment = SentimentModel(onnx_dir)
        self.events = EventModel(onnx_dir, zero_shot=zero_shot)
        self.activity = EntityActivity(time_scale=time_scale)
        self.impact = ImpactScorer(impact_model) if impact_model else ImpactScorer()
        self.market = load_market_map(REPO_ROOT / "data" / "reference" / "market_map.yaml")
        snap = REPO_ROOT / "data" / "reference" / "price_snapshot.parquet"
        self.prices = PriceBook.load(snap) if snap.exists() else None
        self.earnings = EarningsCalendar(REPO_ROOT / "data" / "reference" / "earnings_dates.parquet")
        self.stats = PipelineStats()
        self._lock = threading.Lock()  # the stream worker and /v1/analyze share the models
        self.model_versions = {
            **self.sentiment.versions(),
            **self.events.versions(),
            "impact": self.impact.version,
            "langid": "fasttext-lid176" if self.langid.available else "script-heuristic",
        }

    def reset_state(self) -> None:
        """Forget stream history (dedup index, entity activity, counters); keep the models."""
        with self._lock:
            self.dedup = Deduplicator() if self.dedup is not None else None
            self.activity = EntityActivity(time_scale=self.activity.time_scale)
            self.stats = PipelineStats()

    @property
    def modes(self) -> dict[str, str]:
        return {
            "sentiment": self.sentiment.mode,
            "event": self.events.mode,
            "impact": self.impact.mode,
            "langid": "fasttext" if self.langid.available else "script",
        }

    # ------------------------------------------------------------------ stages 1-4
    def _prepare(self, doc: RawDoc, dedupe: bool = True) -> DocTrace:
        clean = normalise(doc.title, doc.text)
        lang = self.langid.detect(clean.text, hint=doc.language)
        tr = DocTrace(doc=doc, clean=clean, lang=lang)
        self.stats.docs += 1
        self.stats.by_language[lang.lang] = self.stats.by_language.get(lang.lang, 0) + 1
        if not clean.text:
            tr.dropped = "empty"
            return tr
        if dedupe and self.dedup is not None:
            dd = self.dedup.check(doc.doc_id, clean.text, lang.lang)
            tr.syndication_count = dd.syndication_count
            if dd.is_duplicate:
                tr.duplicate_of = dd.cluster_id
                tr.dropped = "duplicate"
                self.stats.duplicates += 1
                return tr
        tr.links = self.linker.link(clean.text, doc.meta)
        if not tr.links.mentions:
            tr.dropped = "no entity"
            self.stats.unlinked += 1
        return tr

    # ------------------------------------------------------------------ stages 5-7
    def process_batch(
        self, docs: list[RawDoc], dedupe: bool = True, zero_shot: bool = True
    ) -> list[DocTrace]:
        with self._lock:
            return self._process(docs, dedupe, zero_shot)

    def _process(self, docs: list[RawDoc], dedupe: bool, zero_shot: bool) -> list[DocTrace]:
        t0 = time.perf_counter()
        traces = [self._prepare(d, dedupe) for d in docs]
        live = [t for t in traces if t.dropped is None]
        if dedupe and self.dedup is not None:
            for t in live:  # copies that arrived later in the same batch count toward the head
                t.syndication_count = self.dedup.count_of(t.doc.doc_id)
        events = self.events.classify_batch(
            [t.clean.text for t in live], [t.doc.meta for t in live], allow_zero_shot=zero_shot
        )
        sent_items: list[tuple[DocTrace, EntityInfo, str]] = []
        for tr, ev in zip(live, events, strict=True):
            tr.event = ev
            if all(m.method == "geo" for m in tr.links.mentions) and not geo_fallback_ok(tr.doc, ev):
                tr.dropped = "no entity"  # a dateline alone does not make local news a sovereign signal
                self.stats.unlinked += 1
                continue
            ents = tr.links.entities()
            for e in ents:
                if e.kind == "sovereign" and ev.primary not in SOVEREIGN_EVENTS:
                    continue  # "Apple sales fall in China" is about Apple; "US man wins prize" is not about US risk
                ms = [m for m in tr.links.mentions if m.entity.id == e.id]
                sentence = _sentence_for(tr.clean, ms)
                tr.sentences[e.id] = sentence
                sent_items.append((tr, e, sentence))
        results = self.sentiment.score_batch(
            [(e.name, s, tr.lang.lang, tr.lang.confidence) for tr, e, s in sent_items]
        )
        now = utcnow()
        feats, fallbacks, acts = [], [], []
        for (tr, e, _sentence), sres in zip(sent_items, results, strict=True):
            tr.sentiments[e.id] = sres
            f, h, act = self._impact_inputs(tr, e, sres, update=dedupe)
            feats.append(f)
            fallbacks.append(h)
            acts.append(act)
        impacts = self.impact.score(feats, fallbacks)
        for (tr, e, sentence), sres, imp, act in zip(sent_items, results, impacts, acts, strict=True):
            tr.signals.append(self._signal(tr, e, sentence, sres, imp, act, now))
        self.stats.signals += sum(len(t.signals) for t in traces)
        del t0
        return traces

    def _impact_inputs(
        self, tr: DocTrace, e: EntityInfo, s: SentimentResult, update: bool
    ) -> tuple[dict[str, float], heuristic.ImpactFeatures, ActivityStats]:
        doc, ev = tr.doc, tr.event
        assert ev is not None
        gd = doc.meta.get("gdelt_event") or {}
        cred = credibility(doc)
        act = self.activity.observe(
            e.id, doc.published_at.timestamp(), doc.outlet, tr.clean.text, tr.lang.lang, update
        )
        info = country_info(e.country, self.market)
        tick = yf_ticker(e.id, e.kind, self.market) or info["index"]
        asof = self.prices.latest_date() + pd.Timedelta(days=1) if self.prices else None
        row = build_feature_row(
            sentiment=s.score,
            confidence=s.confidence,
            event=ev.primary,
            event_confidence=ev.confidence,
            text=tr.clean.text,
            credibility=cred,
            syndication=tr.syndication_count,
            activity=act,
            vol_20d=self.prices.vol_20d(tick, asof) if self.prices and asof is not None else float("nan"),
            log_dollar_volume_20d=(
                self.prices.log_dollar_volume_20d(tick, asof)
                if self.prices and asof is not None and e.kind != "sovereign"
                else float("nan")
            ),
            country=e.country,
            sector=e.sector,
            kind=e.kind,
            source_type=doc.source_type.value,
            gdelt_event=gd or None,
            published_at=doc.published_at,
            close_utc=info["close_utc"],
            earnings_window=self.earnings.window(
                yf_ticker(e.id, e.kind, self.market), session_day(doc.published_at, info["close_utc"])
            ),
        )
        h = heuristic.ImpactFeatures(
            event=ev.primary,
            abs_sentiment=abs(s.score),
            confidence=s.confidence,
            credibility=cred,
            severity_hits=heuristic.severity_hits(tr.clean.text),
            syndication_count=tr.syndication_count,
            goldstein=gd.get("goldstein"),
            num_sources=gd.get("num_sources"),
        )
        return row, h, act

    def _signal(
        self,
        tr: DocTrace,
        e: EntityInfo,
        sentence: str,
        s: SentimentResult,
        imp: ImpactOutput,
        act: ActivityStats,
        now: Any,
    ) -> Signal:
        doc, ev = tr.doc, tr.event
        assert ev is not None
        gd = doc.meta.get("gdelt_event") or {}
        cred = credibility(doc)
        geo = None
        if gd.get("lat") is not None:
            geo = {
                "fips": gd.get("action_fips"),
                "name": gd.get("action_geo"),
                "lat": gd["lat"],
                "lon": gd["lon"],
            }
        elif doc.meta.get("gdelt_locations"):
            loc = doc.meta["gdelt_locations"][0]
            if loc.get("lat") is not None:
                geo = {"fips": loc.get("fips"), "name": loc.get("name"), "lat": loc["lat"], "lon": loc["lon"]}
        version = "+".join(sorted(self.model_versions.values()))
        latency = max(0, int((now - doc.ingested_at).total_seconds() * 1000)) if doc.ingested_at else 0
        return Signal(
            signal_id=Signal.make_id(doc.doc_id, e.id, version),
            doc_id=doc.doc_id,
            published_at=doc.published_at,
            ingested_at=doc.ingested_at,
            source=SourceRef(name=doc.source, outlet=doc.outlet, type=doc.source_type, credibility=cred),
            language=tr.lang.lang,
            entity=EntityRef(id=e.id, name=e.name, country=e.country, sector=e.sector),
            sentiment=Sentiment(
                score=s.score,
                score_human=s.score_human,
                label=s.label,  # type: ignore[arg-type]
                confidence=s.confidence,
            ),
            event=Event(
                primary=EventClass(ev.primary),
                confidence=ev.confidence,
                secondary=[EventClass(c) for c in ev.secondary],
            ),
            impact=Impact(score=imp.score, low=imp.low, high=imp.high, drivers=imp.drivers, mode=imp.mode),  # type: ignore[arg-type]
            novelty=act.novelty,
            # A cold start (replay, fresh deploy) makes every first mention look like a burst.
            attention=Attention(
                # ...and for a rarely-mentioned entity a single mention against an all-zero history
                # is z ~ 2-6, so the entity needs some history of its own too.
                burst_z=act.burst_z
                if act.baseline_windows >= MIN_BASELINE_WINDOWS and act.baseline_active >= 3
                else 0.0,
                outlets_window=act.outlets_window,
            ),
            syndication_count=tr.syndication_count,
            evidence=Evidence(title=doc.title[:300], sentence=sentence[:500], url=doc.url),
            geo=geo,
            model_versions={
                "sentiment": s.model,
                "event": f"{self.model_versions['event']} ({ev.method})",
                "impact": self.impact.version,
            },
            latency_ms=latency,
        )

    def analyze(self, text: str, language: str | None = None) -> DocTrace:
        """Score arbitrary text (any language) without touching the stream's dedup state."""
        doc = RawDoc(
            doc_id=RawDoc.make_id("analyze", text),
            source="analyze",
            source_type=SourceType.NEWS,
            outlet="user",
            title="",
            text=text,
            language=language,
            published_at=utcnow(),
        )
        return self.process_batch([doc], dedupe=False)[0]


_SHARED: dict[tuple[bool, float], Pipeline] = {}
_SHARED_LOCK = threading.Lock()


def get_pipeline(zero_shot: bool = True, time_scale: float = 1.0) -> Pipeline:
    """One pipeline (and one set of ONNX sessions) per process.

    Creating and destroying ONNX Runtime sessions repeatedly from different threads
    deadlocked the test suite; a server needs only one anyway. Stream state is reset
    with :meth:`Pipeline.reset_state` when a new app starts.
    """
    with _SHARED_LOCK:
        key = (zero_shot, time_scale)
        if key not in _SHARED:
            _SHARED[key] = Pipeline(zero_shot=zero_shot, time_scale=time_scale)
        return _SHARED[key]
