"""Typed contracts shared by every stage.

``RawDoc`` is what a connector emits; ``Signal`` is what the engine publishes. The
signal schema is versioned: any breaking change bumps ``SCHEMA_VERSION`` and the
contract test in ``tests/test_schema.py`` fails until the snapshot is updated.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.0"


def utcnow() -> datetime:
    return datetime.now(UTC)


def stable_hash(*parts: str, length: int = 16) -> str:
    """Deterministic short hash used for document and signal ids."""
    h = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()
    return h[:length]


class SourceType(StrEnum):
    NEWS = "news"
    SOCIAL = "social"
    FILING = "filing"
    CENTRAL_BANK = "central_bank"


# Credibility tiers from the plan (section 6): wire/filing 1.0, major press 0.9, blog 0.5, social 0.3.
DEFAULT_CREDIBILITY: dict[SourceType, float] = {
    SourceType.NEWS: 0.9,
    SourceType.SOCIAL: 0.3,
    SourceType.FILING: 1.0,
    SourceType.CENTRAL_BANK: 1.0,
}


class EventClass(StrEnum):
    GEOPOLITICAL = "Geopolitical"
    MACROECONOMIC = "Macroeconomic"
    CREDIT_EVENT = "Credit Event"
    MERGER_ACQUISITION = "Merger/Acquisition"
    PRODUCT_LAUNCH = "Product Launch"
    EARNINGS_GUIDANCE = "Earnings/Guidance"
    REGULATORY_LEGAL = "Regulatory/Legal"
    MANAGEMENT_CHANGE = "Management Change"
    OPERATIONAL_SUPPLY_CHAIN = "Operational/Supply Chain"
    ESG_CLIMATE = "ESG/Climate"
    CYBER_TECH = "Cyber/Technology Risk"
    OTHER = "Other/Non-event"


class RawDoc(BaseModel):
    """One unit of unstructured text as it arrives from a source."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    source: str  # connector name, e.g. "gdelt_gkg", "rss", "mastodon"
    source_type: SourceType
    outlet: str  # domain or account, e.g. "reuters.com", "mastodon.social/@x"
    url: str = ""
    title: str = ""
    text: str = ""
    language: str | None = None  # hint from the source; the NLP stage re-detects
    author: str | None = None
    published_at: datetime
    ingested_at: datetime = Field(default_factory=utcnow)
    replayed: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def make_id(source: str, key: str) -> str:
        return f"{source}:{stable_hash(source, key)}"

    @property
    def body(self) -> str:
        """Title and text joined, the input to the NLP pipeline."""
        if self.text and self.title and not self.text.startswith(self.title):
            return f"{self.title}. {self.text}"
        return self.text or self.title


class SourceRef(BaseModel):
    name: str
    outlet: str
    type: SourceType
    credibility: float = Field(ge=0.0, le=1.0)


class EntityRef(BaseModel):
    id: str
    name: str
    country: str | None = None
    sector: str | None = None


class Sentiment(BaseModel):
    # score: full-range map (drives Module A and the UI); score_human: FiQA-faithful map.
    # Both are monotone in the same model output; docs/DECISIONS.md D38.
    score: float = Field(ge=-1.0, le=1.0)
    score_human: float | None = Field(default=None, ge=-1.0, le=1.0)
    label: Literal["negative", "neutral", "positive"]
    confidence: float = Field(ge=0.0, le=1.0)


class Event(BaseModel):
    primary: EventClass
    confidence: float = Field(ge=0.0, le=1.0)
    secondary: list[EventClass] = Field(default_factory=list)


class ImpactEffect(BaseModel):
    """How one feature moved the impact prediction vs a typical story (occlusion; + raised it)."""

    feature: str
    value: str
    effect: float


class Impact(BaseModel):
    score: int = Field(ge=1, le=10)
    low: int = Field(ge=1, le=10)
    high: int = Field(ge=1, le=10)
    drivers: list[str] = Field(default_factory=list)
    mode: Literal["model", "heuristic"] = "model"
    effects: list[ImpactEffect] = Field(default_factory=list)
    cap_reason: str | None = None  # why the score was capped (DECISIONS D71)


class Attention(BaseModel):
    """How much the stream is talking about this entity right now (used for corroboration)."""

    burst_z: float = 0.0  # mentions in the current window vs. the 7-day baseline; 0 until 8 windows of history and 3 with mentions of this entity
    outlets_window: int = 1  # distinct outlets naming the entity in the current window


class Evidence(BaseModel):
    title: str = ""
    sentence: str = ""
    url: str = ""


class Signal(BaseModel):
    """One scored (document, entity) pair. Immutable once published."""

    model_config = ConfigDict(frozen=True)

    signal_id: str
    schema_version: str = SCHEMA_VERSION
    doc_id: str
    published_at: datetime
    ingested_at: datetime
    source: SourceRef
    language: str
    entity: EntityRef
    sentiment: Sentiment
    event: Event
    impact: Impact
    novelty: float = Field(ge=0.0, le=1.0)
    syndication_count: int = Field(ge=1)
    evidence: Evidence
    attention: Attention | None = None
    geo: dict[str, Any] | None = None  # {"country": "UA", "lat": .., "lon": ..} when known
    original_published_at: datetime | None = None  # replayed items: the recording's own publish time
    model_versions: dict[str, str]
    latency_ms: int = Field(ge=0)

    @field_validator("published_at", "ingested_at")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        return v if v.tzinfo else v.replace(tzinfo=UTC)

    @staticmethod
    def make_id(doc_id: str, entity_id: str, model_version: str) -> str:
        """Idempotency key: replays of the same doc never double-count."""
        return "sig_" + stable_hash(doc_id, entity_id, model_version, length=20)
