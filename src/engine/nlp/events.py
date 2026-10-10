"""Stage 6 - event classification into the 12-class taxonomy.

Primary path: a fine-tuned multilingual mDeBERTa classifier. When its confidence
is below 0.55 the zero-shot NLI model (mDeBERTa XNLI, any language) re-scores the
document. Structured evidence a source already carries outranks a weak model
guess: SEC 8-K item codes, and GDELT CAMEO codes 18-20 (assault, fight, mass
violence) with a strongly negative Goldstein score.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from engine.nlp.credit import credit_phrase, credit_rule_v2, hard_negative
from engine.nlp.onnx_model import OnnxClassifier, softmax
from engine.schema import EventClass

log = logging.getLogger(__name__)
CLASSES = [e.value for e in EventClass]
CREDIT = EventClass.CREDIT_EVENT.value
REGULATORY = EventClass.REGULATORY_LEGAL.value
EDGAR_CREDIT_ITEMS = {"1.03", "2.04", "3.01"}  # bankruptcy; acceleration of obligations; delisting notice
FALLBACK_THRESHOLD = 0.55
SECONDARY_THRESHOLD = 0.2
CREDIT_RULE_V2 = {"rules_v2": True, "threshold": 0.7}

# Natural-language descriptions used as NLI hypotheses ("This text is about {}.").
DESCRIPTIONS = {
    EventClass.GEOPOLITICAL: "war, conflict, sanctions or geopolitical tension",
    EventClass.MACROECONOMIC: "inflation, interest rates, central banks or the economy",
    EventClass.CREDIT_EVENT: "a debt default, bankruptcy or credit rating downgrade",
    EventClass.MERGER_ACQUISITION: "a merger, acquisition or takeover",
    EventClass.PRODUCT_LAUNCH: "a new product launch or approval",
    EventClass.EARNINGS_GUIDANCE: "company earnings, results or financial guidance",
    EventClass.REGULATORY_LEGAL: "a lawsuit, regulation, fine or investigation",
    EventClass.MANAGEMENT_CHANGE: "a change of CEO or company leadership",
    EventClass.OPERATIONAL_SUPPLY_CHAIN: "supply chain disruption, strikes, recalls or plant shutdowns",
    EventClass.ESG_CLIMATE: "climate, emissions or environmental and social issues",
    EventClass.CYBER_TECH: "a cyberattack, data breach or IT outage",
    EventClass.OTHER: "general market commentary",
}

# Rules for when no model is available at all (heuristic mode).
_RULES = [
    (
        EventClass.CREDIT_EVENT,
        r"default|bankrupt|insolven|chapter 11|downgrad|junk|missed (coupon|payment)|restructur",
    ),
    (
        EventClass.GEOPOLITICAL,
        r"\bwar\b|invasion|missile|sanction|troops|military|coup|tariff|conflict|attack",
    ),
    (EventClass.CYBER_TECH, r"cyber|ransomware|hack|data breach|outage"),
    (
        EventClass.MACROECONOMIC,
        r"inflation|interest rate|rate (hike|cut)|central bank|\bfed\b|gdp|recession|unemployment|yield",
    ),
    (EventClass.MERGER_ACQUISITION, r"acquir|acquisition|merger|takeover|buyout"),
    (EventClass.EARNINGS_GUIDANCE, r"earnings|quarterly|results|revenue|profit|guidance|\bq[1-4]\b|dividend"),
    (
        EventClass.REGULATORY_LEGAL,
        r"lawsuit|sue|fine[sd]?\b|antitrust|regulator|probe|investigation|court|settle",
    ),
    (
        EventClass.MANAGEMENT_CHANGE,
        r"\bceo\b|chief executive|resign|steps down|appoint|names .* (ceo|cfo|chair)",
    ),
    (EventClass.OPERATIONAL_SUPPLY_CHAIN, r"supply chain|shortage|recall|strike|shutdown|disruption"),
    (EventClass.ESG_CLIMATE, r"climate|emission|carbon|\besg\b|sustainab|pollut"),
    (EventClass.PRODUCT_LAUNCH, r"launch|unveil|introduc|rolls out|new (product|model)|approval"),
]
_RULES_RX = [(c, re.compile(p, re.I)) for c, p in _RULES]


@dataclass(frozen=True)
class EventResult:
    primary: str
    confidence: float
    secondary: list[str]
    probs: dict[str, float]
    method: str  # model | zero-shot | structured | rules


def apply_hard_negative(
    text: str, p: np.ndarray, primary: str, conf: float, method: str
) -> tuple[str, float, str]:
    """Generic credit hard negatives (validated on validation only; training.credit_tune hardneg)."""
    hn = hard_negative(text)
    if hn == "aml" and primary != REGULATORY:
        return REGULATORY, max(float(p[CLASSES.index(REGULATORY)]), 0.6), method + "+aml"
    if hn == "not_credit" and primary == CREDIT:
        alt = next(k for k in np.argsort(-p) if CLASSES[int(k)] != CREDIT)
        return CLASSES[int(alt)], float(p[alt]), method + "+hardneg"
    return primary, conf, method


class EventModel:
    """Fine-tuned classifier, plus the zero-shot NLI fallback loaded lazily.

    The NLI model (~366 MB) is not part of the default download: on first need it is
    loaded if present, otherwise fetched in the background from the release while the
    classifier's answer is used as-is.
    """

    def __init__(self, onnx_dir: Path, zero_shot: bool = True, auto_download: bool = True) -> None:
        clf = onnx_dir / "event-multi"
        self._zs_dir = onnx_dir / "zeroshot-nli"
        self._zs: OnnxClassifier | None = None
        self.zero_shot = zero_shot
        self.auto_download = auto_download
        self.clf = OnnxClassifier(clf) if (clf / "model.int8.onnx").exists() else None
        # Credit decision rule tuned on validation (training.credit_tune); empty = plain argmax.
        self.credit_cfg: dict[str, Any] = dict(self.clf.cfg.get("credit_rule") or {}) if self.clf else {}
        if self.credit_cfg:  # v2 credit rules and threshold, selected on credit_v2_dev (DECISIONS D74)
            self.credit_cfg.update(CREDIT_RULE_V2)
        self._spec = None
        spec_path = clf / "credit_specialist.joblib"
        if self.credit_cfg.get("specialist_weight") and spec_path.exists():
            import joblib

            self._spec = joblib.load(spec_path)
        if self.clf is None:
            log.warning("no fine-tuned event model: %s", "zero-shot NLI" if self.zs else "keyword rules")

    @property
    def zs(self) -> OnnxClassifier | None:
        if self._zs is None and self.zero_shot:
            if (self._zs_dir / "model.int8.onnx").exists():
                self._zs = OnnxClassifier(self._zs_dir)
            elif self.auto_download:
                from engine.models_download import fetch_lazy_in_background

                fetch_lazy_in_background("zeroshot-nli")
        return self._zs

    def nli(self) -> OnnxClassifier | None:
        """The NLI model for other stages (multilingual sentiment, D78), loaded even when the event
        fallback is off; fetched in the background on first need when missing."""
        if self._zs is None and (self._zs_dir / "model.int8.onnx").exists():
            self._zs = OnnxClassifier(self._zs_dir)
        elif self._zs is None and self.auto_download:
            from engine.models_download import fetch_lazy_in_background

            fetch_lazy_in_background("zeroshot-nli")
        return self._zs

    def _specialist(self, texts: list[str]) -> np.ndarray | None:
        return None if self._spec is None else self._spec.predict_proba(texts)[:, 1]

    def _credit_prob(self, text: str, p: np.ndarray, spec: float | None) -> float:
        """Credit probability after the tuned rule: main classifier, optionally blended with the specialist."""
        pm = float(p[CLASSES.index(CREDIT)])
        w = float(self.credit_cfg.get("specialist_weight", 0.0))
        return pm if spec is None or w == 0 else (1 - w) * pm + w * spec

    @property
    def mode(self) -> str:
        return "model" if self.clf else ("zero-shot" if self.zs else "rules")

    def versions(self) -> dict[str, str]:
        zs_present = (self._zs_dir / "model.int8.onnx").exists()
        return {
            "event": self.clf.version if self.clf else "none",
            "event_zero_shot": "mdeberta-xnli-zs-1.0 (lazy)"
            if self.zero_shot and zs_present
            else "not downloaded",
        }

    # ---------------------------------------------------------------- pieces
    def _zero_shot(self, texts: list[str]) -> np.ndarray:
        """Single-label zero-shot: softmax of the entailment logit across all 12 hypotheses."""
        zs = self.zs
        assert zs is not None
        ent_i = [lbl.lower() for lbl in zs.labels].index("entailment")
        hyps = [f"This text is about {DESCRIPTIONS[EventClass(c)]}." for c in CLASSES]
        # NLI input is (premise, hypothesis); the classifier's "entity" slot carries the premise.
        prem = [t for t in texts for _ in hyps]
        hyp = [h for _ in texts for h in hyps]
        logits = zs.logits(prem, hyp, batch_size=48).reshape(len(texts), len(CLASSES), -1)
        return softmax(logits[:, :, ent_i], axis=-1)

    @staticmethod
    def _rules(text: str) -> np.ndarray:
        p = np.full(len(CLASSES), 0.02)
        hits = [c for c, rx in _RULES_RX if rx.search(text)]
        if hits:
            p[CLASSES.index(hits[0].value)] = 0.6
            for c in hits[1:3]:
                p[CLASSES.index(c.value)] = 0.25
        else:
            p[CLASSES.index(EventClass.OTHER.value)] = 0.6
        return p / p.sum()

    @staticmethod
    def _structured(meta: dict[str, Any]) -> str | None:
        # Source-aware: an 8-K with Item 1.03 / 2.04 / 3.01 is a credit event, whatever else it contains.
        if set(meta.get("items") or []) & EDGAR_CREDIT_ITEMS:
            return EventClass.CREDIT_EVENT.value
        weak = [e for e in meta.get("weak_events") or [] if e != EventClass.OTHER.value]
        if len(weak) == 1:
            return str(weak[0])
        ev = meta.get("gdelt_event") or {}
        if ev.get("cameo_root") in {"18", "19", "20"} and ev.get("goldstein", 0) <= -9:
            return EventClass.GEOPOLITICAL.value
        return None

    # ---------------------------------------------------------------- public
    def classify_batch(
        self, texts: list[str], metas: list[dict[str, Any]] | None = None, allow_zero_shot: bool = True
    ) -> list[EventResult]:
        if not texts:
            return []
        metas = metas or [{} for _ in texts]
        if self.clf is not None:
            probs = self.clf.predict_proba([""] * len(texts), texts)
            methods = ["model"] * len(texts)
        elif self.zero_shot and self.zs is not None:
            probs = self._zero_shot(texts)
            methods = ["zero-shot"] * len(texts)
        else:
            probs = np.stack([self._rules(t) for t in texts])
            methods = ["rules"] * len(texts)

        if self.clf is not None and allow_zero_shot and self.zero_shot and self.zs is not None:
            low = [i for i in range(len(texts)) if probs[i].max() < FALLBACK_THRESHOLD]
            if low:
                zs = self._zero_shot([texts[i] for i in low])
                for j, i in enumerate(low):
                    probs[i] = 0.5 * probs[i] + 0.5 * zs[j]  # blend: keep the fine-tuned prior
                    methods[i] = "model+zero-shot"

        spec = self._specialist(texts) if self.credit_cfg.get("specialist_weight") else None
        results = []
        for i, p in enumerate(probs):
            order = np.argsort(-p)
            primary = CLASSES[int(order[0])]
            conf = float(p[order[0]])
            method = methods[i]
            forced = self._structured(metas[i])
            credit_items = bool(set(metas[i].get("items") or []) & EDGAR_CREDIT_ITEMS)
            if forced and forced != primary and (conf < 0.8 or credit_items):
                primary, conf, method = forced, max(conf, 0.9 if credit_items else 0.8), "structured"
            elif primary != CREDIT and self.credit_cfg:
                pc = self._credit_prob(texts[i], p, spec[i] if spec is not None else None)
                if pc >= self.credit_cfg.get("threshold", 1.1) or (
                    self.credit_cfg.get("backstop") and credit_phrase(texts[i]) and conf < 0.9
                ):
                    primary, conf, method = CREDIT, max(pc, 0.6), method + "+credit"
            if self.credit_cfg.get("hard_negatives") and not credit_items:
                primary, conf, method = apply_hard_negative(texts[i], p, primary, conf, method)
            if self.credit_cfg.get("rules_v2") and not credit_items and method != "structured":
                to = credit_rule_v2(texts[i], primary)
                if to:
                    conf = max(float(p[CLASSES.index(to)]), 0.6) if to != CREDIT else max(conf, 0.7)
                    primary, method = to, method + "+rules_v2"
            secondary = [
                CLASSES[int(k)]
                for k in order[1:4]
                if p[k] >= SECONDARY_THRESHOLD and CLASSES[int(k)] != primary
            ]
            results.append(
                EventResult(
                    primary,
                    round(conf, 4),
                    secondary,
                    {c: round(float(v), 4) for c, v in zip(CLASSES, p, strict=True)},
                    method,
                )
            )
        return results
