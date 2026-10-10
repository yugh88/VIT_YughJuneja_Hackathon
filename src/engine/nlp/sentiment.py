"""Stage 5 - entity-level sentiment in [-1, 1].

Routing: English text goes to the fine-tuned FinBERT; everything else goes to the
fine-tuned XLM-R. When language ID is unsure (< 0.5) both run and their
probabilities are averaged, weighted by each model's validation macro-F1.

raw = P(positive) - P(negative) from the untempered softmax, mapped to the published
score by a monotone power curve chosen on FiQA plus a range check
(``training.calibrate_sentiment``); confidence = max class probability after
temperature scaling. If no fine-tuned model is present the stage falls back to a
small finance lexicon and every signal is marked ``lexicon-fallback``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from engine.nlp.onnx_model import OnnxClassifier, softmax

log = logging.getLogger(__name__)
LABELS = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class SentimentResult:
    score: float
    label: str
    confidence: float
    probs: tuple[float, float, float]
    model: str
    score_human: float | None = None  # FiQA-faithful scale (see training.calibrate_sentiment)


# Intensity cues, declared up front (not fitted to any check set): words that mark *how strong*
# financial news is, plus large percentage moves. Their weight is fitted on FiQA only.
INTENSIFIERS = re.compile(
    r"\b(record[- ](\w+[- ])?(high|low|profit|loss|revenue|sales|earnings|quarter|year|levels?|results)|all-time|soar\w*|surg\w*|skyrocket\w*|rocket\w*|plung\w*|plummet\w*|crash\w*|collaps\w*|"
    r"tumbl\w*|slump\w*|sink\w*|sank|crush\w*|smash\w*|blowout|stellar|landmark|massive|huge|shock\w*|dire|"
    r"default\w*|bankrupt\w*|fraud\w*|junk|wip(e|es|ed) out|halt\w*|devastat\w*|best|worst|double[sd]?|"
    r"triple[sd]?|slash\w*|(\d+|five|ten)[- ]year (low|high)|decade|biggest|largest|disaster\w*|boom\w*)\b"
    r"|रिकॉर्ड|धड़ाम|उछाल|लुढ़क|ज़ोरदार|भारी|急落|急騰|暴落|過去最高|最高益|暴跌|大涨|大跌|创纪录|"
    r"\b(rekord\w*|einbruch|stürz\w*|absturz|effondr\w*|envol\w*|desplom\w*|dispar\w*|despenc\w*|recorde)\b",
    re.I,
)
_PCT = re.compile(r"(\d+(?:[.,]\d+)?)\s?(%|percent|per cent|प्रतिशत)")


def intensity(text: str) -> float:
    """0..1: intensifier words (each 0.5) plus a move of >= 10% (0.5)."""
    hits = len(INTENSIFIERS.findall(text))
    big_move = any(float(m.group(1).replace(",", ".")) >= 10 for m in _PCT.finditer(text))
    return min(1.0, 0.5 * hits + (0.5 if big_move else 0.0))


def raw_of(logits: np.ndarray) -> np.ndarray:
    """P(positive) - P(negative) from the untempered softmax."""
    p = softmax(logits)
    return p[:, 2] - p[:, 0]


class LexiconSentiment:
    """Last-resort fallback: a few hundred finance polarity words (English stems)."""

    POS = re.compile(
        r"\b(beat|beats|surge\w*|soar\w*|jump\w*|rall\w*|gain\w*|record high|upgrade\w*|profit\w*|"
        r"growth|strong|raise[sd]? guidance|outperform\w*|rebound\w*|approv\w*|win[s]?|boost\w*)\b",
        re.I,
    )
    NEG = re.compile(
        r"\b(miss(es|ed)?|plunge\w*|slump\w*|fall\w*|fell|drop\w*|downgrade\w*|loss(es)?|default\w*|"
        r"bankrupt\w*|fraud|probe|lawsuit|cut\w*|warn\w*|weak\w*|sanction\w*|war|recall\w*|layoff\w*|crash\w*)\b",
        re.I,
    )
    version = "lexicon-fallback"

    def score(self, text: str) -> SentimentResult:
        p, n = len(self.POS.findall(text)), len(self.NEG.findall(text))
        s = 0.0 if p + n == 0 else (p - n) / (p + n)
        s = float(np.clip(s * 0.7, -1, 1))
        label = "positive" if s > 0.15 else "negative" if s < -0.15 else "neutral"
        probs = (max(-s, 0.0), 1 - abs(s), max(s, 0.0))
        return SentimentResult(s, label, 0.5, probs, self.version, s)


NLI_WEIGHT = 0.5  # non-English: blend with zero-shot NLI sentiment, chosen on native_v2_dev (DECISIONS D78)
NLI_HYPS = ["This is bad news.", "This is neither good nor bad news.", "This is good news."]  # LABELS order


class SentimentModel:
    def __init__(self, onnx_dir: Path, nli: Callable[[], OnnxClassifier | None] | None = None) -> None:
        self.en = self._load(onnx_dir / "sentiment-en")
        self.multi = self._load(onnx_dir / "sentiment-multi")
        self._nli = nli  # shared with the event model's zero-shot fallback; None disables the blend
        self.lexicon = LexiconSentiment()
        if not (self.en or self.multi):
            log.warning("no fine-tuned sentiment model in %s: lexicon fallback", onnx_dir)

    @staticmethod
    def _load(path: Path) -> OnnxClassifier | None:
        return OnnxClassifier(path) if (path / "model.int8.onnx").exists() else None

    @property
    def mode(self) -> str:
        return "model" if (self.en or self.multi) else "lexicon"

    def versions(self) -> dict[str, str]:
        return {
            "sentiment_en": self.en.version if self.en else "none",
            "sentiment_multi": self.multi.version if self.multi else "none",
        }

    def _nli_probs(self, texts: list[str]) -> np.ndarray | None:
        zs = self._nli() if self._nli else None
        if zs is None:
            return None
        ent = [lbl.lower() for lbl in zs.labels].index("entailment")
        prem = [t for t in texts for _ in NLI_HYPS]
        hyp = [h for _ in texts for h in NLI_HYPS]
        lg = zs.logits(prem, hyp, batch_size=48).reshape(len(texts), len(NLI_HYPS), -1)[:, :, ent]
        return softmax(lg, axis=-1)

    def score_batch(self, items: list[tuple[str, str, str, float]]) -> list[SentimentResult]:
        """items: (entity name, text, language, language confidence)."""
        if not items:
            return []
        if self.mode == "lexicon":
            return [self.lexicon.score(t) for _, t, _, _ in items]
        out: list[SentimentResult | None] = [None] * len(items)
        routes: dict[str, list[int]] = {"en": [], "multi": [], "both": []}
        for i, (_, _, lang, conf) in enumerate(items):
            if conf < 0.5 and self.en and self.multi:
                routes["both"].append(i)
            elif lang == "en" and self.en:
                routes["en"].append(i)
            else:
                routes["multi"].append(i)
        for route, idx in routes.items():
            if not idx:
                continue
            ents = [items[i][0] for i in idx]
            texts = [items[i][1] for i in idx]
            if route == "both":
                assert self.en and self.multi
                we = float(self.en.cfg.get("best_val_macro_f1", 0.5))
                wm = float(self.multi.cfg.get("best_val_macro_f1", 0.5))
                le, lm = self.en.logits(ents, texts), self.multi.logits(ents, texts)
                p = (we * softmax(le / self.en.temperature) + wm * softmax(lm / self.multi.temperature)) / (
                    we + wm
                )
                inten = np.array([intensity(t) for t in texts])
                cal, hum = (
                    (
                        we * self.en.calibrate(raw_of(le), inten, w)
                        + wm * self.multi.calibrate(raw_of(lm), inten, w)
                    )
                    / (we + wm)
                    for w in ("range", "human")
                )
                name = f"{self.en.version}+{self.multi.version}"
            else:
                model = (self.en if route == "en" else self.multi) or self.multi or self.en
                assert model is not None
                logits = model.logits(ents, texts)
                p = softmax(logits / model.temperature)  # confidence: temperature-scaled
                # score: untempered P(pos) - P(neg) and the text's intensity cues, mapped
                raw, inten = raw_of(logits), np.array([intensity(t) for t in texts])
                name = model.version
                if route == "multi" and model is self.multi:
                    pn = self._nli_probs(texts)
                    if pn is not None:  # D78: war, attacks and disasters read as bad news
                        p = NLI_WEIGHT * pn + (1 - NLI_WEIGHT) * p
                        raw = NLI_WEIGHT * (pn[:, 2] - pn[:, 0]) + (1 - NLI_WEIGHT) * raw
                        name = f"{model.version}+nli"
                cal = model.calibrate(raw, inten, "range")
                hum = model.calibrate(raw, inten, "human")
            for j, i in enumerate(idx):
                k = int(p[j].argmax())
                out[i] = SentimentResult(
                    round(float(cal[j]), 4),
                    LABELS[k],
                    round(float(p[j].max()), 4),
                    (float(p[j, 0]), float(p[j, 1]), float(p[j, 2])),
                    name,
                    round(float(hum[j]), 4),
                )
        return [r for r in out if r is not None]
