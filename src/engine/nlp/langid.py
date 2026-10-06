"""Stage 2 - language identification with fastText ``lid.176`` (176 languages, <1 MB).

If the model file is missing the stage degrades to a Unicode-script heuristic
plus the source's language hint, and says so in its result.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from engine.config import REPO_ROOT

log = logging.getLogger(__name__)
MODEL_PATH = REPO_ROOT / "models" / "lid.176.ftz"
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
MULTILINGUAL_THRESHOLD = 0.5  # plan: docs under 0.5 confidence go to the multilingual path

_SCRIPT_LANG = [  # (unicode name prefix, language) for the fallback heuristic
    ("HIRAGANA", "ja"), ("KATAKANA", "ja"), ("HANGUL", "ko"), ("CJK", "zh"), ("DEVANAGARI", "hi"),
    ("ARABIC", "ar"), ("CYRILLIC", "ru"), ("HEBREW", "he"), ("THAI", "th"), ("GREEK", "el"),
    ("BENGALI", "bn"), ("TAMIL", "ta"),
]  # fmt: skip


@dataclass(frozen=True)
class LangResult:
    lang: str
    confidence: float
    method: str  # "fasttext" | "script" | "hint"


class LanguageIdentifier:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self._model = None
        if model_path.exists():
            try:
                import fasttext

                fasttext.FastText.eprint = lambda *a, **k: None  # silence load warning
                self._model = fasttext.load_model(str(model_path))
            except Exception as exc:  # pragma: no cover - depends on platform wheels
                log.warning("fastText unavailable (%s); using script heuristic", exc)
        else:
            log.warning("%s missing (run `make models`); using script heuristic", model_path)

    @property
    def available(self) -> bool:
        return self._model is not None

    def detect(self, text: str, hint: str | None = None) -> LangResult:
        text = text.replace("\n", " ").strip()
        if not text:
            return LangResult(hint or "und", 0.0, "hint")
        if self._model is not None:
            labels, probs = self._model.predict(text[:500], k=1)
            lang = labels[0].replace("__label__", "")
            conf = float(probs[0])
            # Short headlines are hard; a confident source hint wins over a weak guess.
            if hint and conf < MULTILINGUAL_THRESHOLD and hint != lang:
                return LangResult(hint, conf, "hint")
            return LangResult(lang, min(conf, 1.0), "fasttext")
        return self._script_guess(text, hint)

    @staticmethod
    def _script_guess(text: str, hint: str | None) -> LangResult:
        counts: dict[str, int] = {}
        letters = 0
        for ch in text:
            if not ch.isalpha():
                continue
            letters += 1
            name = unicodedata.name(ch, "")
            for prefix, lang in _SCRIPT_LANG:
                if name.startswith(prefix):
                    counts[lang] = counts.get(lang, 0) + 1
                    break
        if counts:
            if counts.get("ja"):
                return LangResult("ja", 0.6, "script")
            lang, n = max(counts.items(), key=lambda kv: kv[1])
            if n / max(letters, 1) > 0.3:
                return LangResult(lang, 0.6, "script")
        return LangResult(hint or "en", 0.4, "script")
