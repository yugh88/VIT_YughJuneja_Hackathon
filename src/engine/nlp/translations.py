"""English display titles for recorded non-English headlines (made once by ``training.translate_titles``).

A dictionary lookup only: no translation model runs at serve time. Scoring always uses the
original text; this is just so the dashboard can show a reader an English headline.
"""

from __future__ import annotations

import gzip
import json
import logging
from functools import lru_cache

from engine.config import REPO_ROOT

log = logging.getLogger(__name__)
PATH = REPO_ROOT / "data" / "reference" / "title_translations.json.gz"


@lru_cache(maxsize=1)
def _table() -> dict[str, str]:
    if not PATH.exists():
        log.info("%s missing; titles are shown in their original language", PATH.name)
        return {}
    with gzip.open(PATH, "rt") as fh:
        return json.loads(fh.read())


def english_title(title: str) -> str:
    """The English version of a recorded headline, or "" when there is none (English or unseen)."""
    return _table().get(title.strip(), "")
