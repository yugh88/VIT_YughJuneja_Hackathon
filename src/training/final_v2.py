"""Final, once-only scoring for the score-improvement round: old vs new system, v2 tests and old sets.

    python -m training.final_v2      # -> docs/eval/v2/final.json

"old" = the system before this round: original credit rule (threshold 0.55, no v2 rules), no NLI
sentiment blend, no Hinglish rule. "new" = the shipped system. Both are scored on the frozen v2
test sets (hash-checked) and on the seen sets (native-300, credit-150, the event topic test).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from engine.config import REPO_ROOT
from engine.nlp.events import CLASSES, CREDIT
from training.eval_v2 import SENT, Native, credit_metrics, load, native_metrics

log = logging.getLogger("final_v2")
OUT = REPO_ROOT / "docs" / "eval" / "v2" / "final.json"
EVAL = REPO_ROOT / "data" / "eval"


@contextmanager
def system(which: str) -> Iterator[Native]:
    import engine.nlp.pipeline as P

    orig = P.is_hinglish
    nat = Native()
    if which == "old":
        nat.pipe.sentiment._nli = None
        cfg = nat.pipe.events.credit_cfg
        cfg.pop("rules_v2", None)
        cfg["threshold"] = 0.55
        P.is_hinglish = lambda _t: False  # type: ignore[assignment]
    try:
        yield nat
    finally:
        P.is_hinglish = orig


def _native(nat: Native, d: pd.DataFrame) -> dict[str, Any]:
    ps, pe, _, _ = nat.probs(d["headline"].tolist())
    return native_metrics(d, [SENT[i] for i in ps.argmax(1)], [CLASSES[i] for i in pe.argmax(1)])


def _credit(nat: Native, d: pd.DataFrame) -> dict[str, Any]:
    rs = nat.pipe.events.classify_batch(d["headline"].tolist(), allow_zero_shot=False)
    m = credit_metrics(d, [r.primary for r in rs], [r.confidence for r in rs])
    return {k: v for k, v in m.items() if k != "errors"} | {"errors": m["errors"]}


def _topic(nat: Native) -> dict[str, Any]:
    from training.data import frozen_test_texts, load_twitter_topic

    t = load_twitter_topic()
    t = t[t["split"] == "test"]
    keep = set(frozen_test_texts())
    t = t[t["text"].isin(keep)] if keep else t
    rs = nat.pipe.events.classify_batch(t["text"].tolist(), allow_zero_shot=False)
    y, p = t["event"].to_numpy(), np.array([r.primary for r in rs])
    present = sorted(set(y))
    return {"n": len(t), "macro_f1_present": round(float(f1_score(y, p, average="macro", labels=present)), 4),
            "credit_f1": round(float(f1_score(y == CREDIT, p == CREDIT)), 4)}  # fmt: skip


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    native_v2, credit_v2 = load("native", "test"), load("credit", "test")  # frozen hash check
    native_old = pd.read_csv(EVAL / "multilingual_labelled.csv")
    credit_old = pd.read_csv(EVAL / "credit_labelled.csv")
    res: dict[str, Any] = {"run_at": datetime.now(UTC).isoformat(timespec="seconds"), "systems": {}}
    for which in ("old", "new"):
        with system(which) as nat:
            res["systems"][which] = {
                "credit_rule": dict(nat.pipe.events.credit_cfg),
                "native_v2_test": _native(nat, native_v2),
                "credit_v2_test": _credit(nat, credit_v2),
                "native_300_seen": _native(nat, native_old),
                "credit_150_seen": _credit(nat, credit_old),
                "event_topic_test_seen": _topic(nat),
            }
        log.info("%s done", which)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for k in (
        "native_v2_test",
        "credit_v2_test",
        "native_300_seen",
        "credit_150_seen",
        "event_topic_test_seen",
    ):
        for which in ("old", "new"):
            m = res["systems"][which][k]
            show = {x: m[x] for x in ("sentiment_macro_f1", "sentiment_macro_f1_ci95", "event_macro_f1", "event_macro_f1_ci95",
                                      "precision", "precision_ci95", "recall", "recall_ci95", "f1", "f1_ci95", "macro_f1_present",
                                      "credit_f1") if x in m}  # fmt: skip
            print(f"{k:22} {which:4} {show}")


if __name__ == "__main__":
    main()
