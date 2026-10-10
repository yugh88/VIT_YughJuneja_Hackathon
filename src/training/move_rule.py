"""Validate the price-move rule (engine.nlp.quality) on VALIDATION data only.

    python -m training.move_rule      # -> docs/eval/move_rule.json

Rows: twitter-financial-news-topic validation split (never the frozen test). The rule applies to
headlines with a price-move verb that name a company: the class of a stated cause (event model on
the cause clause, if it is not Other), else Earnings when earnings words appear, else Other.
Kept only if macro-F1 on all validation rows does not drop by more than 0.005 and accuracy on the
rows it changes is not lower than before.
"""

from __future__ import annotations

import json
import logging

import numpy as np
from sklearn.metrics import f1_score

from engine.config import REPO_ROOT
from engine.nlp.entities import EntityLinker
from engine.nlp.events import EventModel
from engine.nlp.quality import price_move
from training.data import load_twitter_topic

log = logging.getLogger("move_rule")
OUT = REPO_ROOT / "docs" / "eval" / "move_rule.json"
OTHER = "Other/Non-event"
EARN = "Earnings/Guidance"


def apply_rule(texts: list[str], preds: list[str], em: EventModel, linker: EntityLinker) -> list[str]:
    """Only headlines the model left as Other are re-labelled (the rule never overrides a named event)."""
    out = list(preds)
    hits: list[tuple[int, str | None, bool]] = []
    for i, t in enumerate(texts):
        if preds[i] != OTHER:
            continue
        ms = [m for m in linker.link(t).mentions if m.entity.kind != "sovereign"]
        is_move, cause, earn = price_move(t, [(m.start, m.end) for m in ms])
        if is_move and ms:
            hits.append((i, cause, earn))
    with_cause = [(i, c) for i, c, _ in hits if c]
    cls = em.classify_batch([c for _, c in with_cause], allow_zero_shot=False) if with_cause else []
    by_i = {i: r for (i, _), r in zip(with_cause, cls, strict=True)}
    for i, _, earn in hits:
        r = by_i.get(i)
        if r is not None and r.primary != OTHER and r.confidence >= 0.5:
            out[i] = r.primary
        elif earn:
            out[i] = EARN
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    val = load_twitter_topic().query("split == 'val'").reset_index(drop=True)
    texts, gold = val["text"].tolist(), val["event"].tolist()
    em = EventModel(REPO_ROOT / "models" / "onnx", zero_shot=False, auto_download=False)
    linker = EntityLinker()
    base = [r.primary for r in em.classify_batch(texts, allow_zero_shot=False)]
    ruled = apply_rule(texts, base, em, linker)
    changed = [i for i in range(len(texts)) if base[i] != ruled[i]]
    touched = [i for i, t in enumerate(texts) if price_move(t)[0]]  # stock subject via shares/cashtag only
    labels = sorted(set(gold))

    def scores(pred: list[str], idx: list[int]) -> dict[str, float]:
        g = [gold[i] for i in idx]
        p = [pred[i] for i in idx]
        return {"n": len(idx), "accuracy": round(float(np.mean([a == b for a, b in zip(g, p, strict=True)])) if idx else 0.0, 4),
                "macro_f1": round(float(f1_score(g, p, average="macro", labels=labels, zero_division=0)), 4) if idx else 0.0}  # fmt: skip

    allidx = list(range(len(texts)))
    res = {
        "validation_rows": len(texts),
        "all": {"without": scores(base, allidx), "with": scores(ruled, allidx)},
        "price_move_rows": {"without": scores(base, touched), "with": scores(ruled, touched)},
        "changed_rows": {"without": scores(base, changed), "with": scores(ruled, changed)},
        "examples": [
            {"text": texts[i][:120], "gold": gold[i], "from": base[i], "to": ruled[i]} for i in changed[:30]
        ],
    }
    res["kept"] = bool(
        res["all"]["with"]["macro_f1"] >= res["all"]["without"]["macro_f1"] - 0.005
        and res["changed_rows"]["with"]["accuracy"] >= res["changed_rows"]["without"]["accuracy"]
    )
    OUT.write_text(json.dumps(res, indent=1))
    log.info(
        "move rule on validation: all %s | changed %s | kept %s", res["all"], res["changed_rows"], res["kept"]
    )


if __name__ == "__main__":
    main()
