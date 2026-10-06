"""Step C of the credit work: choose the Credit decision rule on VALIDATION only, then test once.

    python -m training.credit_tune tune     # validation only; writes the chosen rule into event-multi
    python -m training.credit_tune test     # run ONCE at the end: frozen test + EDGAR >= 2025 test

Candidates (kept only if they help on validation):
  1. per-class threshold: predict Credit when P(credit) >= t, even if another class is larger
  3. specialist one-vs-rest head (char n-gram logistic regression, multilingual) blended with the main model
  +  phrase backstop (engine.nlp.credit) for unambiguous phrases when the main model is not confident
  5. EDGAR item override (always on; it is source structure, not a tuned parameter)
Objective: Credit F1 subject to Credit precision >= 0.80 and overall macro-F1 not falling more than
0.005 below the plain classifier. Validation rows labelled by the backstop's own phrase rules
(GDELT multi-rule) and templated EDGAR rows are excluded from the objective.
Every configuration tried is written to docs/eval/credit_experiments.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline

from engine.config import REPO_ROOT
from engine.nlp.credit import credit_phrase
from engine.nlp.onnx_model import OnnxClassifier, softmax
from training.data import EVENT_LABELS, frozen_test_texts, load_twitter_topic
from training.train_events import build_frames

log = logging.getLogger("credit_tune")
MODEL_DIR = REPO_ROOT / "models" / "onnx" / "event-multi"
OUT = REPO_ROOT / "docs" / "eval" / "credit_experiments.json"
FINAL = REPO_ROOT / "docs" / "eval" / "credit.json"
CREDIT = "Credit Event"
CI = EVENT_LABELS.index(CREDIT)
# Not used in the objective: GDELT credit rows are labelled by the backstop's own phrase rules, and
# EDGAR rows are templated 8-K text (handled live by the item override); both would flatter the score.
EXCLUDE_FROM_OBJECTIVE = {"gdelt_credit", "edgar_8k"}


def train_specialist(train: pd.DataFrame) -> Any:
    clf = make_pipeline(
        TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=200_000, sublinear_tf=True
        ),
        LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000),
    )
    return clf.fit(train["text"].tolist(), (train["event"] == CREDIT).astype(int))


def decide(p: np.ndarray, texts: list[str], spec: np.ndarray | None, rule: dict[str, Any]) -> np.ndarray:
    """Mirror of engine.nlp.events.EventModel's credit rule (without the EDGAR override)."""
    pred = p.argmax(1)
    if not rule:
        return pred
    w = rule.get("specialist_weight", 0.0)
    pc = p[:, CI] if spec is None or w == 0 else (1 - w) * p[:, CI] + w * spec
    conf = p.max(1)
    for i in range(len(pred)):
        if pred[i] == CI:
            continue
        if pc[i] >= rule.get("threshold", 1.1) or (
            rule.get("backstop") and conf[i] < 0.9 and credit_phrase(texts[i])
        ):
            pred[i] = CI
    if rule.get("hard_negatives"):
        from engine.nlp.events import apply_hard_negative

        for i in range(len(pred)):
            lab, _, _ = apply_hard_negative(texts[i], p[i], EVENT_LABELS[pred[i]], 0.0, "")
            pred[i] = EVENT_LABELS.index(lab)
    return pred


def hardneg() -> dict[str, Any]:
    """Validate the generic hard-negative rules (affirmed / stable-positive outlook / upgrade -> not
    Credit; AML -> Regulatory/Legal) on VALIDATION only, on top of the already chosen rule."""
    _, val = build_frames()
    m = OnnxClassifier(MODEL_DIR)
    cfg_path = MODEL_DIR / "sentinel.json"
    cfg = json.loads(cfg_path.read_text())
    rule = {k: v for k, v in (cfg.get("credit_rule") or {}).items() if k != "hard_negatives"}
    spec_file = MODEL_DIR / "credit_specialist.joblib"
    spec_model = joblib.load(spec_file) if spec_file.exists() else None
    res: dict[str, Any] = {"rule_base": rule}
    for name, v in (
        ("objective_rows", val[~val["source"].isin(EXCLUDE_FROM_OBJECTIVE)]),
        ("all_validation_rows", val),
    ):
        v = v.reset_index(drop=True)
        texts, y = v["text"].tolist(), v["y"].to_numpy()
        p = softmax(m.logits([""] * len(v), texts) / m.temperature)
        spec = spec_model.predict_proba(texts)[:, 1] if spec_model is not None else None
        base = decide(p, texts, spec, rule)
        hn = decide(p, texts, spec, {**rule, "hard_negatives": True})
        res[name] = {"n": len(v), "rows_changed": int((base != hn).sum()),
                     "without": {"macro_f1": macro(y, base), "credit": credit_metrics(y, base, 500)},
                     "with": {"macro_f1": macro(y, hn), "credit": credit_metrics(y, hn, 500)},
                     "changed_examples": [{"text": texts[i][:140], "gold": EVENT_LABELS[y[i]], "from": EVENT_LABELS[base[i]],
                                           "to": EVENT_LABELS[hn[i]]} for i in np.flatnonzero(base != hn)[:25]]}  # fmt: skip
    o = res["objective_rows"]
    keep = (o["with"]["credit"]["f1"] >= o["without"]["credit"]["f1"]
            and o["with"]["macro_f1"] >= o["without"]["macro_f1"] - 0.005)  # fmt: skip
    res["kept"] = bool(keep)
    cfg["credit_rule"] = {**rule, "hard_negatives": bool(keep)}
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    ex = json.loads(OUT.read_text()) if OUT.exists() else {}
    ex["hard_negative_rules"] = res
    OUT.write_text(json.dumps(ex, separators=(",", ":")))
    log.info(
        "hard negatives on validation: %s",
        {k: (res[k]["without"], res[k]["with"]) for k in ("objective_rows",)},
    )
    return res


def credit_metrics(y: np.ndarray, pred: np.ndarray, n_boot: int = 2000) -> dict[str, Any]:
    p, r, f, _ = precision_recall_fscore_support(y == CI, pred == CI, average="binary", zero_division=0)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        boots.append(f1_score(y[i] == CI, pred[i] == CI, zero_division=0))
    return {
        "n_credit": int((y == CI).sum()),
        "predicted_credit": int((pred == CI).sum()),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f), 4),
        "f1_ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
    }


def macro(y: np.ndarray, pred: np.ndarray) -> float:
    return round(float(f1_score(y, pred, average="macro", labels=sorted(set(y)))), 4)


def compare(names: list[str]) -> dict[str, Any]:
    """Plain argmax of candidate models on the same validation rows (rule-labelled rows excluded)."""
    _, val = build_frames()
    v = val[~val["source"].isin(EXCLUDE_FROM_OBJECTIVE)].reset_index(drop=True)
    y = v["y"].to_numpy()
    out = {}
    for name in names:
        m = OnnxClassifier(MODEL_DIR.parent / name)
        pred = softmax(m.logits([""] * len(v), v["text"].tolist())).argmax(1)
        out[name] = {"macro_f1": macro(y, pred), "credit": credit_metrics(y, pred, 500)}
        log.info("%s on validation: macro %.4f credit %s", name, out[name]["macro_f1"], out[name]["credit"])
    (OUT.parent / "credit_model_comparison.json").write_text(json.dumps(out, indent=2))
    return out


def tune() -> dict[str, Any]:
    train, val = build_frames()
    m = OnnxClassifier(MODEL_DIR)
    val_obj = val[~val["source"].isin(EXCLUDE_FROM_OBJECTIVE)].reset_index(drop=True)
    p = softmax(m.logits([""] * len(val_obj), val_obj["text"].tolist()) / m.temperature)
    y = val_obj["y"].to_numpy()
    texts = val_obj["text"].tolist()
    spec_model = train_specialist(train)
    spec = spec_model.predict_proba(texts)[:, 1]
    base_pred = decide(p, texts, None, {})
    base_macro = macro(y, base_pred)
    experiments = [{"rule": {}, "macro_f1": base_macro, "credit": credit_metrics(y, base_pred, 200)}]
    for t in (0.15, 0.25, 0.35, 0.45, 0.55):
        for w in (0.0, 0.3, 0.5, 0.7):
            for backstop in (False, True):
                rule = {"threshold": t, "specialist_weight": w, "backstop": backstop}
                pred = decide(p, texts, spec, rule)
                experiments.append(
                    {"rule": rule, "macro_f1": macro(y, pred), "credit": credit_metrics(y, pred, 200)}
                )
    ok = [e for e in experiments if e["credit"]["precision"] >= 0.80 and e["macro_f1"] >= base_macro - 0.005]
    best = max(ok or experiments[:1], key=lambda e: (e["credit"]["f1"], e["macro_f1"]))
    log.info("baseline val: macro %.4f credit %s", base_macro, experiments[0]["credit"])
    log.info("chosen on val: %s macro %.4f credit %s", best["rule"], best["macro_f1"], best["credit"])
    cfg_path = MODEL_DIR / "sentinel.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["credit_rule"] = best["rule"]
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    spec_file = MODEL_DIR / "credit_specialist.joblib"
    if best["rule"].get("specialist_weight"):
        joblib.dump(spec_model, spec_file, compress=3)
    elif spec_file.exists():
        spec_file.unlink()
    out = {
        "validation": {
            "n": len(val_obj),
            "n_credit": int((y == CI).sum()),
            "excluded_rows": int(val["source"].isin(EXCLUDE_FROM_OBJECTIVE).sum()),
            "by_source": val_obj["source"].value_counts().to_dict(),
        },
        "baseline": experiments[0],
        "chosen": best,
        "tried": experiments,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))  # compact: every configuration tried
    return out


def test_once() -> dict[str, Any]:
    """Frozen event test + EDGAR filings >= 2025 (never used for training or tuning)."""
    from engine.nlp.events import EventModel

    em = EventModel(MODEL_DIR.parent, zero_shot=False, auto_download=False)
    res: dict[str, Any] = {"rule": em.credit_cfg}
    tt = load_twitter_topic().query("split == 'test'")
    assert set(tt["text"]) >= set(frozen_test_texts())
    y = tt["event"].map(EVENT_LABELS.index).to_numpy()
    pred = np.array(
        [EVENT_LABELS.index(r.primary) for r in em.classify_batch(tt["text"].tolist(), allow_zero_shot=False)]
    )
    m = OnnxClassifier(MODEL_DIR)
    plain = softmax(m.logits([""] * len(tt), tt["text"].tolist())).argmax(1)
    res["frozen_test"] = {
        "system": {"macro_f1": macro(y, pred), "credit": credit_metrics(y, pred)},
        "model_only_argmax": {"macro_f1": macro(y, plain), "credit": credit_metrics(y, plain)},
    }
    ed = pd.read_csv(REPO_ROOT / "data" / "eval" / "edgar_credit_test.csv.gz")
    ye = ed["event"].map(EVENT_LABELS.index).to_numpy()
    metas = [{"items": str(i).split(";")} for i in ed["items"]]
    with_meta = np.array(
        [EVENT_LABELS.index(r.primary) for r in em.classify_batch(ed["text"].tolist(), metas, False)]
    )
    text_only = np.array(
        [EVENT_LABELS.index(r.primary) for r in em.classify_batch(ed["text"].tolist(), None, False)]
    )
    res["edgar_2025_test"] = {
        "n": len(ed),
        "system_with_item_override": credit_metrics(ye, with_meta),
        "text_model_only": credit_metrics(ye, text_only),
    }
    # Precision at the confidence Module B uses: Credit predictions with confidence >= 0.8.
    rs = em.classify_batch(tt["text"].tolist(), allow_zero_shot=False)
    hi = np.array([r.primary == CREDIT and r.confidence >= 0.8 for r in rs])
    res["frozen_test"]["precision_at_trigger_confidence_0.8"] = {
        "n": int(hi.sum()),
        "precision": round(float((y[hi] == CI).mean()), 4) if hi.any() else None,
    }
    FINAL.write_text(json.dumps(res, indent=2))
    return res


def test_annotated() -> dict[str, Any]:
    """Run ONCE: the AI-annotated credit set (data/eval/credit_labelled.csv; 2024+ headlines never used
    in training or tuning). Reports the shipped system and, from the same run, the plain argmax."""
    from engine.nlp.credit import distress_term
    from engine.nlp.events import EventModel

    d = pd.read_csv(REPO_ROOT / "data" / "eval" / "credit_labelled.csv")
    texts = d["headline"].tolist()
    y = d["event_label"].map(EVENT_LABELS.index).to_numpy()
    em = EventModel(MODEL_DIR.parent, zero_shot=False, auto_download=False)
    rs = em.classify_batch(texts, allow_zero_shot=False)
    pred = np.array([EVENT_LABELS.index(r.primary) for r in rs])
    m = OnnxClassifier(MODEL_DIR)
    plain = softmax(m.logits([""] * len(d), texts)).argmax(1)
    gate = np.array(
        [
            r.primary == CREDIT and r.confidence >= 0.7 and bool(distress_term(t))
            for r, t in zip(rs, texts, strict=True)
        ]
    )
    res = {"set": "credit_labelled.csv (AI-annotated, blind, two-pass)", "n": len(d), "rule": em.credit_cfg,
           "system": {"macro_f1": macro(y, pred), "credit": credit_metrics(y, pred)},
           "model_only_argmax": {"macro_f1": macro(y, plain), "credit": credit_metrics(y, plain)},
           "module_b_credit_gate": {"rule": "Credit, confidence >= 0.7, distress term", "n": int(gate.sum()),
                                    "precision": round(float((y[gate] == CI).mean()), 4) if gate.any() else None,
                                    "recall": round(float((gate & (y == CI)).sum() / max((y == CI).sum(), 1)), 4)},
           "errors": [{"id": i, "text": t[:120], "gold": EVENT_LABELS[a], "pred": EVENT_LABELS[b]}
                      for i, t, a, b in zip(d["id"], texts, y, pred, strict=True) if (a == CI) != (b == CI)]}  # fmt: skip
    full = json.loads(FINAL.read_text()) if FINAL.exists() else {}
    full["annotated_credit_test"] = res
    FINAL.write_text(json.dumps(full, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["compare", "tune", "hardneg", "test", "test-annotated"])
    ap.add_argument("--models", default="event-multi,event-multi-credit")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    if args.what == "compare":
        out = compare(args.models.split(","))
    else:
        out = {"tune": tune, "hardneg": hardneg, "test": test_once, "test-annotated": test_annotated}[
            args.what
        ]()
    print(json.dumps({k: v for k, v in out.items() if k != "tried"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
