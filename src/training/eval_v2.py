"""Score engine variants on the v2 sets (score-improvement round, DECISIONS E13, D73-D76).

    python -m training.eval_v2 translate --split dev           # cache NLLB X->English translations
    python -m training.eval_v2 native --split dev [--w-sent 0.5 --w-event 0.5]
    python -m training.eval_v2 credit --split dev
    python -m training.eval_v2 native --split test             # once, at the end (frozen file check)
    python -m training.eval_v2 old                             # the same scorers on the seen sets

Native: sentiment macro-F1 (3 classes) and event macro-F1 over the gold classes present, with
bootstrap 95% CIs and per-language scores (per language also over the classes present there).
Credit: Credit Event precision / recall / F1 with CIs, and Module B's credit gate.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from engine.config import REPO_ROOT
from engine.nlp.events import CLASSES, CREDIT

log = logging.getLogger("eval_v2")
V2 = REPO_ROOT / "data" / "eval" / "v2"
OUT = REPO_ROOT / "docs" / "eval" / "v2"
SENT = ["negative", "neutral", "positive"]
NLLB_SRC = {"hi": "hin_Deva", "zh": "zho_Hans", "ja": "jpn_Jpan", "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn",
            "ar": "arb_Arab", "pt": "por_Latn", "ru": "rus_Cyrl", "it": "ita_Latn"}  # fmt: skip


def load(name: str, split: str) -> pd.DataFrame:
    f = f"{name}_v2_{split}_labelled.csv"
    if split == "test":
        from training.v2_sets import check_frozen

        check_frozen(f)
    d = pd.read_csv(V2 / f)
    return d.rename(columns={"text": "headline"})


def _boot(y: np.ndarray, p: np.ndarray, labels: list[str] | None, n: int = 1000) -> list[float]:
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(f1_score(y[i], p[i], average="macro", labels=labels, zero_division=0))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def native_metrics(d: pd.DataFrame, sent: list[str], ev: list[str]) -> dict[str, Any]:
    y_s, y_e, p_s, p_e = (
        d["sentiment_label"].to_numpy(),
        d["event_label"].to_numpy(),
        np.array(sent),
        np.array(ev),
    )
    present = sorted(set(y_e))
    langs = {}
    for lang, idx in d.groupby("lang").groups.items():
        i = d.index.get_indexer(idx)
        pl = sorted(set(y_e[i]))
        langs[str(lang)] = {"n": len(i),
                            "sentiment_macro_f1": round(float(f1_score(y_s[i], p_s[i], average="macro", labels=SENT, zero_division=0)), 4),
                            "event_macro_f1": round(float(f1_score(y_e[i], p_e[i], average="macro", labels=pl, zero_division=0)), 4)}  # fmt: skip
    return {
        "n": len(d),
        "sentiment_macro_f1": round(
            float(f1_score(y_s, p_s, average="macro", labels=SENT, zero_division=0)), 4
        ),
        "sentiment_macro_f1_ci95": _boot(y_s, p_s, SENT),
        "sentiment_accuracy": round(float((y_s == p_s).mean()), 4),
        "event_macro_f1": round(
            float(f1_score(y_e, p_e, average="macro", labels=present, zero_division=0)), 4
        ),
        "event_macro_f1_ci95": _boot(y_e, p_e, present),
        "event_accuracy": round(float((y_e == p_e).mean()), 4),
        "per_language": langs,
    }


def _prf(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    tp, fp, fn = (
        int(((y == 1) & (p == 1)).sum()),
        int(((y == 0) & (p == 1)).sum()),
        int(((y == 1) & (p == 0)).sum()),
    )
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "precision": pr,
        "recall": rc,
        "f1": 2 * pr * rc / max(pr + rc, 1e-9),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def credit_metrics(d: pd.DataFrame, ev: list[str], conf: list[float]) -> dict[str, Any]:
    from engine.nlp.credit import distress_term

    y = (d["event_label"] == CREDIT).to_numpy().astype(int)
    p = (np.array(ev) == CREDIT).astype(int)
    m = _prf(y, p)
    rng = np.random.default_rng(0)
    boots = {"precision": [], "recall": [], "f1": []}
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        b = _prf(y[i], p[i])
        for k in boots:
            boots[k].append(b[k])
    gate = np.array(
        [
            e == CREDIT and c >= 0.7 and bool(distress_term(t))
            for e, c, t in zip(ev, conf, d["headline"], strict=True)
        ]
    )
    g = _prf(y, gate.astype(int))
    present = sorted(set(d["event_label"]))
    errors = [{"id": i, "text": t[:110], "gold": a, "pred": b} for i, t, a, b in zip(d["id"], d["headline"], d["event_label"], ev, strict=True)
              if (a == CREDIT) != (b == CREDIT)]  # fmt: skip
    return {
        "n": len(d), "n_credit": int(y.sum()),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
        **{f"{k}_ci95": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in boots.items()},
        "macro_f1_all_classes": round(float(f1_score(d["event_label"], ev, average="macro", labels=present, zero_division=0)), 4),
        "module_b_gate": {"precision": round(g["precision"], 4), "recall": round(g["recall"], 4), "n": int(gate.sum())},
        "errors": errors,
    }  # fmt: skip


# ------------------------------------------------------------------ scorers
class Native:
    """The shipped pipeline (as in training.evaluate): per-headline sentiment probs and event probs."""

    def __init__(self) -> None:
        from engine.nlp.pipeline import Pipeline

        self.pipe = Pipeline(zero_shot=False, dedupe=False)

    def probs(
        self, texts: list[str], lang: str | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[float]]:
        ps, pe, ev, conf = [], [], [], []
        for t in texts:
            tr = self.pipe.analyze(t, language=lang)
            s = next(iter(tr.sentiments.values()), None)
            if s is None:
                s = self.pipe.sentiment.score_batch([("", t, tr.lang.lang, tr.lang.confidence)])[0]
            e = tr.event or self.pipe.events.classify_batch([t])[0]
            ps.append(np.array(s.probs))
            pe.append(np.array([e.probs[c] for c in CLASSES]))
            ev.append(e.primary)
            conf.append(e.confidence)
        return np.stack(ps), np.stack(pe), ev, conf


NLI_HYPS = ["This is bad news.", "This is neither good nor bad news.", "This is good news."]  # SENT order


def nli_sentiment(texts: list[str]) -> np.ndarray:
    """D78: zero-shot sentiment from the shipped multilingual NLI model (softmax of entailment logits)."""
    from engine.nlp.onnx_model import OnnxClassifier, softmax
    from engine.nlp.pipeline import ONNX_DIR

    zs = OnnxClassifier(ONNX_DIR / "zeroshot-nli")
    ent = [lbl.lower() for lbl in zs.labels].index("entailment")
    prem = [t for t in texts for _ in NLI_HYPS]
    hyp = [h for _ in texts for h in NLI_HYPS]
    lg = zs.logits(prem, hyp, batch_size=48).reshape(len(texts), len(NLI_HYPS), -1)[:, :, ent]
    return softmax(lg, axis=-1)


def run_nli(split: str, w_event: float = 0.25, v: float | None = None) -> dict[str, Any]:
    d = load("native", split)
    nat = Native()
    ps_n, pe_n, _, _ = nat.probs(d["headline"].tolist())
    mt = translations(split).set_index("id").loc[d["id"], "en_mt"].tolist()
    _, pe_t, _, _ = nat.probs(mt, lang="en")
    pe = w_event * pe_t + (1 - w_event) * pe_n
    ev = [CLASSES[i] for i in pe.argmax(1)]
    pn = nli_sentiment(d["headline"].tolist())
    res: dict[str, Any] = {"split": split, "w_event": w_event, "grid": {}}
    for vv in [v] if v is not None else [0.0, 0.25, 0.5, 0.75, 1.0]:
        ps = vv * pn + (1 - vv) * ps_n
        res["grid"][str(vv)] = native_metrics(d, [SENT[i] for i in ps.argmax(1)], ev)
    return res


def to_english(texts: list[str], langs: list[str]) -> list[str]:
    import torch

    from training.translate import Translator

    tr = Translator()
    out = [""] * len(texts)
    eng = tr.tok.convert_tokens_to_ids("eng_Latn")
    by_lang: dict[str, list[int]] = {}
    for i, lg in enumerate(langs):
        by_lang.setdefault(lg, []).append(i)
    with torch.inference_mode():
        for lg, idx in by_lang.items():
            if lg not in NLLB_SRC:
                for i in idx:
                    out[i] = texts[i]
                continue
            tr.tok.src_lang = NLLB_SRC[lg]
            for b in range(0, len(idx), 16):
                chunk = idx[b : b + 16]
                enc = tr.tok(
                    [texts[i] for i in chunk],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                )
                enc = {k: v.to(tr.dev) for k, v in enc.items()}
                gen = tr.model.generate(**enc, forced_bos_token_id=eng, max_new_tokens=128, num_beams=2)
                for i, t in zip(chunk, tr.tok.batch_decode(gen, skip_special_tokens=True), strict=True):
                    out[i] = t
    return out


def translations(split: str) -> pd.DataFrame:
    f = V2 / f"native_v2_{split}_en_mt.csv"
    if not f.exists():
        d = load("native", split)
        d["en_mt"] = to_english(d["headline"].tolist(), d["lang"].tolist())
        d[["id", "en_mt"]].to_csv(f, index=False)
    return pd.read_csv(f)


def run_native(split: str, w_sent: float | None, w_event: float | None) -> dict[str, Any]:
    d = load("native", split)
    nat = Native()
    ps_n, pe_n, _, _ = nat.probs(d["headline"].tolist())
    res: dict[str, Any] = {"split": split}
    res["native"] = native_metrics(d, [SENT[i] for i in ps_n.argmax(1)], [CLASSES[i] for i in pe_n.argmax(1)])
    mt = translations(split).set_index("id").loc[d["id"], "en_mt"].tolist()
    ps_t, pe_t, _, _ = nat.probs(mt, lang="en")
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    res["grid"] = {}
    for ws in grid:
        for we in grid:
            ps, pe = ws * ps_t + (1 - ws) * ps_n, we * pe_t + (1 - we) * pe_n
            m = native_metrics(d, [SENT[i] for i in ps.argmax(1)], [CLASSES[i] for i in pe.argmax(1)])
            res["grid"][f"{ws}|{we}"] = {
                k: m[k] for k in ("sentiment_macro_f1", "event_macro_f1", "event_accuracy")
            }
    if w_sent is not None and w_event is not None:
        ps, pe = w_sent * ps_t + (1 - w_sent) * ps_n, w_event * pe_t + (1 - w_event) * pe_n
        res["chosen"] = {"w_sent": w_sent, "w_event": w_event,
                         **native_metrics(d, [SENT[i] for i in ps.argmax(1)], [CLASSES[i] for i in pe.argmax(1)])}  # fmt: skip
    return res


def run_credit(split: str, cfg: dict[str, Any] | None = None, grid: bool = False) -> dict[str, Any]:
    from engine.nlp.events import EventModel
    from engine.nlp.pipeline import ONNX_DIR

    d = load("credit", split)
    em = EventModel(ONNX_DIR, zero_shot=False, auto_download=False)
    em.credit_cfg.update(cfg or {})
    texts = d["headline"].tolist()
    out: dict[str, Any] = {"split": split}
    if grid:  # dev only: decision threshold x v2 rules
        assert split == "dev"
        base = dict(em.credit_cfg)
        out["grid"] = {}
        for rules in (False, True):
            for th in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
                em.credit_cfg = {**base, "threshold": th, "rules_v2": rules}
                rs = em.classify_batch(texts, allow_zero_shot=False)
                m = credit_metrics(d, [r.primary for r in rs], [r.confidence for r in rs])
                out["grid"][f"rules_v2={rules}|th={th}"] = {
                    k: m[k] for k in ("precision", "recall", "f1", "macro_f1_all_classes")
                } | {"gate_p": m["module_b_gate"]["precision"], "gate_r": m["module_b_gate"]["recall"]}
        em.credit_cfg = base
    rs = em.classify_batch(texts, allow_zero_shot=False)
    out |= {
        "rule": em.credit_cfg,
        "system": credit_metrics(d, [r.primary for r in rs], [r.confidence for r in rs]),
    }
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["translate", "native", "credit", "nli"])
    ap.add_argument("--v", type=float)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--w-sent", type=float)
    ap.add_argument("--w-event", type=float)
    ap.add_argument("--tag", default="")
    ap.add_argument("--rules-v2", action="store_true")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()
    if a.what == "translate":
        translations(a.split)
        return
    if a.what == "nli":
        res = run_nli(a.split, a.w_event if a.w_event is not None else 0.25, a.v)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"nli_{a.split}{'_' + a.tag if a.tag else ''}.json").write_text(json.dumps(res, indent=1))
        for k, m in res["grid"].items():
            print(k, {x: m[x] for x in ("sentiment_macro_f1", "sentiment_macro_f1_ci95", "event_macro_f1")})
        return
    res = (
        run_native(a.split, a.w_sent, a.w_event)
        if a.what == "native"
        else run_credit(
            a.split,
            {
                k: v
                for k, v in {"rules_v2": a.rules_v2 or None, "threshold": a.threshold}.items()
                if v is not None
            },
            a.grid,
        )
    )
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{a.what}_{a.split}{'_' + a.tag if a.tag else ''}.json"
    f.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    show = {k: v for k, v in res.items() if k not in ("grid",)}
    print(json.dumps(show, indent=1, ensure_ascii=False)[:3000])
    if "grid" in res and a.what == "credit":
        for k, v in res["grid"].items():
            print(k, v)
    elif "grid" in res:
        best = sorted(
            res["grid"].items(), key=lambda kv: -(kv[1]["sentiment_macro_f1"] + kv[1]["event_macro_f1"])
        )[:6]
        print("grid top:", best)


if __name__ == "__main__":
    main()
