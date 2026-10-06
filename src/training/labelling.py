"""Blind two-pass annotation workflow for the credit and multilingual evaluation sets.

    python -m training.labelling views        # blind views for pass 1 and pass 2 (scratch/labelling/)
    python -m training.labelling merge        # kappa between passes + the disagreements to adjudicate
    python -m training.labelling finalize     # passes + adjudication -> data/eval/*_labelled.csv
    python -m training.labelling spotcheck    # data/eval/spot_check.csv (20 rows per set, no labels)
    python -m training.labelling agreement    # after the human fills spot_check.csv

Blind views contain only ``id`` and the original headline, in a different seeded order per
pass; machine translations, linked entities, model outputs and selection buckets are left out
(guidelines: data/eval/LABELLING.md, frozen in LABELLING.freeze.json).

Pass files are TSV, one row per headline: ``id<TAB>event<TAB>sentiment<TAB>rationale``
(sentiment is ``-`` for the credit set).
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys

import pandas as pd

from engine.config import REPO_ROOT
from engine.schema import EventClass

EVAL = REPO_ROOT / "data" / "eval"
WORK = REPO_ROOT / "scratch" / "labelling"
SETS = {"credit": ("credit_to_label.csv", "text"), "multi": ("multilingual_to_label.csv", "headline")}
PASS_SEEDS = {1: 11, 2: 29}
EVENTS = {e.value for e in EventClass}
SENTIMENTS = {"negative", "neutral", "positive"}
SPOT_SEED, SPOT_N = 7, 20


def _load(name: str) -> pd.DataFrame:
    f, col = SETS[name]
    return pd.read_csv(EVAL / f).rename(columns={col: "headline"})


def _check_frozen() -> None:
    fz = json.loads((EVAL / "LABELLING.freeze.json").read_text())
    h = hashlib.sha256((EVAL / "LABELLING.md").read_bytes()).hexdigest()
    if h != fz["sha256"]:
        raise SystemExit("LABELLING.md changed after it was frozen")


def views() -> None:
    _check_frozen()
    WORK.mkdir(parents=True, exist_ok=True)
    for name in SETS:
        d = _load(name)
        for p, seed in PASS_SEEDS.items():
            order = d.sample(frac=1.0, random_state=seed)
            lines = [f"{r.id}\t{r.headline}" for r in order.itertuples()]
            (WORK / f"view_pass{p}_{name}.txt").write_text("\n".join(lines) + "\n")
    print(f"blind views written to {WORK}")


def _read_pass(p: int, name: str) -> pd.DataFrame:
    rows = []
    for line in (WORK / f"pass{p}_{name}.tsv").read_text().splitlines():
        if not line.strip():
            continue
        rid, ev, se, why = [*line.split("\t"), "", "", ""][:4]
        rows.append(
            {"id": rid.strip(), "event": ev.strip(), "sentiment": se.strip(), "rationale": why.strip()}
        )
    df = pd.DataFrame(rows).drop_duplicates("id", keep="last").set_index("id")
    bad = df[~df["event"].isin(EVENTS)]
    if len(bad):
        raise SystemExit(f"pass{p}_{name}: unknown event labels {bad['event'].unique().tolist()}")
    if name == "multi" and (~df["sentiment"].isin(SENTIMENTS)).any():
        raise SystemExit(f"pass{p}_{name}: bad sentiment labels")
    return df


def kappa(a: list[str], b: list[str]) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def merge() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name in SETS:
        ids = _load(name)["id"].tolist()
        p1, p2 = _read_pass(1, name), _read_pass(2, name)
        missing = [i for i in ids if i not in p1.index or i not in p2.index]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} ids missing from a pass, e.g. {missing[:5]}")
        a, b = p1.loc[ids], p2.loc[ids]
        res = {"n": len(ids), "event_kappa": round(kappa(a["event"].tolist(), b["event"].tolist()), 4),
               "event_agreement": round(float((a["event"] == b["event"]).mean()), 4)}  # fmt: skip
        credit_a = (a["event"] == "Credit Event").map(str).tolist()
        credit_b = (b["event"] == "Credit Event").map(str).tolist()
        res["credit_binary_kappa"] = round(kappa(credit_a, credit_b), 4)
        dis = (a["event"] != b["event"]).to_numpy().copy()
        if name == "multi":
            res["sentiment_kappa"] = round(kappa(a["sentiment"].tolist(), b["sentiment"].tolist()), 4)
            res["sentiment_agreement"] = round(float((a["sentiment"] == b["sentiment"]).mean()), 4)
            dis |= (a["sentiment"] != b["sentiment"]).to_numpy()
        res["disagreements"] = int(dis.sum())
        out[name] = res
        d = _load(name).set_index("id").loc[ids]
        todo = [f"{i}\t{d.loc[i, 'headline']}\tP1: {a.loc[i, 'event']} / {a.loc[i, 'sentiment']} ({a.loc[i, 'rationale']})"
                f"\tP2: {b.loc[i, 'event']} / {b.loc[i, 'sentiment']} ({b.loc[i, 'rationale']})"
                for i, flag in zip(ids, dis, strict=True) if flag]  # fmt: skip
        (WORK / f"adjudicate_{name}.txt").write_text("\n".join(todo) + "\n")
    (WORK / "kappa.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def finalize() -> None:
    stats = json.loads((WORK / "kappa.json").read_text())
    for name in SETS:
        d = _load(name)
        ids = d["id"].tolist()
        p1, p2 = _read_pass(1, name).loc[ids], _read_pass(2, name).loc[ids]
        adj = _read_pass(3, name) if (WORK / f"pass3_{name}.tsv").exists() else pd.DataFrame()
        rows = []
        for i in ids:
            disagree = p1.loc[i, "event"] != p2.loc[i, "event"] or (
                name == "multi" and p1.loc[i, "sentiment"] != p2.loc[i, "sentiment"]
            )
            if disagree and i not in adj.index:
                raise SystemExit(f"{name}: {i} disagrees between passes but has no adjudication")
            src = adj.loc[i] if disagree else p1.loc[i]
            rows.append({"id": i, "event_label": src["event"],
                         "sentiment_label": src["sentiment"] if name == "multi" else "",
                         "rationale": src["rationale"], "adjudicated": "yes" if disagree else "no",
                         "pass1_event": p1.loc[i, "event"], "pass2_event": p2.loc[i, "event"],
                         "pass1_sentiment": p1.loc[i, "sentiment"] if name == "multi" else "",
                         "pass2_sentiment": p2.loc[i, "sentiment"] if name == "multi" else ""})  # fmt: skip
        lab = pd.DataFrame(rows)
        keep = (
            ["id", "lang", "headline", "english_translation_mt", "published"]
            if name == "multi"
            else ["id", "headline", "date", "outlet"]
        )
        out = d[keep].merge(lab, on="id")
        out.insert(0, "annotation", "AI-annotated, blind, two-pass")
        f = SETS[name][0].replace("_to_label.csv", "_labelled.csv")
        out.to_csv(EVAL / f, index=False)
        print(
            f"{f}: {len(out)} rows, {int((lab['adjudicated'] == 'yes').sum())} adjudicated; kappa {stats[name]}"
        )
    meta = {"guidelines_sha256": json.loads((EVAL / "LABELLING.freeze.json").read_text())["sha256"],
            "annotator": "AI (the assistant that built this project); blind, two passes in different seeded orders, third-pass adjudication",
            "kappa": stats}  # fmt: skip
    (EVAL / "annotation_stats.json").write_text(json.dumps(meta, indent=1))


def spotcheck() -> None:
    rows = []
    for name in SETS:
        d = _load(name).sample(SPOT_N, random_state=SPOT_SEED)
        for r in d.itertuples():
            rows.append({"set": name, "id": r.id, "lang": getattr(r, "lang", "en"), "headline": r.headline,
                         "translation_mt": getattr(r, "english_translation_mt", ""),
                         "verdict_event": "", "verdict_sentiment": "", "notes": ""})  # fmt: skip
    pd.DataFrame(rows).to_csv(EVAL / "spot_check.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"spot_check.csv: {len(rows)} rows")


def agreement() -> None:
    sc = pd.read_csv(EVAL / "spot_check.csv", keep_default_na=False)
    out: dict[str, dict[str, float | int]] = {}
    for name in SETS:
        lab = pd.read_csv(
            EVAL / SETS[name][0].replace("_to_label.csv", "_labelled.csv"), keep_default_na=False
        )
        m = sc[(sc["set"] == name) & (sc["verdict_event"] != "")].merge(
            lab[["id", "event_label", "sentiment_label"]], on="id"
        )
        if not len(m):
            continue
        res: dict[str, float | int] = {
            "n": len(m),
            "event_agreement": round(float((m["verdict_event"] == m["event_label"]).mean()), 4),
        }
        res["credit_binary_agreement"] = round(
            float(((m["verdict_event"] == "Credit Event") == (m["event_label"] == "Credit Event")).mean()), 4
        )
        s = m[m["verdict_sentiment"] != ""]
        if len(s):
            res["sentiment_agreement"] = round(
                float((s["verdict_sentiment"] == s["sentiment_label"]).mean()), 4
            )
        out[name] = res
    stats = json.loads((EVAL / "annotation_stats.json").read_text())
    stats["human_spot_check"] = out
    (EVAL / "annotation_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"views": views, "merge": merge, "finalize": finalize, "spotcheck": spotcheck, "agreement": agreement}[
        sys.argv[1]
    ]()
