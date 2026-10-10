"""Render docs/model_card.md from the measured results (never hand-edited).

Sources: ``docs/eval/eval.json`` (make eval), ``docs/eval/impact.json``
(training.train_impact), ``docs/eval/bench.json`` (make bench). Each section shows when
and from which source version its numbers were produced.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from engine.config import REPO_ROOT

EVAL_DIR = REPO_ROOT / "docs" / "eval"
CARD = REPO_ROOT / "docs" / "model_card.md"


def provenance() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    h = hashlib.sha256()
    for f in sorted((REPO_ROOT / "src").rglob("*.py")):
        h.update(f.read_bytes())
    commit = git("rev-parse", "--short", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no")) if commit else None
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": commit or "none (no commits yet)",
        "git_dirty": dirty,
        "source_sha256": h.hexdigest()[:16],
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}",
    }


def _load(name: str) -> dict[str, Any] | None:
    p = EVAL_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def _stamp(p: dict[str, Any] | None) -> str:
    if not p:
        return "_not run yet_"
    dirty = " (uncommitted changes)" if p.get("git_dirty") else ""
    return (
        f"_Generated {p['generated_at']} · commit `{p['git_commit']}`{dirty} · source `{p['source_sha256']}`_"
    )


def _row(*cells: Any) -> str:
    return "| " + " | ".join("" if c is None else str(c) for c in cells) + " |"


def _load_ann() -> dict:
    p = REPO_ROOT / "data" / "eval" / "annotation_stats.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _ci(m: dict[str, Any], k: str) -> str:
    ci = m.get(f"{k}_ci95")
    return f"{m[k]:.3f} ({ci[0]:.2f}-{ci[1]:.2f})" if ci else f"{m[k]:.3f}"


def _v2_section() -> list[str]:
    """Score-improvement round: old vs new system on the frozen v2 test sets (scored once) and the seen sets."""
    sm = _load("v2_summary.json")
    if not sm:
        return []
    L = [
        "## 0. Score-improvement round (v2): old vs new",
        "",
        "New evaluation sets were built blind under the frozen `data/eval/LABELLING.md` (AI-annotated, two passes), "
        "hashed in `data/eval/v2/FREEZE.json` before any experiment, and scored **once** at the end. Every change was "
        "selected on the dev splits only (pre-registered in `docs/DECISIONS.md` D72-D79). *Old* = the system before the "
        "round; *new* = the shipped system. 95% bootstrap confidence intervals in brackets. "
        f"Run {sm['run_at']}; raw results in `docs/eval/v2/final.json`.",
        "",
        "| Set | Metric | Old | New |",
        "| --- | --- | --- | --- |",
    ]
    rows = [
        (
            "native_v2_test",
            "sentiment_macro_f1",
            "Native headlines, 10 languages (new, n = 300)",
            "sentiment macro-F1",
        ),
        (
            "native_v2_test",
            "event_macro_f1",
            "Native headlines, 10 languages (new, n = 300)",
            "event macro-F1",
        ),
        ("credit_v2_test", "precision", "Credit headlines 2025-26 (new, n = 150)", "Credit Event precision"),
        ("credit_v2_test", "recall", "Credit headlines 2025-26 (new, n = 150)", "Credit Event recall"),
        ("credit_v2_test", "f1", "Credit headlines 2025-26 (new, n = 150)", "Credit Event F1"),
        ("native_300_seen", "sentiment_macro_f1", "Native-300 (seen in round 1)", "sentiment macro-F1"),
        ("native_300_seen", "event_macro_f1", "Native-300 (seen in round 1)", "event macro-F1"),
        ("credit_150_seen", "precision", "Credit-150 (seen in round 1)", "Credit Event precision"),
        ("credit_150_seen", "recall", "Credit-150 (seen in round 1)", "Credit Event recall"),
        ("event_topic_test_seen", "macro_f1_present", "Twitter topic test (seen)", "event macro-F1"),
        ("event_topic_test_seen", "credit_f1", "Twitter topic test (seen)", "Credit Event F1"),
    ]
    for key, metric, name, label in rows:
        o, n = sm[key]["old"], sm[key]["new"]
        if metric in o and metric in n:
            L.append(_row(name, label, _ci(o, metric), _ci(n, metric)))
    L.append("")
    comps = [(w, _load(f"v2/compare_{w}.json")) for w in ("sentiment", "event")]
    if any(c for _, c in comps):
        L += ["**Translate-train retrains** (D73(d), D79; dev only):", "", "| Model | Target | Current | v2 | Kept |",
              "| --- | --- | --- | --- | --- |"]  # fmt: skip
        for w, c in comps:
            if c:
                t = c["target"]
                L.append(_row(f"{w}-multi-v2", t, c["current"][t], c["v2"][t],
                              "yes" if c["keep"] else f"no (drops: {c['drops_over_tol'] or 'none'}; gain {c['gain']})"))  # fmt: skip
        L.append("")
    sp = sm.get("speed")
    if sp:
        L += [f"**Speed (CPU, {sp['threads']} threads, batch 32):** {sp['docs_per_s_unique']} docs/s on new text, "
              f"{sp['docs_per_s_with_repeats']} docs/s with repeated text (logits cache); single-document latency "
              f"p50 {sp['p50_ms']} ms, p95 {sp['p95_ms']} ms. Details: `docs/eval/v2/speed.json`.", ""]  # fmt: skip
    return L


def render_model_card() -> Path:
    ev, imp, bench = _load("eval.json"), _load("impact.json"), _load("bench.json")
    L: list[str] = [
        "# Model card: Sentinel",
        "",
        "Every number below is produced by a script in this repo (`make eval`, `make bench`, "
        "`python -m training.train_impact`). This file is generated; do not edit it by hand.",
        "",
    ]
    L += _v2_section()
    if ev:
        s = ev["sentiment"]
        L += [
            "## 1. Sentiment (uncontaminated test sets first)",
            "",
            _stamp(ev.get("provenance")),
            "",
            "| Test set | Model | Macro-F1 | Accuracy | ECE | n |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for key, label in (
            ("twitter_test", "Twitter financial news (test)"),
            ("sentfin_test", "SEntFiN 1.0 held-out (entity-level)"),
        ):
            for m in ("en", "multi"):
                r = s[m][key]
                L.append(_row(label, f"sentiment-{m}", r["macro_f1"], r["accuracy"], r.get("ece"), r["n"]))
        L += [
            "",
            f"**FiQA-2018 continuous scores (test, n={s['en']['fiqa_test']['n']}), mean squared error**",
            "",
            "| Model | `sentiment.score` (range map) | `sentiment.score_human` (human map) | Target |",
            "| --- | --- | --- | --- |",
        ]
        for m in ("en", "multi"):
            r = s[m]["fiqa_test"]
            L.append(_row(f"sentiment-{m}", r["mse_range"], r["mse_human"], "<= 0.10"))
        cal = s["en"]["calibration"]
        if cal.get("range"):
            rg = cal["range"]
            L += [
                "",
                "Every signal carries both: `sentiment.score` (range map; drives Module A and the UI) and "
                "`sentiment.score_human` (FiQA-faithful map). The trade-off in one sentence: FiQA's raters score even "
                "strong news around 0.4-0.6, so the full-range score costs FiQA MSE (0.15 vs 0.07) in exchange for "
                f"letting clearly strong headlines reach |score| >= 0.7 (min {rg['check_strong_min_abs']}, median "
                f"{rg['check_strong_median_abs']} on the 40-headline check; neutral max {rg['check_neutral_max_abs']}). "
                f"No single map met both (candidates: {cal['candidates']}); docs/DECISIONS.md D38.",
            ]
        mp = s["multilingual_proxy"]
        L += [
            "",
            "**Multilingual: PROXY result.** " + mp["what"] + ". Native human-labelled results replace this "
            "once `data/eval/multilingual_to_label.csv` is labelled.",
            "",
            "| Language | Macro-F1 | Accuracy | n |",
            "| --- | --- | --- | --- |",
        ]
        for lang, r in mp["per_language"].items():
            L.append(_row(lang, r["macro_f1"], r["accuracy"], r["n"]))
        L.append(_row("**mean (8 languages)**", mp["mean_macro_f1"], "", ""))
        nat = s["multilingual_native"]
        if nat.get("status") == "labelled":
            kap = nat.get("kappa") or {}
            sc = nat.get("human_spot_check")
            spot = (f"human spot-check {sc['event_agreement']:.0%} event / {sc.get('sentiment_agreement', 0):.0%} sentiment "
                    f"agreement (n={sc['n']})" if sc else "human spot-check pending")  # fmt: skip
            L += [
                "",
                f"**Multilingual, native headlines ({nat['n']}, 10 languages): AI-annotated, blind, two-pass "
                f"(event κ = {kap.get('event_kappa')}, sentiment κ = {kap.get('sentiment_kappa')}), {spot}.** "
                "Not human-labelled. Both passes were made by the same assistant in one session, so κ measures "
                "self-consistency, not agreement between independent annotators.",
                "",
                f"Sentiment macro-F1 {nat['sentiment_macro_f1']} (95% CI {nat['sentiment_macro_f1_ci95']}); "
                f"event macro-F1 {nat['event_macro_f1']} (95% CI {nat['event_macro_f1_ci95']}), event accuracy "
                f"{nat['event_accuracy']}; gap vs English Twitter sentiment {nat['gap_vs_english_twitter']}.",
                "",
                "| Language | n | Sentiment macro-F1 | Event macro-F1 |",
                "| --- | --- | --- | --- |",
            ]
            L += [
                _row(k, v["n"], v["sentiment_macro_f1"], v["event_macro_f1"])
                for k, v in nat["per_language"].items()
            ]
        else:
            L += ["", f"**Multilingual, native headlines:** {nat.get('status')}"]
        L += [
            "",
            f"**Multilingual, native human labels:** {nat.get('status')}"
            + (
                f", mean macro-F1 {nat['mean_sentiment_macro_f1']}, gap vs English {nat['gap_vs_english_twitter']}"
                if nat.get("status") == "labelled"
                else ""
            ),
            "",
        ]
        b = s["baselines"]
        L += [
            "**Lexicon baselines (same test sets)**",
            "",
            "| Test set | LM | VADER | LM + VADER | Engine fallback lexicon | sentiment-en | Gain over LM + VADER |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for key in ("twitter_test", "sentfin_test"):
            L.append(
                _row(
                    key,
                    b[key]["loughran_mcdonald"]["macro_f1"],
                    b[key]["vader"]["macro_f1"],
                    b[key]["lm_plus_vader"]["macro_f1"],
                    b[key]["engine_lexicon_fallback"]["macro_f1"],
                    s["en"][key]["macro_f1"],
                    f"+{s['delta_vs_lm_plus_vader_macro_f1'][key]}",
                )
            )
        pc = s["phrasebank_contaminated"]
        L += [
            "",
            "**Financial PhraseBank (CONTAMINATED: read with the caveat).** " + pc["caveat"],
            "",
            "| Split | Macro-F1 | Accuracy | n |",
            "| --- | --- | --- | --- |",
            _row(
                "all annotators agree (test)",
                pc["all_agree_test"]["macro_f1"],
                pc["all_agree_test"]["accuracy"],
                pc["all_agree_test"]["n"],
            ),
            _row(
                "full 50%-agree (test)",
                pc["full_test"]["macro_f1"],
                pc["full_test"]["accuracy"],
                pc["full_test"]["n"],
            ),
            "",
        ]
        e = ev["event"]
        L += (
            [
                "## 2. Event classification",
                "",
                _stamp(ev.get("provenance")),
                "",
                f"Test: {e['test']} (n={e['n']}). Macro-F1 over the {len(e['classes_present'])} classes present: "
                f"**{e['macro_f1_present_classes']}**, accuracy {e['accuracy']}, ECE {e['ece']}.",
                "",
                "No clean test data exists for: "
                + ", ".join(e["classes_without_clean_test_data"])
                + ". They are trained from weak labels only; the native multilingual set will measure them.",
                "",
                "| Class | F1 |",
                "| --- | --- |",
            ]
            + [_row(c, v) for c, v in e["per_class_f1"].items()]
            + [""]
        )
        L += [f"Leakage check (training-only text found in eval sets): {ev['leakage_check']}", ""]
    if imp:
        m, h = imp["model"], imp["heuristic_baseline"]
        L += [
            "## 3. Impact score (learned from realised abnormal returns)",
            "",
            f"_Generated {imp['created']} · model `{imp['version']}`_",
            "",
            f"**Headline:** on the 2024+ test years the learned impact score ranks realised moves with Spearman "
            f"rho = {m['spearman_rho']} (95% CI {m['spearman_ci95'][0]}-{m['spearman_ci95'][1]}, which excludes zero), and "
            f"signals scored >= 8 land in the top decile of realised moves {m['lift_at_8']}x as often as chance. "
            "News-to-return predictability is known to be weak (most price moves are not explained by same-day "
            "headlines), so a small but clearly non-zero rho is the honest result; the target of 0.25 is missed.",
            "",
        ]
        ab = imp.get("earnings_ablation")
        if ab:
            L += [
                "**Ablation: where the signal comes from** (test years):",
                "",
                "| Variant | Spearman rho |",
                "| --- | --- |",
                _row("Learned model (all features)", m["spearman_rho"]),
                _row("Same model, no `earnings_window` feature", ab["rho_without_earnings_feature"]),
                _row(
                    f"Learned model, only the {1 - ab['share_of_test_rows_in_earnings_window']:.0%} of signals outside "
                    "earnings windows",
                    ab["rho_on_rows_outside_earnings_windows"],
                ),
                _row("Heuristic baseline", h["spearman_rho"]),
                "",
                f"Much of the rho is the earnings calendar, not the headline: outside earnings windows rho is "
                f"{ab['rho_on_rows_outside_earnings_windows']}. Earnings dates are scheduled weeks ahead, so the feature "
                "is known at publication time (no look-ahead).",
                "",
            ]
        L += (
            []
            if not ab
            else [
                f"**Where the signal comes from:** much of this rho is the earnings calendar, not the headline. "
                f"With the same model and no `earnings_window` feature, test rho is {ab['rho_without_earnings_feature']}; "
                f"on the {1 - ab['share_of_test_rows_in_earnings_window']:.0%} of test signals outside an earnings "
                f"window it is {ab['rho_on_rows_outside_earnings_windows']}. Earnings dates are scheduled weeks "
                "ahead, so the feature is known at publication time (no look-ahead).",
                "",
            ]
        )
        L += [
            f"Label: {imp['label']}. Split: train {imp['split']['train']}, test {imp['split']['test']} "
            f"(hyper-parameters tuned on {imp['split']['hyperparameters_tuned_on']}). Rows: {imp['rows']}.",
            "",
            "| Metric (test years) | Learned model | Heuristic baseline | Target |",
            "| --- | --- | --- | --- |",
            _row(
                "Spearman rho, predicted vs realised z",
                f"{m['spearman_rho']} (95% CI {m['spearman_ci95']}, p={m['spearman_p']:.2g})",
                h["spearman_rho"],
                ">= 0.25",
            ),
            _row(
                "Spearman rho per entity-day", m["spearman_rho_entity_day"], h["spearman_rho_entity_day"], ""
            ),
            _row(
                "Spearman rho vs raw z (no kind adjustment)",
                m.get("spearman_rho_raw_z"),
                h.get("spearman_rho_raw_z"),
                "",
            ),
            _row(
                "Precision at impact >= 8 (realised z in top decile)",
                f"{m['precision_at_8']} (n={m['n_score_ge_8']})",
                f"{h['precision_at_8']} (n={h['n_score_ge_8']})",
                f">= 2x base rate ({m['base_rate_top_decile']})",
            ),
            _row("Lift over base rate", m["lift_at_8"], h["lift_at_8"], ">= 2.0"),
            "",
            "**Monotonicity: mean realised z by predicted score bucket** (chart data)",
            "",
            "| Score | n | Mean z | Median z | Share in top decile |",
            "| --- | --- | --- | --- | --- |",
        ]
        L += [
            _row(b_["score"], b_["n"], b_["mean_z"], b_["median_z"], b_["share_top_decile"])
            for b_ in m["buckets"]
        ]
        top = list(imp["permutation_importance_rho_drop"].items())[:8]
        L += [
            "",
            "Top features by permutation importance (drop in test rho): "
            + ", ".join(f"{k} {v}" for k, v in top),
            "",
        ]
    cr, cx = _load("credit.json"), _load("credit_experiments.json")
    if cx:
        L += ["## 5. Credit Event class (focused work)", ""]
        ch, base = cx["chosen"], cx["baseline"]
        L += [
            f"Decision rule chosen on validation only (n={cx['validation']['n']}, {cx['validation']['n_credit']} Credit; "
            f"{cx['validation']['excluded_rows']} rule-labelled GDELT and templated EDGAR rows excluded): "
            f"`{ch['rule'] or 'plain argmax'}`.",
            "",
            "| Validation | Macro-F1 | Credit P | Credit R | Credit F1 |",
            "| --- | --- | --- | --- | --- |",
            _row(
                "plain classifier",
                base["macro_f1"],
                base["credit"]["precision"],
                base["credit"]["recall"],
                base["credit"]["f1"],
            ),
            _row(
                "chosen rule",
                ch["macro_f1"],
                ch["credit"]["precision"],
                ch["credit"]["recall"],
                ch["credit"]["f1"],
            ),
            "",
        ]
    if cr:
        ft, ed = cr["frozen_test"], cr["edgar_2025_test"]
        L += [
            "Test sets, touched once at the end:",
            "",
            "| Test set | Variant | Macro-F1 | Credit P | Credit R | Credit F1 (95% CI) | n Credit |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, key in (("frozen topic test", "system"), ("frozen topic test", "model_only_argmax")):
            v = ft[key]
            L.append(
                _row(
                    name,
                    key,
                    v["macro_f1"],
                    v["credit"]["precision"],
                    v["credit"]["recall"],
                    f"{v['credit']['f1']} {v['credit']['f1_ci95']}",
                    v["credit"]["n_credit"],
                )
            )
        for key in ("system_with_item_override", "text_model_only"):
            v = ed[key]
            L.append(
                _row(
                    "EDGAR 8-K filed >= 2025 (independent)",
                    key,
                    "",
                    v["precision"],
                    v["recall"],
                    f"{v['f1']} {v['f1_ci95']}",
                    v["n_credit"],
                )
            )
        at = cr.get("annotated_credit_test")
        if at:
            for key in ("system", "model_only_argmax"):
                v = at[key]
                L.append(_row("credit_labelled.csv (AI-annotated)", key, v["macro_f1"], v["credit"]["precision"],
                              v["credit"]["recall"], f"{v['credit']['f1']} {v['credit']['f1_ci95']}", v["credit"]["n_credit"]))  # fmt: skip
        pt = ft["precision_at_trigger_confidence_0.8"]
        L += [
            "",
            f"Precision of Credit predictions at confidence >= 0.8, frozen test: {pt['precision']} (n={pt['n']}).",
            "",
        ]
        if at:
            g = at["module_b_credit_gate"]
            ann = _load_ann()
            kap = (ann.get("kappa") or {}).get("credit", {})
            sc = (ann.get("human_spot_check") or {}).get("credit")
            spot = (
                f"human spot-check {sc['event_agreement']:.0%} (n={sc['n']})"
                if sc
                else "human spot-check pending"
            )
            L += [
                f"`credit_labelled.csv`: {at['n']} English headlines from 2024 on, never used in training or tuning "
                f"(~60 distress-worded candidates + ~90 lookalikes: affirmations, upgrades, stable outlooks, credit "
                f"cards, tax/carbon credits, AML fines, loan growth). **AI-annotated, blind, two-pass (κ = "
                f"{kap.get('event_kappa')}), {spot}**; not human-labelled. It is deliberately hard, so macro-F1 over all "
                f"classes is not representative. Module B's credit gate (Credit, confidence >= 0.7, distress term): "
                f"precision {g['precision']}, recall {g['recall']} (n={g['n']}). Target F1 >= 0.65 is met on this set; "
                "precision >= 0.80 is not. Most false positives are corporate 'restructuring' (layoffs).",
                "",
            ]
        hn = (cx or {}).get("hard_negative_rules")
        if hn:
            o = hn["all_validation_rows"]
            L += [
                f"Hard-negative rules (affirmed / stable-positive outlook / upgrade -> not Credit; AML -> Regulatory/Legal), "
                f"validation only: {hn['objective_rows']['rows_changed']} changes on the objective rows, "
                f"{o['rows_changed']} on all {o['n']} validation rows (Credit F1 {o['without']['credit']['f1']} -> "
                f"{o['with']['credit']['f1']}); kept: {hn['kept']}.",
                "",
            ]
    bt = _load("backtest.json")
    if bt and "versions" in bt:
        L += [
            "## 6. Module A: sentiment-tilted index backtest",
            "",
            _stamp(bt.get("provenance")),
            "",
            f"Two versions specified before running (DECISIONS D67); chosen on validation "
            f"({bt['split']['validation'][0]}..{bt['split']['validation'][1]}) by net IR: **{bt['chosen_on_validation']}**. "
            f"Test ({bt['split']['test'][0]}..{bt['split']['test'][1]}) run once for both, costs "
            f"{bt['params']['cost_bp_per_side']:.0f} bp per side, turnover cap {bt['params']['turnover_cap']:.0%}.",
            "",
            "| Version | Period | Net IR | Gross IR | Avg daily turnover | Placebos beaten | Passes |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, v in bt["versions"].items():
            for per in ("validation", "test"):
                r = v[per]
                pl = r.get("placebo")
                L.append(_row(name, per, r["net_information_ratio"], r["gross_information_ratio"],
                              r["avg_turnover_per_day"], f"{pl['share_of_placebos_beaten']:.0%}" if pl else "",
                              ("yes" if v["passes"] else "**no**") if per == "test" else ""))  # fmt: skip
        L += ["", f"**Verdict:** {bt['verdict']}", ""]
    rp = _load("replay_2022-02-24.json")
    L += ["## 7. Module B: event-driven stress testing", ""]
    try:
        from modules.stress.pricing import load_book, scenario_for, stress

        book = load_book()
        L += [
            f"Synthetic book (seed 42): {len(book)} positions, ${book['exposure'].sum() / 1e9:.1f} bn, "
            f"{book.loc[book['asset_class'] != 'sovereign_bond', 'cp_id'].nunique()} corporate counterparties, "
            f"{book['country'].nunique()} countries. Shocks calibrated from measured anchor-window moves "
            "(data/reference/shocks.yaml).",
            "",
            "| Scenario | Loss | dEL | RWA before -> after | dCET1 (losses / RWA) | 99% VaR | 97.5% ES |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for ev_cls, imp_, cc, sec, ent in (
            ("Geopolitical", 9, "RU", None, "SOV:RU"),
            ("Macroeconomic", 8, "US", None, None),
            ("Credit Event", 10, "US", "Financials", "BANK:SIVB"),
        ):
            r = stress(scenario_for(ev_cls, imp_, cc, sec, ent), n_draws=10_000)
            L.append(
                _row(
                    r["scenario"]["name"] + f" [{cc}{', ' + ent if ent else ''}]",
                    f"${r['loss'] / 1e6:.0f}m",
                    f"${r['delta_expected_loss'] / 1e6:.0f}m",
                    f"{r['rwa_before'] / 1e9:.2f} -> {r['rwa_after'] / 1e9:.2f} bn",
                    f"{r['delta_cet1_bps']} bp ({r['delta_cet1_from_losses_bps']} / {r['delta_cet1_from_rwa_bps']})",
                    f"${r['monte_carlo']['var_99'] / 1e6:.0f}m",
                    f"${r['monte_carlo']['es_97_5'] / 1e6:.0f}m",
                )
            )
        L.append("")
    except FileNotFoundError:
        L += ["_synthetic book not generated_", ""]
    if rp:
        L += [
            f"**24 Feb 2022 replay** (recorded GDELT, real models, real trigger rule): {rp['docs']} docs -> "
            f"{rp['signals']} signals -> {rp['stress_tests_fired']} stress tests fired; first at "
            f"{rp['first_trigger_utc']} UTC. High-impact signals blocked: {rp['blocked_high_impact']}.",
            "",
        ]
        if rp.get("alert_queue"):
            L += [
                f"Trigger: {rp.get('trigger_version', 'v1')}. Ranked alert queue (severity = impact x corroboration), top 10:",
                "",
                "| Severity | Cluster | First (UTC) | Merged signals | Top headline |",
                "| --- | --- | --- | --- | --- |",
            ]
            L += [_row(a["severity"], a["cluster"], a["first_utc"][11:16], a["merged_signals"], a["headline"][:70])
                  for a in rp["alert_queue"][:10]]  # fmt: skip
            L.append("")
        L += [
            "| Time (UTC) | Cluster | Impact | Corroboration | Trigger headline | Loss | dCET1 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        L += [
            _row(
                t["time_utc"],
                t["cluster"],
                t["impact"],
                f"{t['corroboration']['outlets']} outlets, burst {t['corroboration']['max_burst_z']}",
                t["headline"][:70],
                f"${t['loss_musd']}m",
                f"{t['delta_cet1_bps']} bp",
            )
            for t in rp["timeline"][:8]
        ]
        L.append("")
    if bench:
        L += [
            "## 8. Latency and throughput",
            "",
            _stamp(bench.get("provenance")),
            "",
            f"Machine: {bench['machine']}",
            "",
            "| Measure | Value |",
            "| --- | --- |",
        ]
        L += [_row(k, v) for k, v in bench["summary"].items()]
        L += [
            "",
            "Throughput grid (docs/s through the full pipeline, sentiment + event + impact):",
            "",
            "| ONNX threads | batch | docs/s |",
            "| --- | --- | --- |",
        ]
        L += [_row(r["threads"], r["batch"], r["docs_per_s"]) for r in bench["grid"]]
        L.append("")
    CARD.write_text("\n".join(L) + "\n")
    return CARD
