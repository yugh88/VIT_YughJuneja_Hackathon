import { useEffect, useMemo, useState } from "react";
import { get } from "../api";
import { Chart, axisStyle } from "../components/Chart";
import { Panel, Skeleton, Status } from "../components/ui";
import { ci, cssVar, num } from "../format";

type Row = { area: string; metric: string; result: string; target: string; met: boolean | null; badge?: string; tip: string };

export default function ModelLab({ theme }: { theme: string }) {
  const [d, setD] = useState<Record<string, any> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    Promise.all(["eval", "impact", "credit", "bench", "backtest"].map((n) => get(`/v1/eval/${n}`).catch(() => null)))
      .then(([evalj, impact, credit, bench, backtest]) => setD({ evalj, impact, credit, bench, backtest }))
      .catch((e) => setErr(String(e)));
  }, []);
  const c = useMemo(() => ({ pos: cssVar("--pos"), neg: cssVar("--neg"), muted: cssVar("--muted"), border: cssVar("--border"), accent: cssVar("--accent"), text: cssVar("--text") }), [theme]);

  if (err) return <main><div className="verdict">{err}</div></main>;
  if (!d) return <main><Skeleton h={400} /></main>;
  const rows = scorecard(d);
  const imp = d.impact;
  const ev = d.evalj?.event;
  const nat = d.evalj?.sentiment?.multilingual_native;
  const proxy = d.evalj?.sentiment?.multilingual_proxy;
  const ax = axisStyle(c.muted, c.border);

  const buckets = imp && {
    grid: { left: 50, right: 50, top: 30, bottom: 40 },
    legend: { textStyle: { color: c.muted }, top: 0 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", name: "predicted impact score", nameLocation: "middle", nameGap: 26, nameTextStyle: { color: c.muted }, data: imp.model.buckets.map((b: any) => b.score), ...ax },
    yAxis: [
      { type: "value", name: "mean realised z", nameTextStyle: { color: c.muted }, ...ax },
      { type: "value", name: "share in top decile", nameTextStyle: { color: c.muted }, ...ax, splitLine: { show: false }, axisLabel: { color: c.muted, formatter: (v: number) => `${(v * 100).toFixed(0)}%` } },
    ],
    series: [
      { name: "mean realised z (kind-adjusted)", type: "bar", data: imp.model.buckets.map((b: any) => b.mean_z), itemStyle: { color: c.pos } },
      {
        name: "share in top decile", type: "line", yAxisIndex: 1, data: imp.model.buckets.map((b: any) => b.share_top_decile), itemStyle: { color: c.accent },
        markLine: { symbol: "none", label: { color: c.muted, formatter: "top-decile base rate", position: "insideStartTop" }, lineStyle: { color: c.muted, type: "dashed" }, data: [{ yAxis: imp.model.base_rate_top_decile }] },
      },
    ],
  };

  const confusion = ev && (() => {
    const labels: string[] = ev.confusion_matrix.labels;
    const m: number[][] = ev.confusion_matrix.matrix;
    const keep = labels.map((_, i) => m[i].some((v) => v > 0) || m.some((r) => r[i] > 0));
    const L = labels.filter((_, i) => keep[i]);
    const data: [number, number, number][] = [];
    labels.forEach((_, i) => labels.forEach((__, j) => keep[i] && keep[j] && data.push([L.indexOf(labels[j]), L.indexOf(labels[i]), m[i][j]])));
    const short = (s: string) => s.replace("/Non-event", "").replace("/Guidance", "").replace("/Acquisition", "").replace("/Supply Chain", "").replace("/Technology Risk", "").replace("/Legal", "").replace("/Climate", "");
    return {
      grid: { left: 110, right: 16, top: 10, bottom: 90 },
      tooltip: { formatter: (p: any) => `gold ${L[p.data[1]]} → predicted ${L[p.data[0]]}: ${p.data[2]}` },
      xAxis: { type: "category", data: L.map(short), name: "predicted", nameLocation: "middle", nameGap: 70, nameTextStyle: { color: c.muted }, ...ax, axisLabel: { color: c.muted, rotate: 40, fontSize: 10 } },
      yAxis: { type: "category", data: L.map(short), ...ax, axisLabel: { color: c.muted, fontSize: 10 } },
      visualMap: { min: 0, max: Math.max(...data.map((x) => x[2])), show: false, inRange: { color: [c.border, c.pos] } },
      series: [{ type: "heatmap", data, label: { show: true, color: c.text, fontSize: 9 } }],
    };
  })();

  const langs = nat?.per_language && {
    grid: { left: 40, right: 10, top: 30, bottom: 30 },
    legend: { textStyle: { color: c.muted }, top: 0 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: Object.keys(nat.per_language), ...ax },
    yAxis: { type: "value", min: 0, max: 1, name: "macro-F1", nameTextStyle: { color: c.muted }, ...ax },
    series: [
      { name: "native sentiment (AI-annotated)", type: "bar", data: Object.values(nat.per_language).map((v: any) => v.sentiment_macro_f1), itemStyle: { color: c.neg } },
      { name: "native event (AI-annotated)", type: "bar", data: Object.values(nat.per_language).map((v: any) => v.event_macro_f1), itemStyle: { color: c.muted } },
      { name: "translated proxy sentiment", type: "bar", data: Object.keys(nat.per_language).map((k) => proxy?.per_language?.[k]?.macro_f1 ?? null), itemStyle: { color: c.pos } },
      { type: "line", data: [], markLine: { symbol: "none", lineStyle: { color: c.accent, type: "dashed" }, label: { color: c.muted, formatter: "target 0.75" }, data: [{ yAxis: 0.75 }] } },
    ],
  };

  const missed = rows.filter((r) => r.met === false).length;
  return (
    <main>
      <h1>Full test report</h1>
      <p className="sub">
        Every result as measured by the scripts in this repo (<code>make eval</code>, <code>make bench</code>, the impact, credit and backtest scripts), misses included.
        {` ${rows.filter((r) => r.met).length} targets met, ${missed} missed, ${rows.filter((r) => r.met === null).length} pending or proxy.`}
      </p>
      <Panel title="Scorecard">
        <table>
          <thead><tr><th>Area</th><th>Metric</th><th>Result (95% CI)</th><th>Target</th><th>Status</th></tr></thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} title={r.tip}>
                <td className="muted">{r.area}</td>
                <td>{r.metric} {r.badge && <span className="chip warn">{r.badge}</span>}</td>
                <td>{r.result}</td>
                <td className="muted">{r.target}</td>
                <td><Status met={r.met} label={r.met === null ? r.badge ?? "report" : undefined} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
      <div className="grid cols-2" style={{ marginTop: 14 }}>
        {imp && (
          <Panel title="Impact score: realised move by predicted bucket (2024+ test)" tip="Mean kind-adjusted realised z and share of signals in the training top decile, per predicted score 1-10">
            <Chart option={buckets} height={300} label="Impact monotonicity chart" />
            {imp.earnings_ablation && (
              <table style={{ marginTop: 8 }}>
                <thead><tr><th>Ablation</th><th className="num">Spearman ρ</th></tr></thead>
                <tbody>
                  <tr><td>Learned model, all features</td><td className="num">{num(imp.model.spearman_rho)} {ci(imp.model.spearman_ci95, 3)}</td></tr>
                  <tr><td>Same model without the earnings-calendar feature</td><td className="num">{num(imp.earnings_ablation.rho_without_earnings_feature)}</td></tr>
                  <tr><td>Only the {Math.round((1 - imp.earnings_ablation.share_of_test_rows_in_earnings_window) * 100)}% of signals outside earnings windows</td><td className="num">{num(imp.earnings_ablation.rho_on_rows_outside_earnings_windows)}</td></tr>
                  <tr><td>Heuristic baseline</td><td className="num">{num(imp.heuristic_baseline.spearman_rho)}</td></tr>
                </tbody>
              </table>
            )}
            <p className="muted" style={{ fontSize: 12 }}>News-to-return predictability is known to be weak; most of this ρ comes from the earnings calendar, not the headline.</p>
          </Panel>
        )}
        {confusion && (
          <Panel title={`Event classifier: confusion matrix (topic test, n=${ev.n})`} tip="Rows = gold class, columns = predicted; classes without clean test data are omitted">
            <Chart option={confusion} height={360} label="Event confusion matrix" />
            <p className="muted" style={{ fontSize: 12 }}>No clean test data for: {ev.classes_without_clean_test_data.join(", ")}.</p>
          </Panel>
        )}
      </div>
      {langs && (
        <div style={{ marginTop: 14 }}>
          <Panel title="Multilingual: native headlines vs translated proxy" tip="Native set: 300 GDELT headlines, 30 per language, AI-annotated blind in two passes; proxy: PhraseBank machine-translated">
            <Chart option={langs} height={280} label="Multilingual results by language" />
            <p className="muted" style={{ fontSize: 12 }}>
              Native labels are <b>AI-annotated, blind, two-pass</b> (event κ = {nat.kappa?.event_kappa}, sentiment κ = {nat.kappa?.sentiment_kappa}; self-consistency of one annotator),{" "}
              {nat.human_spot_check ? `human spot-check ${Math.round(nat.human_spot_check.event_agreement * 100)}%` : "human spot-check pending"}. Not human-labelled. The translated proxy overstated multilingual quality.
            </p>
          </Panel>
        </div>
      )}
    </main>
  );
}

function scorecard(d: Record<string, any>): Row[] {
  const rows: Row[] = [];
  const s = d.evalj?.sentiment;
  if (s) {
    const tw = s.en.twitter_test;
    rows.push({ area: "Sentiment", metric: "Twitter financial news, macro-F1", result: num(tw.macro_f1), target: "≥ 0.85", met: tw.macro_f1 >= 0.85, tip: `n=${tw.n}, uncontaminated test` });
    rows.push({ area: "Sentiment", metric: "Calibration (ECE)", result: num(tw.ece), target: "≤ 0.05", met: tw.ece <= 0.05, tip: "Expected calibration error after temperature scaling" });
    rows.push({ area: "Sentiment", metric: "SEntFiN entity-level, macro-F1", result: num(s.en.sentfin_test.macro_f1), target: "report", met: null, badge: "report", tip: `n=${s.en.sentfin_test.n}` });
    rows.push({ area: "Sentiment", metric: "FiQA MSE (score_human / score)", result: `${num(s.en.fiqa_test.mse_human)} / ${num(s.en.fiqa_test.mse_range)}`, target: "≤ 0.10", met: s.en.fiqa_test.mse_human <= 0.1, tip: "score_human is the FiQA-faithful map; score is the full-range map the UI uses" });
    rows.push({ area: "Sentiment", metric: "Gain over LM + VADER (Twitter)", result: `+${num(s.delta_vs_lm_plus_vader_macro_f1.twitter_test)}`, target: "report", met: null, badge: "report", tip: "Macro-F1 difference vs the better lexicon baseline" });
    rows.push({ area: "Multilingual", metric: "8 languages, translated PhraseBank", result: num(s.multilingual_proxy.mean_macro_f1), target: "≥ 0.75", met: null, badge: "proxy", tip: s.multilingual_proxy.what });
    const n = s.multilingual_native;
    if (n?.status === "labelled") {
      rows.push({ area: "Multilingual", metric: "10 languages, native headlines: sentiment macro-F1", result: `${num(n.sentiment_macro_f1)} ${ci(n.sentiment_macro_f1_ci95)}`, target: "≥ 0.75", met: n.sentiment_macro_f1 >= 0.75, badge: "AI-annotated", tip: "AI-annotated, blind, two-pass; human spot-check pending until spot_check.csv is filled" });
      rows.push({ area: "Multilingual", metric: "Native headlines: event macro-F1", result: `${num(n.event_macro_f1)} ${ci(n.event_macro_f1_ci95)}`, target: "≥ 0.80", met: n.event_macro_f1 >= 0.8, badge: "AI-annotated", tip: `event accuracy ${n.event_accuracy}` });
      rows.push({ area: "Multilingual", metric: "Gap vs English", result: num(n.gap_vs_english_twitter), target: "≤ 0.10", met: n.gap_vs_english_twitter <= 0.1, tip: "English Twitter macro-F1 minus native mean" });
    }
  }
  const e = d.evalj?.event;
  if (e) rows.push({ area: "Events", metric: "Topic test, macro-F1 (9 classes with data)", result: num(e.macro_f1_present_classes), target: "≥ 0.80", met: e.macro_f1_present_classes >= 0.8, tip: "train/test near-duplicates removed (E5)" });
  const cr = d.credit;
  if (cr) {
    const f = cr.frozen_test.model_only_argmax.credit;
    rows.push({ area: "Credit", metric: "Frozen topic test (11 Credit)", result: `F1 ${num(f.f1)} ${ci(f.f1_ci95)}, P ${num(f.precision)}`, target: "F1 ≥ 0.65, P ≥ 0.80", met: false, tip: "about a third of the 11 gold labels look wrong (E1)" });
    const a = cr.annotated_credit_test?.system?.credit;
    if (a) rows.push({ area: "Credit", metric: "150 headlines 2024+ (24 Credit)", result: `F1 ${num(a.f1)} ${ci(a.f1_ci95)}, P ${num(a.precision)}, R ${num(a.recall)}`, target: "F1 ≥ 0.65, P ≥ 0.80", met: a.f1 >= 0.65 && a.precision >= 0.8, badge: "AI-annotated", tip: "F1 target met, precision target missed" });
    const g = cr.annotated_credit_test?.module_b_credit_gate;
    if (g) rows.push({ area: "Credit", metric: "Module B credit gate precision", result: `${num(g.precision)} (n=${g.n})`, target: "≥ 0.80", met: g.precision >= 0.8, badge: "AI-annotated", tip: g.rule });
  }
  const i = d.impact?.model;
  if (i) {
    rows.push({ area: "Impact", metric: "Spearman ρ vs realised abnormal return (2024+)", result: `${num(i.spearman_rho)} ${ci(i.spearman_ci95, 3)}`, target: "≥ 0.25", met: i.spearman_rho >= 0.25, tip: "kind-adjusted z; much of it from the earnings calendar" });
    rows.push({ area: "Impact", metric: "Lift at impact ≥ 8 (top-decile moves)", result: `${num(i.lift_at_8, 2)}×`, target: "≥ 2×", met: i.lift_at_8 >= 2, tip: `precision ${i.precision_at_8} vs base rate ${i.base_rate_top_decile}` });
  }
  const b = d.bench?.summary;
  if (b) {
    const lat = Object.entries(b).find(([k]) => k.startsWith("single-doc latency"));
    const thr = Object.entries(b).find(([k]) => k.startsWith("best throughput"));
    if (lat) rows.push({ area: "Speed", metric: "Single-doc latency p50 / p95, 4 threads", result: String(lat[1]), target: "p95 ≤ 150 ms", met: false, tip: lat[0] });
    if (thr) rows.push({ area: "Speed", metric: "Throughput (best)", result: String(thr[1]), target: "≥ 150 docs/s", met: false, tip: "measured 10-13 docs/s; README scale tables use this" });
  }
  const bt = d.backtest;
  if (bt?.versions) {
    for (const [name, v] of Object.entries<any>(bt.versions)) {
      rows.push({ area: "Module A", metric: `${name}: test net IR, placebo`, result: `IR ${num(v.test.net_information_ratio, 2)} (gross ${num(v.test.gross_information_ratio, 2)}), beats ${Math.round(v.test.placebo.share_of_placebos_beaten * 100)}% of placebos`, target: "IR > 0 and ≥ 95%", met: v.passes, tip: "pre-registered (D67); test run once" });
    }
  }
  return rows;
}
