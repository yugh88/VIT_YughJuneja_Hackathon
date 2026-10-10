import { useEffect, useMemo, useState } from "react";
import { get } from "../api";
import { Chart } from "../components/Chart";
import { Skeleton } from "../components/ui";

// Plain rating against the target fixed before testing.
function rating(v: number, target: number): [string, string] {
  if (v >= target) return ["Strong", "blue"];
  if (v >= target - 0.1) return ["Fair", "amber"];
  return ["Weak · below target", "orange"];
}

export default function TrackRecord({ go }: { go: (h: string) => void }) {
  const [d, setD] = useState<Record<string, any> | null>(null);
  useEffect(() => {
    Promise.all(["eval", "impact", "credit", "bench", "backtest"].map((n) => get(`/v1/eval/${n}`).catch(() => null))).then(([ev, imp, cr, bench, bt]) => setD({ ev, imp, cr, bench, bt }));
  }, []);

  const chart = useMemo(() => {
    const b = d?.imp?.model?.buckets;
    if (!b) return null;
    const b1 = b[0].mean_z;
    return {
      grid: { left: 10, right: 10, top: 30, bottom: 30 },
      tooltip: { formatter: (p: any) => `Impact Score ${p.name}: average move ${p.value.toFixed(2)}× normal volatility (n=${b[p.dataIndex].n})` },
      xAxis: { type: "category", data: b.map((x: any) => x.score), axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: "#5b6b85", fontSize: 14 } },
      yAxis: { type: "value", show: false },
      series: [{
        type: "bar", barWidth: "55%",
        data: b.map((x: any, i: number) => ({ value: x.mean_z, itemStyle: { color: i === b.length - 1 ? "#1d4ed8" : "#c4d3f5", borderRadius: [4, 4, 0, 0] },
          label: i === b.length - 1 ? { show: true, position: "top", color: "#1d4ed8", fontWeight: 800, formatter: `+${Math.round((x.mean_z / b1 - 1) * 100)}%` } : undefined })),
      }],
    };
  }, [d]);

  if (!d) return <main><Skeleton h={600} /></main>;
  const ac = d.cr?.annotated_credit_test?.system?.credit;
  const m = d.imp?.model;
  const b = m?.buckets;
  const ratio = b ? b[b.length - 1].mean_z / b[0].mean_z - 1 : null;
  const s = d.ev?.sentiment;
  const nat = s?.multilingual_native;
  const tiles: [string, number | undefined, number, string][] = [
    ["Sentiment Score · English financial news", s?.en?.twitter_test?.macro_f1, 0.85, "Twitter financial news test"],
    ["Sentiment Score · Indian headlines", s?.en?.sentfin_test?.macro_f1, 0.85, "SEntFiN entity-level test"],
    ["Event Classification · 12 classes", d.ev?.event?.macro_f1_present_classes, 0.8, "9 of 12 classes have test data"],
    ["Sentiment Score · other languages", nat?.sentiment_macro_f1, 0.75, "300 headlines in 10 languages (AI-annotated)"],
  ];
  const v = d.bt?.versions?.[d.bt?.chosen_on_validation];
  const thr = d.bench?.summary ? Object.entries<string>(d.bench.summary).find(([k]) => k.startsWith("best throughput"))?.[1] : null;

  return (
    <main>
      <h1>Track record</h1>
      <p className="lead" style={{ marginBottom: 22 }}>We tested Sentinel on more than two years of news it had never seen before, then checked what markets actually did next.</p>
      <div className="grid g3" style={{ marginBottom: 18 }}>
        <section className="card dark">
          <div className="lbl" style={{ fontWeight: 600 }}>Credit Event detection</div>
          <div className="big" style={{ fontSize: 56, margin: "10px 0" }}>{ac ? `${Math.round(ac.recall * 100)}%` : "–"}</div>
          <div style={{ fontSize: 18 }}>of the {ac?.n_credit ?? "?"} defaults, downgrades and insolvencies in our 150-headline test (2024 onward) were flagged.</div>
          <div className="sub small" style={{ marginTop: 10 }}>
            {ac ? `About 1 in ${Math.round(1 / Math.max(1 - ac.precision, 0.01))} flags was a false alarm (precision ${Math.round(ac.precision * 100)}%). ` : ""}
            Before any stress test, the alert must also be confirmed by several news outlets.
          </div>
        </section>
        <section className="card">
          <div className="muted" style={{ fontWeight: 600 }}>Biggest news, biggest moves</div>
          <div className="big pos" style={{ fontSize: 56, margin: "10px 0", color: "var(--blue)" }}>{ratio != null ? `+${Math.round(ratio * 100)}%` : "–"}</div>
          <div style={{ fontSize: 18 }}>larger price moves followed news with Impact Score 10.</div>
          <div className="muted small" style={{ marginTop: 10 }}>Compared with Impact Score 1 · average move in units of the stock's normal volatility, news from 2024 onward.</div>
        </section>
        <section className="card">
          <div className="muted" style={{ fontWeight: 600 }}>Early warning</div>
          <div className="big" style={{ fontSize: 56, margin: "10px 0", color: "var(--blue)" }}>{m ? `${m.lift_at_8.toFixed(1)}×` : "–"}</div>
          <div style={{ fontSize: 18 }}>more likely that a big market move follows a high-impact alert.</div>
          <div className="muted small" style={{ marginTop: 10 }}>Impact Score 8+ compared with chance (a top-10% move).</div>
        </section>
      </div>

      <section className="card" style={{ marginBottom: 18 }}>
        <h2>How well it reads the news</h2>
        <div className="grid g4">
          {tiles.map(([label, val, target, note]) => {
            const [r, tone] = val != null ? rating(val, target) : ["Pending", ""];
            return (
              <div key={label} className="card" style={{ background: "var(--card-2)", border: 0 }} title={`${note}; macro-F1 target ${target}`}>
                <h3 style={{ fontSize: 17 }}>{label}</h3>
                <div className="row" style={{ marginTop: 8 }}>
                  <span className={`pill ${tone}`}>{r}</span>
                  <span className="muted small">score {val != null ? val.toFixed(2) : "–"} (target {target})</span>
                </div>
              </div>
            );
          })}
        </div>
        <p className="muted small" style={{ marginBottom: 0 }}>Ratings compare each result (macro-F1, 0 to 1) with the target we set before testing. Strong = target met; Fair = within 0.10; Weak = further below.</p>
      </section>

      <div className="grid wide-side" style={{ marginBottom: 18 }}>
        <section className="card">
          <h2 style={{ marginBottom: 2 }}>Higher scores, bigger moves</h2>
          <div className="muted">Average real price move after the news, by Impact Score</div>
          {chart && <Chart option={chart} height={300} label="Average move by Impact Score" />}
          {m && <div className="muted small">Overall, the Impact Score ranks moves only weakly (rank correlation {m.spearman_rho.toFixed(3)}; most of it comes from earnings dates). The jump at 10 is the useful part.</div>}
        </section>
        <section className="card soft">
          <h2>How we tested</h2>
          {[
            "Learned from news up to 2023; tested only on news from 2024 onward.",
            "Repeated copies of the same story were removed before testing.",
            "Test answers were labelled without seeing the system's guesses (by an AI annotator, blind, in two passes; a human spot-check is pending).",
            "Results are checked against real market prices, not opinions.",
          ].map((t) => <div key={t} className="row" style={{ flexWrap: "nowrap", alignItems: "flex-start", margin: "8px 0", fontSize: 16 }}><span className="check">✓</span><span>{t}</span></div>)}
          <button className="btn" style={{ marginTop: 10 }} onClick={() => go("lab")}>Full test report for analysts</button>
        </section>
      </div>

      <section className="card cream">
        <h2>Where it falls short</h2>
        <div className="grid g2" style={{ gap: 8 }}>
          {m && <div>• The Impact Score's overall ranking of moves is weak: {m.spearman_rho.toFixed(3)} against a 0.25 target.</div>}
          {nat && <div>• Other languages: sentiment scores {nat.sentiment_macro_f1.toFixed(2)} against a 0.75 target.</div>}
          {ac && <div>• Credit Event flags: precision {ac.precision.toFixed(2)} against a 0.80 target.</div>}
          {thr && <div>• Speed: {thr} per worker against a 150 docs/s target.</div>}
          {v && <div>• Index tilt: no edge after costs (information ratio {v.test.net_information_ratio.toFixed(2)}; beats {Math.round(v.test.placebo.share_of_placebos_beaten * 100)}% of placebos, needs 95%).</div>}
        </div>
      </section>
    </main>
  );
}
