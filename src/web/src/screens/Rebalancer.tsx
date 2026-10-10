import { useEffect, useMemo, useState } from "react";
import { get, post } from "../api";
import { Chart } from "../components/Chart";
import { Card, Skeleton } from "../components/ui";
import { eventShort } from "../labels";

const COLOURS = ["#1d4ed8", "#0d9488", "#7c3aed", "#e8590c", "#f59e0b"];
const OTHERS = "#cbd5e1";
const KAPPAS = [0, 0.25, 0.5, 1, 2];
const KAPPA_LABEL = ["Off", "Low", "Medium", "High", "Very high"];
const NAMES: Record<string, string> = {
  AAPL: "Apple", MSFT: "Microsoft", NVDA: "NVIDIA", GOOGL: "Alphabet", META: "Meta", AMZN: "Amazon", TSLA: "Tesla", JPM: "JPMorgan",
  BAC: "Bank of America", GS: "Goldman", XOM: "Exxon", CVX: "Chevron", JNJ: "J&J", UNH: "UnitedHealth", PFE: "Pfizer", PG: "P&G",
  KO: "Coca-Cola", WMT: "Walmart", CAT: "Caterpillar", BA: "Boeing",
};
type Range = "today" | "7d" | "30d";

export default function Rebalancer({ go }: { go: (h: string) => void }) {
  const [st, setSt] = useState<any>(null);
  const [bt, setBt] = useState<any>(null);
  const [range, setRange] = useState<Range>("7d");
  const [k, setK] = useState<number | null>(null);

  useEffect(() => {
    get("/v1/eval/backtest").then(setBt).catch(() => undefined);
    let stop = false;
    const tick = () => get("/v1/modules/index?history=2000").then((s) => !stop && setSt(s)).catch(() => undefined);
    tick();
    const id = window.setInterval(tick, 5000);
    return () => { stop = true; window.clearInterval(id); };
  }, []);

  // Points to plot: [label, {ticker: weight}]
  const points = useMemo((): [string, Record<string, number>][] => {
    if (range === "today") {
      const h: { t: number; weights: Record<string, number> }[] = st?.history ?? [];
      if (!h.length) return [];
      const step = Math.max(1, Math.ceil(h.length / 8));
      const pick = h.filter((_, i) => i % step === 0 || i === h.length - 1).slice(-8);
      return pick.map((p, i) => [i === pick.length - 1 ? "Now" : new Date(p.t * 1000).toISOString().slice(11, 16), p.weights]);
    }
    const d = bt?.daily_last30;
    if (!d) return [];
    const n = range === "7d" ? 7 : 30;
    const idx = d.dates.map((_: string, i: number) => i).slice(-n).filter((_: number, j: number, arr: number[]) => n === 7 || j % 4 === (arr.length - 1) % 4);
    return idx.map((i: number) => {
      const day = new Date(d.dates[i] + "T00:00:00Z");
      const label = n === 7 ? day.toLocaleDateString("en-GB", { weekday: "short", timeZone: "UTC" }) : day.toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });
      return [label, Object.fromEntries(Object.entries<number[]>(d.weights).map(([tk, w]) => [tk, w[i]]))];
    });
  }, [range, st, bt]);

  const topNames = useMemo(() => {
    const lastW = points.length ? points[points.length - 1][1] : {};
    return Object.entries(lastW).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([tk]) => tk);
  }, [points]);

  const option = useMemo(() => ({
    grid: { left: 4, right: 4, top: 6, bottom: 28 },
    tooltip: { trigger: "axis", valueFormatter: (v: number) => `${(v * 100).toFixed(1)}%` },
    xAxis: { type: "category", data: points.map((p) => p[0]), axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: "#5b6b85", fontSize: 13 } },
    yAxis: { type: "value", max: 1, show: false },
    series: [
      ...topNames.map((tk, i) => ({ name: NAMES[tk] ?? tk, type: "bar", stack: "w", barCategoryGap: "12%", itemStyle: { color: COLOURS[i], borderColor: "#fff", borderWidth: 2, borderRadius: 3 }, data: points.map((p) => p[1][tk] ?? 0) })),
      { name: "15 others", type: "bar", stack: "w", itemStyle: { color: OTHERS, borderColor: "#fff", borderWidth: 2, borderRadius: 3 }, data: points.map((p) => 1 - topNames.reduce((a, tk) => a + (p[1][tk] ?? 0), 0)) },
    ],
  }), [points, topNames]);

  const kIdx = KAPPAS.indexOf(KAPPAS.reduce((best, x) => (Math.abs(x - (k ?? st?.kappa ?? 0.5)) < Math.abs(best - (k ?? st?.kappa ?? 0.5)) ? x : best), 0.5));
  const setKappa = async (i: number) => { setK(KAPPAS[i]); await post(`/v1/modules/index/kappa?value=${KAPPAS[i]}`).catch(() => undefined); };

  const changes = useMemo(() => {
    if (!st) return [];
    return [...st.weights].sort((a: any, b: any) => Math.abs(b.active) - Math.abs(a.active)).slice(0, 3).map((w: any) => {
      const sig = st.log?.find((e: any) => e.signal?.entity === w.ticker)?.signal;
      return { ...w, sig };
    });
  }, [st]);
  const liveOk = st ? Math.max(...st.weights.map((w: any) => w.weight)) <= 0.15 + 1e-9 && Math.max(...Object.values<number>(st.sector_weights)) <= 0.35 + 1e-9 : null;

  const v = bt ? bt.versions[bt.chosen_on_validation] : null;
  const other = bt ? Object.entries<any>(bt.versions).find(([n]) => n !== bt.chosen_on_validation) : null;
  const beats = v ? v.passes : false;
  const story = v
    ? v.test.gross_information_ratio < 0
      ? "The index reacts to news as designed, but it trailed the cap-weighted index even before trading costs, and the result was not clearly better than luck."
      : "The index reacts to news as designed; before costs it was ahead, but trading costs ate the gains, and the result was not clearly better than luck."
    : "";

  return (
    <main>
      <div className="page-head">
        <div>
          <h1>Tactical index rebalancing</h1>
          <p className="lead">A mock index of 20 S&P 100 stocks. Positive Sentiment Score increases a stock's weight; negative Sentiment Score decreases it, within risk limits.</p>
        </div>
        <div className="seg" role="tablist">
          {(["today", "7d", "30d"] as Range[]).map((r) => (
            <button key={r} className={range === r ? "on" : ""} onClick={() => setRange(r)}>{r === "today" ? "Today" : r === "7d" ? "Last 7 days" : "Last 30 days"}</button>
          ))}
        </div>
      </div>

      <div className="grid main-side" style={{ marginBottom: 22 }}>
        <Card
          title="Index weights over time"
          right={
            <label className="row" style={{ fontWeight: 600 }} title={`Tilt strength κ: weight × e^(κ·S), clipped to 0.5×–2× (κ = ${KAPPAS[kIdx]})`}>
              Sentiment tilt strength
              <input type="range" min={0} max={4} step={1} value={kIdx} onChange={(e) => setKappa(Number(e.target.value))} style={{ accentColor: "var(--blue)", width: 170 }} />
              <span className="muted">{KAPPA_LABEL[kIdx]}</span>
            </label>
          }
        >
          {points.length ? <Chart option={option} height={360} label="Stacked index weights over time" /> : <Skeleton h={360} />}
          <div className="row" style={{ gap: 18, marginTop: 6 }}>
            {topNames.map((tk, i) => <span key={tk} className="row" style={{ gap: 6 }}><i style={{ width: 13, height: 13, borderRadius: 3, background: COLOURS[i], display: "inline-block" }} />{NAMES[tk] ?? tk}</span>)}
            <span className="row" style={{ gap: 6 }}><i style={{ width: 13, height: 13, borderRadius: 3, background: OTHERS, display: "inline-block" }} />15 others</span>
          </div>
          <div className="muted small" style={{ marginTop: 6 }}>
            {range === "today" ? "Live weights since the engine started (demo: recorded news replayed)." : `Backtest weights, last ${range === "7d" ? 7 : 30} trading days to ${bt?.daily_last30?.dates?.slice(-1)[0] ?? "–"} (${bt?.chosen_on_validation ?? ""}).`}
          </div>
        </Card>
        <div className="stack" style={{ gap: 18 }}>
          <Card title="Largest weight changes" tip="Live weights vs the cap-weighted base, with the latest signal on that stock">
            {!st ? <Skeleton h={160} /> : changes.map((w: any) => (
              <div key={w.ticker} className="row" style={{ alignItems: "flex-start", flexWrap: "nowrap", margin: "10px 0", gap: 14 }}>
                <span className="num-dot" style={{ width: 36, height: 36, borderRadius: 8, background: w.active >= 0 ? "var(--blue-soft)" : "var(--orange-soft)", color: w.active >= 0 ? "var(--blue)" : "var(--orange)" }}>{w.active >= 0 ? "▲" : "▼"}</span>
                <div>
                  <b style={{ fontSize: 16 }}>{NAMES[w.ticker] ?? w.ticker} {(w.base * 100).toFixed(1)}% → {(w.weight * 100).toFixed(1)}%</b>
                  <div className="muted small">
                    {w.sig ? <>Sentiment Score {w.sig.sentiment >= 0 ? "+" : "−"}{Math.abs(w.sig.sentiment).toFixed(2)} · {eventShort(w.sig.event)}</> : <>Sentiment Score used {w.score >= 0 ? "+" : "−"}{Math.abs(w.score).toFixed(2)}</>}
                  </div>
                </div>
              </div>
            ))}
          </Card>
          <Card title="Risk limits">
            <div className="kv"><span>Max single-stock weight</span><b>15%</b></div>
            <div className="kv"><span>Max sector weight</span><b>35%</b></div>
            <div className="kv"><span>Max turnover per rebalance</span><b>10%</b></div>
            {liveOk !== null && <span className={`pill ${liveOk ? "green" : "orange"}`} style={{ marginTop: 8 }}>{liveOk ? "All limits held" : "A limit is breached"}</span>}
          </Card>
        </div>
      </div>

      {v && (
        <section className="card cream">
          <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1.25fr) repeat(3, minmax(0, 1fr))", alignItems: "stretch" }}>
            <div>
              <span className="pill amber">Backtest result</span>
              <h2 style={{ fontSize: 26, margin: "10px 0" }}>Does the sentiment tilt beat the cap-weighted index? {beats ? "Yes." : "No."}</h2>
              <p style={{ margin: 0 }}>We tested on market data from 2024 onward, after choosing between two designs on 2020–2023 data. {story}</p>
              <button className="link small" style={{ marginTop: 8 }} onClick={() => go("lab")}>Full comparison of both designs →</button>
            </div>
            <div className="card" title="Information ratio of the tilted index against the cap-weighted index, after 10 bp per side trading costs">
              <div className="muted small">Information ratio, after costs</div>
              <div className={`val ${v.test.net_information_ratio < 0 ? "neg" : "pos"}`} style={{ fontSize: 34, fontWeight: 800 }}>{v.test.net_information_ratio < 0 ? "−" : "+"}{Math.abs(v.test.net_information_ratio).toFixed(2)}</div>
              <div className="muted small">vs. cap-weighted index · before costs {v.test.gross_information_ratio.toFixed(2)}{other ? ` · other design ${other[1].test.net_information_ratio.toFixed(2)}` : ""}</div>
            </div>
            <div className="card" title="The same backtest with the sentiment dates shuffled 100 times; a real effect should beat 95% of them">
              <div className="muted small">Placebo test</div>
              <div style={{ fontSize: 34, fontWeight: 800 }}>{Math.round(v.test.placebo.share_of_placebos_beaten * 100)}%</div>
              <div className="muted small">of random shuffles beaten · needs 95%</div>
            </div>
            <div className="card" title="Share of backtest rebalance days on which post-trade weights met every limit (name 15%, floor 1%, sector 35%, turnover 10%)">
              <div className="muted small">Risk limits held</div>
              <div style={{ fontSize: 34, fontWeight: 800, color: "var(--green)" }}>{bt.risk_limits ? `${Math.round(bt.risk_limits.share_held * 100)}%` : "–"}</div>
              <div className="muted small">{bt.risk_limits ? `${bt.risk_limits.rebalance_days.toLocaleString()} rebalance days` : ""}</div>
            </div>
          </div>
        </section>
      )}
      <p className="footnote">Weights and test results are measured values from the rebalancer and its backtest (10 bp per side costs).</p>
    </main>
  );
}
