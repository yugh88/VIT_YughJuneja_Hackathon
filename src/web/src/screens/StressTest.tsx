import { useEffect, useMemo, useRef, useState } from "react";
import { onAction, post, type Alert, type StressResult } from "../api";
import { Chart } from "../components/Chart";
import { Card, Empty, ImpactBadge, Skeleton } from "../components/ui";
import { useData } from "../data";
import countryData from "../data/countries.json";
import { ASSET_LABELS, eventShort, shockChips } from "../labels";

const NAMES = (countryData as any).names as Record<string, string>;
const CET1_MIN = 0.07; // Basel III: 4.5% CET1 minimum + 2.5% capital conservation buffer
const COUNTRIES = ["US", "GB", "DE", "FR", "IT", "ES", "NL", "CH", "PL", "RU", "TR", "SA", "AE", "JP", "CN", "KR", "TW", "IN", "SG", "ID", "AU", "CA", "BR", "MX", "ZA"];
const SECTORS = ["Communication Services", "Consumer Discretionary", "Consumer Staples", "Energy", "Financials", "Health Care", "Industrials", "Information Technology", "Materials", "Real Estate", "Utilities"];
const REPLAY_SECONDS = 45;
const hhmm = (iso: string) => new Date(iso).toISOString().slice(11, 16);
const M = (v: number, d = 0) => `${v < 0 ? "−" : v > 0 ? "+" : ""}$${Math.abs(v / 1e6).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d })} M`;

export default function StressTest({ param, go }: { param?: string; go: (h: string) => void }) {
  const { data, setView } = useData();
  const [selected, setSelected] = useState<string | null>(param ?? null);
  const [result, setResult] = useState<StressResult | null>(null);
  const [custom, setCustom] = useState<StressResult | null>(null);
  const [showBuilder, setShowBuilder] = useState(false);
  const [form, setForm] = useState({ event: "Geopolitical", impact: 9, country: "RU", sector: "" });
  const [busy, setBusy] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const [clock, setClock] = useState<number | null>(null); // replay animation, hours since 00:00 UTC
  const timer = useRef<number | null>(null);

  useEffect(() => {
    if (param?.startsWith("custom~")) {
      // From Try it: "Stress-test a credit event at <company>"
      const [, entity, country, sector] = param.split("~").map(decodeURIComponent);
      const f = { event: "Credit Event", impact: 9, country, sector };
      setForm(f);
      setShowBuilder(true);
      runCustom(f, entity);
    } else setSelected(param ?? null);
  }, [param]); // eslint-disable-line react-hooks/exhaustive-deps

  const startReplay = () => {
    setView("replay");
    setCustom(null);
    setSelected(null);
    setClock(0);
    if (timer.current) window.clearInterval(timer.current);
    const t0 = performance.now();
    timer.current = window.setInterval(() => {
      const h = Math.min(12, ((performance.now() - t0) / 1000 / REPLAY_SECONDS) * 12);
      setClock(h);
      if (h >= 12 && timer.current) { window.clearInterval(timer.current); setClock(null); }
    }, 200);
  };
  const runCustom = async (f = form, entity?: string) => {
    setBusy(true);
    try {
      setCustom(await post<StressResult>("/v1/modules/stress/run", { event: f.event, impact: f.impact, country: f.country || null, sector: f.sector || null, entity_id: entity || null }));
    } finally {
      setBusy(false);
    }
  };
  useEffect(() => onAction((a) => {
    if (a.type === "replay-2022") startReplay();
    if (a.type === "custom-scenario") { setShowBuilder(true); runCustom({ event: "Credit Event", impact: 10, country: "US", sector: "Financials" }); }
  }), []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => void (timer.current && window.clearInterval(timer.current)), []);

  const alerts: Alert[] = useMemo(() => {
    const hours = (iso: string) => { const d = new Date(iso); return d.getUTCHours() + d.getUTCMinutes() / 60; };
    return data.alerts.filter((a) => clock === null || data.view !== "replay" || hours(a.first_utc) <= clock).sort((a, b) => b.severity - a.severity);
  }, [data.alerts, clock, data.view]);

  // During the replay animation, follow the newest alert; otherwise default to the most serious one.
  const last = useRef(0);
  useEffect(() => {
    if (clock !== null && alerts.length !== last.current) {
      last.current = alerts.length;
      const newest = [...alerts].sort((a, b) => b.first_utc.localeCompare(a.first_utc))[0];
      if (newest?.result_id) setSelected(newest.result_id);
    } else if (!selected && alerts[0]?.result_id) setSelected(alerts[0].result_id);
  }, [alerts, clock, selected]);

  useEffect(() => {
    if (!selected) { setResult(null); return; }
    data.result(selected).then(setResult);
  }, [selected, data]);

  const r = custom ?? result;
  const sel = alerts.find((a) => a.result_id === selected);
  return (
    <main>
      <div className="page-head">
        <div>
          <h1>Bank portfolio stress test</h1>
          <p className="lead">When a serious event is detected, we check how much a sample $10 billion bank portfolio of loans, bonds and derivatives could lose.</p>
        </div>
        <div className="row">
          <button className="btn primary" onClick={startReplay} title="Recorded news of 24 Feb 2022 through the real models and trigger, played back on a 45-second clock">↻ Replay 24 Feb 2022</button>
          <button className="btn" onClick={() => setShowBuilder(!showBuilder)}>Build your own scenario</button>
        </div>
      </div>

      {showBuilder && (
        <section className="card" style={{ marginBottom: 18 }}>
          <div className="row" style={{ gap: 22 }}>
            <label className="row">Event
              <select value={form.event} onChange={(e) => setForm({ ...form, event: e.target.value })}>
                <option>Geopolitical</option><option>Macroeconomic</option><option>Credit Event</option>
              </select>
            </label>
            <label className="row" title="Scales the measured anchor-episode shock: 0.5 + 0.5 × (impact − 7) / 3">Impact Score <b>{form.impact}</b>
              <input type="range" min={1} max={10} value={form.impact} onChange={(e) => setForm({ ...form, impact: Number(e.target.value) })} />
            </label>
            <label className="row" title="Exposures in this country take 1.5× the shock, its region 1.2×">Country
              <select value={form.country} onChange={(e) => setForm({ ...form, country: e.target.value })}>
                <option value="">Global</option>{COUNTRIES.map((c) => <option key={c} value={c}>{NAMES[c] ?? c}</option>)}
              </select>
            </label>
            <label className="row" title="Exposures in this sector take 1.3× the shock">Sector
              <select value={form.sector} onChange={(e) => setForm({ ...form, sector: e.target.value })}>
                <option value="">All</option>{SECTORS.map((s) => <option key={s}>{s}</option>)}
              </select>
            </label>
            <button className="btn primary" onClick={() => runCustom()} disabled={busy}>{busy ? "Running…" : "Run stress test"}</button>
            {custom && <button className="link" onClick={() => setCustom(null)}>Back to alerts</button>}
          </div>
        </section>
      )}

      <div className="grid side-main">
        <section className="card" style={{ alignSelf: "start" }}>
          <h2 style={{ marginBottom: 2 }}>Alerts, most serious first</h2>
          <div className="muted" style={{ marginBottom: 12 }}>
            {data.view === "replay"
              ? clock !== null ? `Replaying 24 Feb 2022 · ${String(Math.floor(clock)).padStart(2, "0")}:${String(Math.floor((clock % 1) * 60)).padStart(2, "0")} UTC · ${alerts.length} so far` : `${alerts.length} stress tests triggered on 24 Feb 2022 (00:00–12:00 UTC)`
              : `${alerts.length} stress tests triggered so far`}
          </div>
          {clock !== null && <div className="progress"><span style={{ width: `${(clock / 12) * 100}%` }} /></div>}
          <div className="stack" aria-live="polite">
            {alerts.length === 0 && <Empty>{clock !== null ? "Waiting for the first trigger…" : "No stress test yet. Press “Replay 24 Feb 2022” or build your own scenario."}</Empty>}
            {(showAll ? alerts : alerts.slice(0, 5)).map((a) => (
              <button key={(a.result_id ?? a.cluster) + a.first_utc} className={`alert-item ${a.result_id === selected && !custom ? "sel" : ""}`} onClick={() => { setCustom(null); setSelected(a.result_id); }}>
                <ImpactBadge score={a.impact} />
                <div style={{ minWidth: 0 }}>
                  <div className="t">{a.headline}</div>
                  <div className="muted small">{eventShort(a.event)} · {a.corroboration?.outlets ?? "?"} outlets · {hhmm(a.first_utc)}</div>
                </div>
              </button>
            ))}
          </div>
          {alerts.length > 5 && <button className="link" style={{ marginTop: 12 }} onClick={() => setShowAll(!showAll)}>{showAll ? "Show fewer" : `Show all ${alerts.length} alerts →`}</button>}
          <div className="card" style={{ background: "var(--slate-soft)", border: 0, padding: 14, marginTop: 16 }}>
            <span className="small">A stress test runs only when an event has an <b>Impact Score above 7</b> and at least <b>3 different outlets</b>, including a major one, confirm it. Credit events also need a clear distress word (default, downgrade, bankruptcy…).</span>
          </div>
        </section>

        <div>
          {!r ? (selected ? <Skeleton h={520} /> : <Card><Empty>Select an alert, replay 24 Feb 2022, or build your own scenario.</Empty></Card>) : <Detail r={r} alert={custom ? undefined : sel} />}
        </div>
      </div>
      <p className="footnote">
        Figures from the stress-test engine: a seeded synthetic book, shocks measured from real market moves, Basel IRB capital.{" "}
        <button className="link" onClick={() => go("lab")}>Full test report →</button>
      </p>
    </main>
  );
}

function Detail({ r, alert }: { r: StressResult; alert?: Alert }) {
  const lossPct = r.loss / r.book_value_before;
  const rows = useMemo(() => {
    const items = Object.entries(r.waterfall).map(([k, v]) => ({ label: ASSET_LABELS[k] ?? k, v }));
    items.push({ label: "New loan-loss provisions (ΔEL)", v: -r.delta_expected_loss });
    return items.sort((a, b) => a.v - b.v);
  }, [r]);
  const maxAbs = Math.max(...rows.map((x) => Math.abs(x.v)));
  const exposures = useMemo(() => {
    const { countries, pnl } = r.heatmap;
    return countries.map((c, i) => ({ c, v: pnl[i].reduce((a, b) => a + b, 0) * 1e6 })).sort((a, b) => a.v - b.v).slice(0, 3);
  }, [r]);
  const chips = shockChips(r.scenario.factors, (r.scenario as any).em_fx_pct);
  const src = (r.scenario as any).source ?? {};
  const after = r.cet1_ratio_after;
  const t = r.trigger;
  const title = alert?.headline ?? t?.title;
  const firstBy = alert && t?.title && t.title !== alert.headline ? t.title : null;
  const impact = alert?.impact ?? t?.impact;

  return (
    <div className="stack" style={{ gap: 18 }}>
      <section className="card dark">
        <div className="row" style={{ marginBottom: 18 }}>
          <span className="pill dark">{r.scenario.event}{impact != null ? ` · Impact Score ${impact}` : ""}</span>
          <span style={{ color: "#dbe4f3" }}>{title ? `“${title}”` : `Custom scenario${r.scenario.country ? ` · ${NAMES[r.scenario.country] ?? r.scenario.country}` : ""}${r.scenario.sector ? ` · ${r.scenario.sector}` : ""}`}</span>
        </div>
        {firstBy && <div className="small" style={{ color: "#b8c4da", margin: "-8px 0 14px" }}>Stress test first triggered by: “{firstBy}” ({hhmm(t!.published_at)} UTC)</div>}
        <div className="row between" style={{ alignItems: "center", gap: 24 }}>
          <div className="row" style={{ gap: 28 }}>
            <div><div className="lbl">Portfolio value before</div><div className="big">${(r.book_value_before / 1e9).toFixed(2)} bn</div></div>
            <div style={{ fontSize: 34, color: "var(--cyan)" }}>⟶</div>
            <div><div className="lbl">Portfolio value after stress test</div><div className="big">${(r.book_value_after / 1e9).toFixed(2)} bn</div></div>
          </div>
          <div style={{ background: "#3a2416", borderRadius: 14, padding: "14px 22px" }} title="Mark-to-market loss plus new loan-loss provisions">
            <div style={{ color: "#f9c9a8", fontWeight: 600 }}>Estimated loss</div>
            <div style={{ color: "#fb8c4f", fontSize: 34, fontWeight: 800 }}>{M(-r.loss)} <span style={{ fontSize: 18 }}>({(-lossPct * 100).toFixed(1)}%)</span></div>
          </div>
        </div>
      </section>

      <div className="grid g3">
        <div className="card kpi" title="Change in expected loss: default probabilities stressed with the Vasicek one-factor model, × loss given default × exposure">
          <div className="lbl">Expected Loss (ΔEL)</div>
          <div className="val neg">{M(r.delta_expected_loss)}</div>
          <div className="sub">More borrowers may miss payments</div>
        </div>
        <div className="card kpi" title={`CET1 = capital / risk-weighted assets. Losses cut capital (${r.delta_cet1_from_losses_bps.toFixed(0)} bp); downgrades raise risk weights (${r.delta_cet1_from_rwa_bps.toFixed(0)} bp, RWA ${(r.rwa_before / 1e9).toFixed(2)} → ${(r.rwa_after / 1e9).toFixed(2)} bn)`}>
          <div className="lbl">CET1 ratio</div>
          <div className="val">{(r.cet1_ratio_before * 100).toFixed(1)}% <span style={{ fontSize: 26 }}>→</span> <span className={after < CET1_MIN ? "neg" : ""}>{(after * 100).toFixed(1)}%</span></div>
          <div className="sub">{r.delta_cet1_bps.toFixed(0)} bps · {after >= CET1_MIN ? "still above" : "below"} the 7.0% minimum incl. buffer</div>
          {r.cet1_capital_before != null && (
            <div className="sub">CET1 capital ${(r.cet1_capital_before / 1e6).toFixed(0)} M → ${(r.cet1_capital_after! / 1e6).toFixed(0)} M (loss after {((r.tax_rate ?? 0.25) * 100).toFixed(0)}% tax) · RWA ${(r.rwa_before / 1e9).toFixed(2)} → ${(r.rwa_after / 1e9).toFixed(2)} bn{r.rwa_cap_binds ? " (capped at +15%)" : ""}</div>
          )}
        </div>
        <div className="card kpi" title={`99% VaR from ${r.monte_carlo.draws.toLocaleString()} correlated Monte Carlo draws; 97.5% Expected Shortfall ${M(r.monte_carlo.es_97_5)}`}>
          <div className="lbl">99% Value at Risk</div>
          <div className="val neg">{M(r.monte_carlo.var_99).replace("+", "")}</div>
          <div className="sub">{r.monte_carlo.draws.toLocaleString()} Monte Carlo scenarios · ES 97.5% {M(r.monte_carlo.es_97_5).replace("+", "")}</div>
        </div>
      </div>

      <div className="grid g2" style={{ gridTemplateColumns: "minmax(0, 1.12fr) minmax(0, 1fr)" }}>
        <Card title="Loss by asset class" tip="Mark-to-market P&L by asset class plus new provisions; together they make the estimated loss">
          {rows.map((x) => (
            <div key={x.label} className="row" style={{ gap: 14, margin: "9px 0", flexWrap: "nowrap" }}>
              <span style={{ width: 190, flex: "none" }}>{x.label}</span>
              <div style={{ flex: 1 }}><div className="track" style={{ height: 12 }}><span style={{ left: 0, width: `${Math.max(3, (Math.abs(x.v) / maxAbs) * 100)}%`, background: x.v < 0 ? "var(--orange)" : "var(--blue)" }} /></div></div>
              <b className={x.v < 0 ? "neg" : "pos"} style={{ width: 80, textAlign: "right", flex: "none" }}>{M(x.v)}</b>
            </div>
          ))}
        </Card>
        <Card title="Shocks applied" tip="Mean factor moves of the scenario, scaled by severity, then targeted by country (×1.5), region (×1.2) and sector (×1.3)">
          <div className="muted" style={{ marginBottom: 12 }}>Calibrated to market moves {src.anchor ? `in the ${src.anchor}` : ""}{src.window ? ` (${src.window[0]} to ${src.window[1]})` : ""}, scaled to {Math.round(r.scenario.severity * 100)}%</div>
          <div className="row">{chips.map((c) => <span key={c.label} className={`pill ${c.tone}`} style={{ fontSize: 15, padding: "6px 14px" }}>{c.label}</span>)}</div>
          <h2 style={{ fontSize: 17, marginTop: 20 }}>Largest exposures hit</h2>
          {exposures.map((e) => (
            <div key={e.c} className="kv"><span>{NAMES[e.c] ?? e.c} counterparties</span><b className={e.v < 0 ? "neg" : "pos"}>{M(e.v)}</b></div>
          ))}
        </Card>
      </div>

      <details className="card">
        <summary>More detail: loss distribution, country × sector map, top counterparties</summary>
        <MoreDetail r={r} />
      </details>
    </div>
  );
}

function MoreDetail({ r }: { r: StressResult }) {
  const hist = useMemo(() => {
    const { counts, edges } = r.monte_carlo.histogram_musd;
    const idx = (v: number) => Math.max(0, edges.findIndex((e) => e >= v / 1e6) - 1);
    return {
      grid: { left: 50, right: 16, top: 30, bottom: 40 },
      tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: counts.map((_, i) => ((edges[i] + edges[i + 1]) / 2).toFixed(0)), name: "loss $M", nameLocation: "middle", nameGap: 26 },
      yAxis: { type: "value", name: "scenarios" },
      series: [{ type: "bar", data: counts, itemStyle: { color: "#94a3b8" }, markLine: { symbol: "none", data: [
        { name: "VaR 99%", xAxis: idx(r.monte_carlo.var_99), lineStyle: { color: "#1d4ed8" }, label: { formatter: "VaR 99%", position: "insideEndTop" } },
        { name: "ES 97.5%", xAxis: idx(r.monte_carlo.es_97_5), lineStyle: { color: "#e8590c" }, label: { formatter: "ES 97.5%", position: "insideEndBottom" } },
      ] } }],
    };
  }, [r]);
  const heat = useMemo(() => {
    const { countries, sectors, pnl } = r.heatmap;
    const keep = countries.map((_, i) => pnl[i].some((v) => Math.abs(v) > 0.05));
    const rows = countries.filter((_, i) => keep[i]);
    const data: [number, number, number][] = [];
    let lo = 0;
    countries.forEach((cc, i) => keep[i] && pnl[i].forEach((v, j) => { data.push([j, rows.indexOf(cc), v]); lo = Math.min(lo, v); }));
    return {
      grid: { left: 40, right: 10, top: 10, bottom: 90 },
      tooltip: { formatter: (p: any) => `${rows[p.data[1]]} × ${sectors[p.data[0]]}: $${p.data[2].toFixed(2)} M` },
      xAxis: { type: "category", data: sectors, axisLabel: { rotate: 40, fontSize: 10 } },
      yAxis: { type: "category", data: rows, axisLabel: { fontSize: 10 } },
      visualMap: { min: Math.min(lo, -0.01), max: 0, show: false, inRange: { color: ["#e8590c", "#f4f6fa"] } },
      series: [{ type: "heatmap", data }],
    };
  }, [r]);
  return (
    <div className="grid g2" style={{ marginTop: 14 }}>
      <div><h3>Loss distribution (10,000 correlated scenarios)</h3><Chart option={hist} height={280} label="Monte Carlo loss histogram" /></div>
      <div><h3>Loss by country and sector ($M)</h3><Chart option={heat} height={320} label="Country by sector loss heatmap" /></div>
      <div>
        <h3>Top 10 losing counterparties</h3>
        <table><tbody>{r.top_losers.map((x) => <tr key={x.name}><td>{x.name}</td><td className={`num ${x.pnl < 0 ? "neg" : "pos"}`}>{M(x.pnl, 1)}</td></tr>)}</tbody></table>
      </div>
      <div>
        <h3>Capital, step by step</h3>
        <table><tbody>
          <tr><td>Mark-to-market P&L</td><td className="num">{M(r.mtm_pnl, 1)}</td></tr>
          <tr><td>New provisions (ΔEL)</td><td className="num">{M(-r.delta_expected_loss, 1)}</td></tr>
          {r.loss_after_tax != null && <>
            <tr><td>Loss before tax → after {((r.tax_rate ?? 0.25) * 100).toFixed(0)}% tax</td><td className="num">{M(-r.loss_pre_tax!, 1)} → {M(-r.loss_after_tax, 1)}</td></tr>
            <tr><td>CET1 capital (12% of RWA before)</td><td className="num">${(r.cet1_capital_before! / 1e6).toFixed(0)} M → ${(r.cet1_capital_after! / 1e6).toFixed(0)} M</td></tr>
          </>}
          <tr><td>Risk-weighted assets</td><td className="num">${(r.rwa_before / 1e9).toFixed(2)} → ${(r.rwa_after / 1e9).toFixed(2)} bn</td></tr>
          {r.rwa_cap_binds && <tr><td>RWA before the +15% cap on rating migration</td><td className="num">${(r.rwa_after_uncapped! / 1e9).toFixed(2)} bn (cap applied)</td></tr>}
          <tr><td>ΔCET1 from losses alone</td><td className="num">{r.delta_cet1_from_losses_bps.toFixed(0)} bp</td></tr>
          <tr><td>ΔCET1 from higher risk weights alone</td><td className="num">{r.delta_cet1_from_rwa_bps.toFixed(0)} bp</td></tr>
        </tbody></table>
      </div>
    </div>
  );
}
