import { useCallback, useEffect, useMemo, useState } from "react";
import type { StressResult } from "../api";
import { Chart } from "../components/Chart";
import { Card, ImpactBadge, Kpi, SentimentBar, Skeleton } from "../components/ui";
import { alertCountry, levelAt, LEVEL_TONE, riskSeries, stories, TRIGGER_EVENTS, useData, type Level, type S } from "../data";
import mapData from "../data/worldmap.json";
import countryData from "../data/countries.json";
import { ASSET_LABELS, eventShort, eventTone, langName } from "../labels";

const NAMES = (countryData as any).names as Record<string, string>;
const CENTROID = (mapData as any).centroids as Record<string, [number, number]>;
const fmtInt = (n: number) => n.toLocaleString("en-US");
const hhmm = (t: number | string) => new Date(t).toISOString().slice(11, 16);

export default function Home({ go }: { go: (hash: string) => void }) {
  const { data } = useData();
  const sig = data.signals;
  const [start, end] = data.window;
  const at = end.getTime();

  const langs = Object.entries(data.languages).sort((a, b) => b[1] - a[1]);
  const langNote = langs.slice(0, 3).map(([k]) => langName(k)).join(", ") + (langs.length > 3 ? " and more" : "");

  // Country states (D68) and story counts for the map.
  const countries = useMemo(() => {
    const n = new Map<string, { stories: Map<string, S>; sum: number; k: number; maxImp: number }>();
    for (const s of sig) {
      if (!s.country || !TRIGGER_EVENTS.has(s.ev)) continue;
      const c = n.get(s.country) ?? { stories: new Map(), sum: 0, k: 0, maxImp: 0 };
      const prev = c.stories.get(s.doc);
      if (!prev || s.imp > prev.imp) c.stories.set(s.doc, s);
      c.sum += s.s; c.k += 1;
      c.maxImp = Math.max(c.maxImp, s.imp);
      n.set(s.country, c);
    }
    const rank: Record<Level, number> = { High: 3, Elevated: 2, Watch: 1, Calm: 0 };
    return [...n.entries()]
      .filter(([cc]) => CENTROID[cc])
      .map(([cc, v]) => {
        const alert = [...data.alerts].filter((a) => alertCountry(a, sig) === cc).sort((a, b) => b.severity - a.severity)[0];
        const top = [...v.stories.values()].sort((a, b) => b.imp - a.imp || b.t.localeCompare(a.t)).slice(0, 2);
        return { cc, n: v.stories.size, maxImp: v.maxImp, mean: v.sum / v.k, top, alert, level: levelAt(at, data.alerts, sig, cc) };
      })
      .sort((a, b) => rank[b.level] - rank[a.level] || b.n - a.n)
      .slice(0, 24);
  }, [sig, data.alerts, at]);

  const onMapClick = useCallback((p: any) => {
    const c = p?.data?.c;
    if (!c) return;
    go(c.alert?.result_id ? `stress/${c.alert.result_id}` : "news");
  }, [go]);

  // ?tip=UA opens that country's hover card (shareable link; also used for screenshots).
  const showTipFromUrl = useCallback((chart: any) => {
    const cc = new URLSearchParams(location.search).get("tip");
    const i = cc ? countries.findIndex((c) => c.cc === cc.toUpperCase()) : -1;
    if (i >= 0) window.setTimeout(() => chart.dispatchAction({ type: "showTip", seriesIndex: 1, dataIndex: i }), 600);
  }, [countries]);

  const level = levelAt(at, data.alerts, sig);
  const levelStart = levelAt(start.getTime() + 30 * 60e3, data.alerts, sig);
  const series = useMemo(() => riskSeries(sig, start, end), [sig, start, end]);
  const peak = series.reduce((m, p) => (p.v > m.v ? p : m), { t: 0, v: -1 });

  const top = useMemo(() => [...data.alerts].sort((a, b) => b.severity - a.severity).slice(0, 3), [data.alerts]);
  const [results, setResults] = useState<Record<string, StressResult | null>>({});
  useEffect(() => {
    top.forEach((a) => a.result_id && !(a.result_id in results) && data.result(a.result_id).then((r) => setResults((p) => ({ ...p, [a.result_id!]: r }))));
  }, [top]); // eslint-disable-line react-hooks/exhaustive-deps

  const companies = useMemo(() => {
    const by = new Map<string, { name: string; sum: number; n: number }>();
    for (const s of sig) {
      if (s.eid.startsWith("SOV:")) continue;
      const c = by.get(s.eid) ?? { name: s.name, sum: 0, n: 0 };
      c.sum += s.s; c.n += 1; by.set(s.eid, c);
    }
    const all = [...by.values()].filter((c) => c.n >= 3).map((c) => ({ name: c.name, score: c.sum / c.n, n: c.n }));
    const pos = [...all].sort((a, b) => b.score - a.score).slice(0, 3);
    const neg = [...all].sort((a, b) => a.score - b.score).slice(0, 3).reverse();
    return [...pos, ...neg.filter((x) => !pos.includes(x))].sort((a, b) => b.score - a.score);
  }, [sig]);

  const events = useMemo(() => {
    const st = stories(sig);
    const c: Record<string, number> = {};
    st.forEach((g) => (c[g.main.ev] = (c[g.main.ev] ?? 0) + 1));
    const total = st.length || 1;
    const rows = Object.entries(c).filter(([k]) => k !== "Other/Non-event").sort((a, b) => b[1] - a[1]);
    const shown = rows.slice(0, 5);
    const other = total - shown.reduce((a, [, v]) => a + v, 0);
    return [...shown.map(([k, v]) => ({ k: eventShort(k), v: v / total, other: false })), { k: "Other", v: other / total, other: true }];
  }, [sig]);

  // Label at most 4 countries, skipping any too close to one already labelled (labels would overlap).
  const labelled = useMemo(() => {
    const out: string[] = [];
    for (const c of countries) {
      const [x, y] = CENTROID[c.cc];
      if (out.every((o) => Math.abs(CENTROID[o][0] - x) > 28 || Math.abs(CENTROID[o][1] - y) > 12)) out.push(c.cc);
      if (out.length === 4) break;
    }
    return new Set(out);
  }, [countries]);

  const mapOption = useMemo(() => ({
    grid: { left: 0, right: 0, top: 0, bottom: 0 },
    xAxis: { type: "value", min: -170, max: 190, show: false },
    yAxis: { type: "value", min: -58, max: 84, show: false },
    tooltip: {
      trigger: "item", confine: true, padding: 0, borderWidth: 0, backgroundColor: "transparent", extraCssText: "box-shadow:none;",
      formatter: (p: any) => (p.seriesIndex === 1 ? tipHtml(p.data.c) : ""),
    },
    series: [
      { type: "scatter", silent: true, symbolSize: 3.4, itemStyle: { color: "#2d4777" }, data: (mapData as any).dots },
      {
        type: "scatter",
        labelLayout: { hideOverlap: true },
        data: countries.map((c) => ({
          value: CENTROID[c.cc], symbolSize: 10 + 7 * Math.log2(1 + c.n),
          itemStyle: { color: LEVEL_TONE[c.level], shadowBlur: c.level === "High" ? 30 : 12, shadowColor: LEVEL_TONE[c.level], borderColor: "#fff", borderWidth: c.level === "High" ? 2 : 1, opacity: 0.9 },
          c,
          emphasis: { scale: 1.25, itemStyle: { borderWidth: 3, shadowBlur: 36 } },
          cursor: "pointer",
          label: {
            show: labelled.has(c.cc), position: CENTROID[c.cc][0] > 100 ? "left" : "right", distance: 12,
            formatter: `${NAMES[c.cc] ?? c.cc} · ${c.level === "High" || c.level === "Elevated" ? `Impact ${c.maxImp}` : c.level.toLowerCase()}`,
            backgroundColor: "#fff", borderRadius: 8, padding: [6, 10], color: c.level === "High" ? "#c2410c" : c.level === "Calm" ? "#1d4ed8" : "#0f1b33", fontWeight: 700, fontSize: 12,
          },
        })),
      },
    ],
  }), [countries, labelled]);

  const riskOption = useMemo(() => ({
    grid: { left: 6, right: 6, top: 30, bottom: 26 },
    tooltip: { trigger: "axis", formatter: (p: any) => `${hhmm(p[0].data[0])} UTC · ${p[0].data[1]} risk points in the hour before` },
    xAxis: { type: "time", axisLine: { show: false }, axisTick: { show: false }, splitLine: { show: false }, axisLabel: { color: "#8a97ad", formatter: (v: number) => hhmm(v), hideOverlap: true } },
    yAxis: { type: "value", show: true, axisLabel: { show: false }, splitLine: { lineStyle: { color: "#eef1f6" } } },
    series: [{
      type: "line", smooth: 0.3, showSymbol: false, data: series.map((p) => [p.t, p.v]), lineStyle: { color: "#e8590c", width: 2.5 },
      areaStyle: { color: "rgba(232, 89, 12, 0.12)" },
      markPoint: peak.v > 0 ? { symbol: "circle", symbolSize: 10, itemStyle: { color: "#c2410c" }, label: { show: true, position: "right", formatter: `peak ${hhmm(peak.t)}`, color: "#0f1b33", fontWeight: 700 }, data: [{ coord: [peak.t, peak.v] }] } : undefined,
    }],
  }), [series, peak]);

  const dateLine = data.view === "replay" ? "Thursday, 24 Feb 2022 · 00:00–12:00 UTC (recorded)" : new Date().toLocaleString("en-GB", { weekday: "long", day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" }).replace(" at ", " · ");
  if (data.loading) return <main><Skeleton h={600} /></main>;

  return (
    <main>
      <div className="page-head">
        <div>
          <h1>What could move markets {data.view === "replay" ? "on 24 Feb 2022" : "today"}</h1>
          <p className="lead">Sentinel reads news and social posts in many languages, spots risky events, and shows what they could do to a bank's portfolio and to a stock index.</p>
        </div>
        <span className="muted small">{dateLine}</span>
      </div>

      <div className="grid g4" style={{ marginBottom: 18 }}>
        <Kpi label="Stories read" value={fmtInt(data.stories)} sub="News sites, filings and social posts" tip="Documents ingested (live: since the engine started; replay: the recorded morning of 24 Feb 2022)" />
        <Kpi label="Risk signals found" value={fmtInt(data.signalsTotal)} sub="Linked to a company or country" tip="One signal per (story, company or country) pair" />
        <Kpi label="Languages read" value={langs.length} sub={langNote || "–"} tip="Distinct languages detected (fastText language ID)" />
        <Kpi dark label="Stress tests run" value={data.stressTests} sub={<button className="link" style={{ color: "#7dd3fc" }} onClick={() => go("stress")}>Open the latest →</button>} tip="Stress tests fired by the trigger (impact > 7, corroborated, tier-1 outlet)" />
      </div>

      <div className="grid wide-side" style={{ marginBottom: 18 }}>
        <section className="card dark" style={{ paddingBottom: 14 }}>
          <div className="row between">
            <h2 style={{ color: "#fff", margin: 0 }}>Where risk is building {data.view === "replay" ? "" : "right now"}</h2>
            <span className="status"><span className={`dot ${data.view === "replay" ? "replay" : ""}`} />{data.view === "replay" ? "Recorded replay" : "Updating live"}</span>
          </div>
          <Chart option={mapOption} height={360} label="World map of where risk is building" onClick={onMapClick} onReady={showTipFromUrl} />
          <div className="row between small" style={{ color: "#b8c4da" }}>
            <span className="row" style={{ gap: 16 }}>
              {(["High", "Elevated", "Watch", "Calm"] as Level[]).map((l) => (
                <span key={l} className="row" style={{ gap: 6 }}><span style={{ width: 10, height: 10, borderRadius: 5, background: LEVEL_TONE[l], display: "inline-block" }} />{l === "High" ? "High risk" : l}</span>
              ))}
            </span>
            <span>Bigger circle = more stories</span>
          </div>
        </section>
        <Card title="Overall risk level" tip="High = a stress-test alert with severity ≥ 100 in the last 6 h; Elevated = any alert; Watch = a risk event with impact ≥ 8; Calm otherwise (DECISIONS D68)">
          <div className="row" style={{ alignItems: "baseline" }}>
            <span style={{ fontSize: 30, fontWeight: 800, color: LEVEL_TONE[level] }}>{level}</span>
            <span className="muted">{levelStart === level ? `since ${hhmm(start.getTime())} UTC` : `up from ${levelStart} at ${hhmm(start.getTime())} UTC`}</span>
          </div>
          <Chart option={riskOption} height={200} label="Risk points over time" />
          <p className="muted small" style={{ margin: 0 }}>
            Risk points: for each Geopolitical, Macroeconomic or Credit story in the past hour, its Impact Score above 6, added up. The level comes from the stress-test alerts.
          </p>
        </Card>
      </div>

      <div className="grid wide-side" style={{ marginBottom: 18 }}>
        <Card title="Needs your attention" right={<button className="link" onClick={() => go("news")}>All news →</button>}>
          <div className="stack">
            {top.length === 0 && <div className="muted">No stress-test alert yet. Alerts appear when a serious story is confirmed by several outlets.</div>}
            {top.map((a) => {
              const r = a.result_id ? results[a.result_id] : null;
              const cls = a.impact >= 9 ? "i9" : a.impact >= 8 ? "i8" : "i7";
              const hit = r ? Object.entries(r.waterfall).sort((x, y) => x[1] - y[1])[0] : null;
              const cc = alertCountry(a, sig);
              return (
                <div key={a.result_id ?? a.cluster} className={`attention ${cls}`}>
                  <ImpactBadge score={a.impact} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="row between">
                      <b style={{ fontSize: 16 }}>{a.headline}</b>
                    </div>
                    <div className="row small" style={{ margin: "6px 0" }}>
                      <span className={`pill ${eventTone(a.event)}`}>{eventShort(a.event)}</span>
                      {cc && <span className="pill">{NAMES[cc] ?? cc}</span>}
                      <span className="pill">Reported by {a.corroboration?.outlets ?? "?"} outlets</span>
                      <span className="pill">{hhmm(a.first_utc)} UTC</span>
                    </div>
                    <div className="small">
                      {r ? <>Sample $10 bn bank portfolio would lose about <b>${(r.loss / 1e6).toFixed(0)} M</b>; hit hardest: {ASSET_LABELS[hit![0]]?.toLowerCase() ?? hit![0]}.</> : <span className="muted">Stress test result loading…</span>}
                    </div>
                  </div>
                  <button className="link" onClick={() => go(`stress/${a.result_id}`)}>See the stress test →</button>
                </div>
              );
            })}
          </div>
        </Card>
        <Card title="Sentiment Score by company">
          <div className="muted small">Scale −1.0 (negative) to +1.0 (positive), {data.view === "replay" ? "average over the recorded morning" : "average of recent signals"}; companies with 3+ stories</div>
          <div className="row between small" style={{ margin: "10px 0 4px", fontWeight: 700 }}><span className="neg">◀ Negative</span><span className="pos">Positive ▶</span></div>
          {companies.length === 0 && <div className="muted">Not enough company news yet.</div>}
          {companies.map((c) => (
            <div key={c.name} className="kv" style={{ alignItems: "center" }}>
              <span style={{ width: 130, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={`${c.name}: ${c.n} signals`}>{c.name}</span>
              <SentimentBar score={c.score} width={120} />
              <b className={c.score >= 0 ? "pos" : "neg"} style={{ width: 52, textAlign: "right" }}>{c.score >= 0 ? "+" : "−"}{Math.abs(c.score).toFixed(2)}</b>
            </div>
          ))}
          <button className="link small" style={{ marginTop: 8 }} onClick={() => go("index")}>See the index rebalancer →</button>
        </Card>
      </div>

      <Card title="Event Classification" tip="Share of stories by their main event class">
        {events.map((e) => (
          <div key={e.k} className="row" style={{ gap: 16, margin: "6px 0" }}>
            <span style={{ width: 160 }}>{e.k}</span>
            <div className="track" style={{ flex: 1, height: 10 }}><span style={{ left: 0, width: `${e.v * 100}%`, background: e.other ? "#94a3b8" : "var(--blue)" }} /></div>
            <b style={{ width: 44, textAlign: "right" }}>{Math.round(e.v * 100)}%</b>
          </div>
        ))}
      </Card>
      <p className="footnote">{data.view === "replay" ? "Recorded GDELT news of 24 Feb 2022 run through the real models and trigger." : "Live values from the risk engine; in demo mode the recorded sources replay offline."}</p>
    </main>
  );
}

const esc = (t: string) => t.replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]!);

const short = (t: string) => (t.length > 90 ? t.slice(0, 88) + "…" : t);

/** Hover card for a country on the map (plain HTML: ECharts renders tooltips outside React).
 * Headlines show in English when a translation was recorded (evidence.title_en). */
function tipHtml(c: any): string {
  const tone = LEVEL_TONE[c.level as Level];
  const sent = `${c.mean >= 0 ? "+" : "−"}${Math.abs(c.mean).toFixed(2)}`;
  const rows = c.top
    .map((s: S) => `<div style="margin-top:6px;display:flex;gap:8px;align-items:flex-start"><span style="flex:none;background:${s.imp >= 8 ? "#c2410c" : "#475569"};color:#fff;border-radius:5px;padding:0 6px;font-weight:800;font-size:12px">${s.imp}</span><span style="white-space:normal">${esc(short(s.title_en || s.title))}</span></div>`)
    .join("");
  const alert = c.alert
    ? `<div style="margin-top:8px;padding:6px 8px;border-radius:8px;background:#fdf1ea;color:#c2410c;font-weight:700">Stress test: loss ${c.alert.loss != null ? `$${Math.round(c.alert.loss / 1e6)} M` : "computed"} · severity ${Math.round(c.alert.severity)}</div>`
    : "";
  return `<div style="width:320px;white-space:normal;background:#fff;color:#0f1b33;border-radius:12px;padding:12px 14px;box-shadow:0 10px 30px rgba(15,27,51,.3);font:13px/1.4 'Plus Jakarta Sans',system-ui,sans-serif">
    <div style="display:flex;justify-content:space-between;align-items:center"><b style="font-size:16px">${esc(NAMES[c.cc] ?? c.cc)}</b><span style="background:${tone};color:#fff;border-radius:999px;padding:1px 10px;font-weight:700;font-size:12px">${c.level}</span></div>
    <div style="color:#5b6b85;margin-top:4px">${c.n} ${c.n === 1 ? "story" : "stories"} · highest Impact Score ${c.maxImp} · average Sentiment Score ${sent}</div>
    ${rows}${alert}
    <div style="color:#8a97ad;margin-top:8px;font-size:12px">Click to open ${c.alert?.result_id ? "its stress test" : "the news"}</div></div>`;
}

