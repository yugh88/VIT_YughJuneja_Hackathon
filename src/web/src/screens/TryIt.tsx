import { useEffect, useMemo, useState } from "react";
import { get, onAction, post, type Signal } from "../api";
import { Chart } from "../components/Chart";
import { Card, ImpactDots, SentimentBar, SentimentText, Skeleton } from "../components/ui";
import { stories, toS, useData, whenLabel, type S } from "../data";
import { Title } from "../components/Title";
import { eventShort, eventTone, impactReasons, INDEX_TICKERS, languageLabel, outletName } from "../labels";

// Written for this page; not taken from any training or evaluation set.
export const EXAMPLES: Record<string, string> = {
  Hindi: "टाटा मोटर्स का तिमाही मुनाफा 12% बढ़ा, शेयरों में तेजी",
  Hinglish: "tata motors ka munafa badha",
  Japanese: "トヨタ自動車、半導体不足で国内工場の稼働を一時停止",
  German: "Moody's stuft die Kreditwürdigkeit der Commerzbank herab",
  Arabic: "أرامكو السعودية تعلن ارتفاع أرباحها الفصلية بنسبة 15 في المئة",
  English: "JPMorgan beats estimates as trading revenue jumps; Fed signals rates stay higher for longer",
};

interface Match { id: string; name: string; kind: string; ticker?: string | null; country?: string | null; sector?: string | null; signals: number; score: number }
type Result =
  | { kind: "headline"; res: any; text: string }
  | { kind: "choose"; matches: Match[]; query: string }
  | { kind: "company"; entity: Match; signals: S[]; now: number | null };

export default function TryIt({ go, param }: { go: (h: string) => void; param?: string }) {
  const { view, data } = useData();
  // #try/<text> opens the page with that text (shareable demo links)
  const initial = param ? decodeURIComponent(param) : EXAMPLES.Hindi;
  const [pick, setPick] = useState(param ? Object.keys(EXAMPLES).find((k) => EXAMPLES[k] === initial) ?? "" : "Hindi");
  const [text, setText] = useState(initial);
  const [out, setOut] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const company = async (m: Match) => {
    setBusy(true);
    try {
      if (view === "replay") {
        const sig = data.signals.filter((s) => s.eid === m.id);
        setOut({ kind: "company", entity: m, signals: sig, now: sig.length ? sig.reduce((a, s) => a + s.s, 0) / sig.length : null });
      } else {
        const [sig, agg] = await Promise.all([
          get<{ items: Signal[] }>(`/v1/signals?entity=${encodeURIComponent(m.id)}&limit=500`),
          get<any>(`/v1/entities/${encodeURIComponent(m.id)}/aggregate`).catch(() => null),
        ]);
        setOut({ kind: "company", entity: m, signals: sig.items.map(toS), now: agg?.score ?? null });
      }
    } finally {
      setBusy(false);
    }
  };

  const run = async (t = text) => {
    if (!t.trim()) return;
    setBusy(true);
    setErr(null);
    try {
      const lk = await get<any>(`/v1/lookup?q=${encodeURIComponent(t)}`);
      if (lk.mode === "company") {
        if (lk.resolved) await company(lk.matches.find((m: Match) => m.id === lk.resolved));
        else setOut({ kind: "choose", matches: lk.matches.filter((m: Match) => m.score === lk.matches[0].score), query: lk.query });
      } else {
        setOut({ kind: "headline", res: await post("/v1/analyze", { text: t, explain: true }), text: t });
      }
    } catch (e) {
      setErr(String(e).includes("503") ? "The language models are still loading; try again in a few seconds." : String(e));
    } finally {
      setBusy(false);
    }
  };
  useEffect(() => { setText(initial); run(initial); }, [param]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => onAction((a) => {
    if (a.type === "analyze-example") { const k = a.arg ?? "Hindi"; setPick(k); setText(EXAMPLES[k]); run(EXAMPLES[k]); }
  }), []); // eslint-disable-line react-hooks/exhaustive-deps

  const modeLine = !out ? null
    : out.kind === "company" ? `Showing recent news for ${out.entity.name}${view === "replay" ? " on 24 Feb 2022 (recorded)" : ""}`
    : out.kind === "choose" ? `“${out.query}” matches several companies`
    : "Scoring this text as a headline";

  return (
    <main className="narrow">
      <h1>Try it yourself</h1>
      <p className="lead" style={{ marginBottom: 22 }}>Paste any news headline or social post, in any language, and see how Sentinel reads it. Or ask about a company: “infosys ka kya hua”, “latest on Reliance”.</p>
      <section className="card" style={{ marginBottom: 12 }}>
        <div className="row" style={{ marginBottom: 16 }}>
          <span style={{ fontWeight: 600 }}>Try an example:</span>
          {Object.keys(EXAMPLES).map((k) => (
            <button key={k} className={`filter ${pick === k ? "on" : ""}`} onClick={() => { setPick(k); setText(EXAMPLES[k]); run(EXAMPLES[k]); }}>{k}</button>
          ))}
        </div>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Your text</div>
        <textarea rows={3} value={text} onChange={(e) => { setText(e.target.value); setPick(""); }} onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run(); }} aria-label="headline or company to look up" />
        <div className="row" style={{ marginTop: 14 }}>
          <button className="btn primary" onClick={() => run()} disabled={busy || !text.trim()}>{busy ? "Reading…" : "Analyse this text"}</button>
          {out?.kind === "headline" && <span className="muted">Detected: {languageLabel(out.text, out.res.language.code, out.res.language.confidence)}</span>}
          {err && <span className="neg">{err}</span>}
        </div>
      </section>
      {modeLine && <div className="muted small" style={{ margin: "0 0 14px 4px" }}>{modeLine}</div>}

      {busy && !out ? <Skeleton h={300} /> : !out ? null
        : out.kind === "choose" ? <Choose out={out} onPick={(m) => company(m)} />
        : out.kind === "company" ? <Company out={out} go={go} />
        : <Headline out={out} />}
    </main>
  );
}

// ------------------------------------------------------------------ company lookup

function Choose({ out, onPick }: { out: Extract<Result, { kind: "choose" }>; onPick: (m: Match) => void }) {
  return (
    <Card title="Did you mean">
      <div className="row">
        {out.matches.map((m) => (
          <button key={m.id} className="filter" onClick={() => onPick(m)} title={`${m.sector ?? ""} · ${m.signals} recent signals`}>
            {m.name}{m.signals ? <span className="muted"> · {m.signals}</span> : null}
          </button>
        ))}
      </div>
      <div className="muted small" style={{ marginTop: 10 }}>Numbers are recent signals in our feed.</div>
    </Card>
  );
}

function Company({ out, go }: { out: Extract<Result, { kind: "company" }>; go: (h: string) => void }) {
  const { view } = useData();
  const { entity: e, signals } = out;
  const [book, setBook] = useState<string[]>([]);
  useEffect(() => { get<any>("/v1/modules/stress").then((s) => setBook(s.book?.named_entities ?? [])).catch(() => undefined); }, []);
  const st = useMemo(() => stories(signals), [signals]);
  const top = [...st].sort((a, b) => b.main.imp - a.main.imp || b.main.t.localeCompare(a.main.t)).slice(0, 5);
  const now = out.now ?? (signals.length ? signals.reduce((a, s) => a + s.s, 0) / signals.length : null);

  // Trend: daily means over the last 7 days when the feed covers several days, else hourly.
  const trend = useMemo(() => {
    if (!signals.length) return null;
    const ts = signals.map((s) => new Date(s.t).getTime());
    const span = Math.max(...ts) - Math.min(...ts);
    const daily = span > 2 * 86400e3;
    const bucket = daily ? 86400e3 : 3600e3;
    const end = Math.max(...ts);
    const from = daily ? end - 7 * 86400e3 : Math.min(...ts);
    const by = new Map<number, number[]>();
    signals.forEach((s) => { const t = new Date(s.t).getTime(); if (t >= from) { const k = Math.floor(t / bucket) * bucket; by.set(k, [...(by.get(k) ?? []), s.s]); } });
    const pts = [...by.entries()].sort((a, b) => a[0] - b[0]).map(([k, v]) => [k, +(v.reduce((a, x) => a + x, 0) / v.length).toFixed(3)]);
    return { daily, option: {
      grid: { left: 36, right: 12, top: 14, bottom: 26 },
      tooltip: { trigger: "axis", valueFormatter: (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}` },
      xAxis: { type: "time", axisLabel: { color: "#5b6b85", hideOverlap: true },
        ...(pts.length < 2 ? { min: (pts[0][0] as number) - 3 * 3600e3, max: (pts[0][0] as number) + 3 * 3600e3 } : {}) },
      yAxis: { type: "value", min: -1, max: 1, axisLabel: { color: "#8a97ad" }, splitLine: { lineStyle: { color: "#eef1f6" } } },
      series: [{ type: "line", data: pts, smooth: 0.25, symbolSize: 7, lineStyle: { color: "#1d4ed8", width: 2.5 }, itemStyle: { color: "#1d4ed8" },
        markLine: { symbol: "none", silent: true, lineStyle: { color: "#cbd5e1" }, label: { show: false }, data: [{ yAxis: 0 }] } }],
    } };
  }, [signals]);

  const mix = useMemo(() => {
    const c: Record<string, number> = {};
    st.forEach((g) => (c[g.main.ev] = (c[g.main.ev] ?? 0) + 1));
    return Object.entries(c).sort((a, b) => b[1] - a[1]).slice(0, 6).map(([k, v]) => ({ k, v: v / (st.length || 1) }));
  }, [st]);

  if (!signals.length) {
    return <Card><div className="empty">No recent news for {e.name} in our feed. Try a headline instead.</div></Card>;
  }
  const inIndex = INDEX_TICKERS.has(e.id);
  const inBook = book.includes(e.id);
  return (
    <div className="stack" style={{ gap: 18 }}>
      <section className="card">
        <div className="row between">
          <div>
            <div className="muted small">{e.kind === "sovereign" ? "Country" : [e.sector, e.country, e.ticker].filter(Boolean).join(" · ")}</div>
            <h2 style={{ fontSize: 30, margin: "2px 0 0" }}>{e.name}</h2>
            <div className="muted">{st.length} {st.length === 1 ? "story" : "stories"} in our feed</div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div className="muted small">Sentiment Score now</div>
            <div style={{ fontSize: 34, fontWeight: 800 }} className={now != null && now >= 0.05 ? "pos" : now != null && now <= -0.05 ? "neg" : ""}>
              {now == null ? "–" : `${now >= 0 ? "+" : "−"}${Math.abs(now).toFixed(2)}`}
            </div>
            {now != null && <SentimentBar score={now} width={180} />}
          </div>
        </div>
        <div className="row" style={{ marginTop: 14, gap: 18 }}>
          {inIndex && <button className="link" onClick={() => go("index")}>See it in the index rebalancer →</button>}
          {inBook && <button className="link" onClick={() => go(`stress/custom~${e.id}~${e.country ?? ""}~${e.kind === "sovereign" ? "" : e.sector ?? ""}`)}>Stress-test a credit event at {e.name} →</button>}
        </div>
      </section>
      <div className="grid g2">
        <Card title={trend?.daily ? "Sentiment Score, last 7 days" : "Sentiment Score over time"} tip="Average Sentiment Score of the stories in each period">
          {trend && <Chart option={trend.option} height={220} label="Sentiment trend" />}
          {!trend?.daily && <div className="muted small">Hourly averages; the feed covers less than two days, so a 7-day line is not available yet.</div>}
        </Card>
        <Card title="Event Classification mix">
          {mix.map((m) => (
            <div key={m.k} className="row" style={{ gap: 14, margin: "8px 0", flexWrap: "nowrap" }}>
              <span style={{ width: 150, flex: "none" }}>{eventShort(m.k)}</span>
              <div className="track" style={{ flex: 1, height: 10 }}><span style={{ left: 0, width: `${m.v * 100}%`, background: "var(--blue)" }} /></div>
              <b style={{ width: 44, textAlign: "right" }}>{Math.round(m.v * 100)}%</b>
            </div>
          ))}
        </Card>
      </div>
      <Card title="Top stories by Impact Score">
        <div className="stack">
          {top.map(({ main }) => (
            <div key={main.doc} style={{ borderBottom: "1px solid var(--border)", paddingBottom: 10 }}>
              <div className="row small">
                <span className="pill">{outletName(main.outlet, main.src)}</span>
                <span className="muted">{whenLabel(main, view)}</span>
                <span className={`pill ${eventTone(main.ev)}`}>{eventShort(main.ev)}</span>
              </div>
              <div style={{ fontWeight: 700, fontSize: 16, margin: "6px 0" }}>{main.url ? <a href={main.url} target="_blank" rel="noreferrer" style={{ color: "inherit" }}><Title text={main.title} /></a> : <Title text={main.title} />}</div>
              <div className="row small" style={{ gap: 24 }}>
                <span>Sentiment Score <SentimentText score={main.s} /></span>
                <ImpactDots score={main.imp} low={main.lo} high={main.hi} />
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ------------------------------------------------------------------ headline mode

function Headline({ out }: { out: Extract<Result, { kind: "headline" }> }) {
  const { res, text } = out;
  const sig = res.signals.length ? [...res.signals].sort((a: any, b: any) => b.impact.score - a.impact.score || Math.abs(b.sentiment.score) - Math.abs(a.sentiment.score))[0] : null;
  if (!sig) {
    return (
      <div className="stack" style={{ gap: 18 }}>
        <Card><div className="empty">No company or country we track is named in this text, so no risk signal is produced. Try one of the examples, or type a company name.</div></Card>
        <ReadPanel res={res} text={text} sig={null} />
      </div>
    );
  }
  const others = [...new Set<string>(res.signals.map((s: any) => s.entity.name))].filter((n) => n !== sig.entity.name);
  const ex = res.explanation?.[sig.entity.id];
  const s = sig.sentiment.score;
  const keyWords = ex ? [...ex.words].filter((w: any) => Math.sign(w.delta) === Math.sign(s) && Math.abs(w.delta) >= 0.01).sort((a: any, b: any) => Math.abs(b.delta) - Math.abs(a.delta)).slice(0, 3) : [];
  const reasons = impactReasons(sig.impact, s);
  return (
    <div className="stack" style={{ gap: 18 }}>
      <div className="grid g4">
        <div className="card kpi">
          <div className="lbl">Company / country</div>
          <div className="val" style={{ fontSize: 26 }}>{sig.entity.name}</div>
          <div className="sub">{[sig.entity.sector === "Sovereign" ? "Country" : sig.entity.sector, others.length ? `also mentioned: ${others.join(", ")}` : null].filter(Boolean).join(" · ")}</div>
        </div>
        <div className="card kpi">
          <div className="lbl">Sentiment Score</div>
          <div className={`val ${s >= 0.05 ? "pos" : s <= -0.05 ? "neg" : ""}`} style={{ fontSize: 26 }}>{s >= 0 ? "+" : "−"}{Math.abs(s).toFixed(2)} {sig.sentiment.label[0].toUpperCase() + sig.sentiment.label.slice(1)}</div>
          <SentimentBar score={s} width={240} />
        </div>
        <div className="card kpi">
          <div className="lbl">Event Classification</div>
          <div className="val" style={{ fontSize: 26 }}>{eventShort(sig.event.primary)}</div>
          <div className="sub">{Math.round(sig.event.confidence * 100)}% confident</div>
        </div>
        <div className="card kpi dark">
          <div className="lbl">Impact Score</div>
          <div className="val" style={{ fontSize: 30 }}>{sig.impact.score} / 10</div>
          <div className="sub">Likely range {sig.impact.low} to {sig.impact.high}{sig.impact.mode === "heuristic" ? " · heuristic mode" : ""}</div>
        </div>
      </div>
      <div className="grid g2">
        <Card title="Words that drove the Sentiment Score" tip={ex?.method ?? "word occlusion"}>
          {ex ? (
            <>
              <p style={{ fontSize: 22, margin: "4px 0 10px", lineHeight: 1.6 }}>{mark(ex.sentence, keyWords, s >= 0 ? "up" : "")}</p>
              <div className="muted small">
                {keyWords.length ? <>Highlighted: {keyWords.map((w: any) => `“${w.text}”`).join(" and ")}. </> : "No single word pushes the score in this direction on its own; the reading comes from the sentence as a whole. "}
                Found by removing one word at a time and reading the sentence again.
              </div>
            </>
          ) : <div className="muted">The sentence is too short to explain word by word.</div>}
        </Card>
        <Card title={`Why Impact Score ${sig.impact.score}?`}>
          {sig.impact.cap_reason && (
            <div className="row" style={{ flexWrap: "nowrap", alignItems: "flex-start", margin: "8px 0", fontSize: 16 }}>
              <span className="num-dot" style={{ background: "var(--amber-soft)", color: "#9a5b00" }}>!</span>
              <span>Capped at 5: {sig.impact.cap_reason}.</span>
            </div>
          )}
          {reasons.map((r, i) => (
            <div key={r} className="row" style={{ flexWrap: "nowrap", alignItems: "flex-start", margin: "8px 0", fontSize: 16 }}>
              <span className="num-dot">{i + 1}</span><span>{r}</span>
            </div>
          ))}
        </Card>
      </div>
      <ReadPanel res={res} text={text} sig={sig} />
    </div>
  );
}

function mark(sentence: string, words: { start: number; end: number }[], cls: string) {
  const spans = [...words].sort((a, b) => a.start - b.start);
  const out: (string | JSX.Element)[] = [];
  let i = 0;
  spans.forEach((w, k) => {
    if (w.start < i) return;
    out.push(sentence.slice(i, w.start));
    out.push(<mark key={k} className={cls}>{sentence.slice(w.start, w.end)}</mark>);
    i = w.end;
  });
  out.push(sentence.slice(i));
  return out;
}

/** "How the engine read it": the same numbers as the signal JSON, drawn instead of printed. */
function ReadPanel({ res, text, sig }: { res: any; text: string; sig: any }) {
  const probs: [string, number][] = res.event ? Object.entries<number>(res.event.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 5) : [];
  const s = sig?.sentiment;
  const gauge = s && {
    series: [{
      type: "gauge", min: -1, max: 1, startAngle: 200, endAngle: -20, radius: "100%", center: ["50%", "62%"], splitNumber: 4,
      axisLine: { lineStyle: { width: 16, color: [[0.4, "#e8590c"], [0.6, "#cbd5e1"], [1, "#1d4ed8"]] } },
      pointer: { width: 6, length: "62%", itemStyle: { color: "#0f1b33" } }, anchor: { show: true, size: 12, itemStyle: { color: "#0f1b33" } },
      axisTick: { show: false }, splitLine: { length: 8, lineStyle: { color: "#fff", width: 2 } },
      axisLabel: { color: "#5b6b85", distance: -36, formatter: (v: number) => (v === -1 ? "−1.0" : v === 1 ? "+1.0" : v === 0 ? "0" : "") },
      detail: { show: false }, data: [{ value: s.score }],
    }],
  };
  const download = () => {
    const blob = new Blob([JSON.stringify(sig ?? res, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `sentinel-signal${sig ? `-${sig.entity.id}` : ""}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const lang = languageLabel(text, res.language.code, res.language.confidence);
  return (
    <section className="card">
      <div className="row between"><h2 style={{ margin: 0 }}>How the engine read it</h2><button className="link small" onClick={download}>Download signal (JSON)</button></div>
      <div className="grid g4" style={{ marginTop: 14, alignItems: "start" }}>
        <div>
          <h3>Sentiment Score</h3>
          {s ? (
            <>
              <Chart option={gauge} height={150} label="Sentiment gauge from −1 to +1" />
              <div style={{ textAlign: "center", marginTop: -14 }}>
                <b className={s.score >= 0.05 ? "pos" : s.score <= -0.05 ? "neg" : ""} style={{ fontSize: 20 }}>{s.score >= 0 ? "+" : "−"}{Math.abs(s.score).toFixed(2)} {s.label[0].toUpperCase() + s.label.slice(1)}</b>
                <div className="muted small">{Math.round(s.confidence * 100)}% confident</div>
              </div>
            </>
          ) : <div className="muted">No company or country to score.</div>}
        </div>
        <div>
          <h3>Event Classification</h3>
          {probs.map(([k, v]) => {
            const chosen = res.event && k === res.event.primary;
            return (
              <div key={k} style={{ margin: "7px 0" }}>
                <div className="row between small"><span style={{ fontWeight: chosen ? 800 : 500 }}>{eventShort(k)}</span><span className="muted">{Math.round(v * 100)}%</span></div>
                <div className="track" style={{ height: 8 }}><span style={{ left: 0, width: `${Math.max(1.5, v * 100)}%`, background: chosen ? "var(--blue)" : "#cbd5e1" }} /></div>
              </div>
            );
          })}
        </div>
        <div>
          <h3>Impact Score</h3>
          {sig ? (
            <>
              <div style={{ fontSize: 56, fontWeight: 800, lineHeight: 1 }}>{sig.impact.score}<span className="muted" style={{ fontSize: 22 }}> / 10</span></div>
              <div className="muted">likely range {sig.impact.low}–{sig.impact.high}</div>
              <ImpactDots score={sig.impact.score} />
            </>
          ) : <div className="muted">Needs a company or country.</div>}
        </div>
        <div>
          <h3>Found in the text</h3>
          <p style={{ fontSize: 17, margin: "6px 0", lineHeight: 1.6 }}>{mark(text, res.entities, "up")}</p>
          <div className="muted small">{res.entities.length ? res.entities.map((e: any) => e.name).filter((v: string, i: number, a: string[]) => a.indexOf(v) === i).join(", ") : "No company or country we track."}</div>
          <h3 style={{ marginTop: 14 }}>Language</h3>
          <div>{lang}</div>
        </div>
      </div>
    </section>
  );
}
