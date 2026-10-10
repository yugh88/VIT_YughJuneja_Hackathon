import { useMemo, useState } from "react";
import { Card, ImpactDots, SentimentText, Skeleton } from "../components/ui";
import { stories, useData, whenLabel } from "../data";
import { Title } from "../components/Title";
import { eventShort, eventTone, langName, outletName } from "../labels";

const FILTERS: [string, string | null][] = [
  ["All news", null], ["Geopolitical", "Geopolitical"], ["Macroeconomic", "Macroeconomic"], ["Credit Event", "Credit Event"],
  ["Merger/Acquisition", "Merger/Acquisition"], ["Product Launch", "Product Launch"], ["Earnings", "Earnings/Guidance"],
];

// How each engine source is described to a non-specialist (System page has the technical view).
const GROUPS: [string, string[]][] = [
  ["Global news sites", ["gdelt_gkg", "gdelt_doc"]],
  ["Business news feeds", ["rss", "central_banks"]],
  ["Company filings", ["edgar_8k"]],
  ["Social media", ["mastodon", "reddit", "hackernews", "tweets"]],
];

export default function News({ go }: { go: (h: string) => void }) {
  const { data, health } = useData();
  const promo = health?.pipeline?.promotional_today;
  const [filter, setFilter] = useState<string | null>(null);
  const [only7, setOnly7] = useState(false);
  const [limit, setLimit] = useState(30);
  const all = useMemo(() => stories(data.signals), [data.signals]);
  const shown = all.filter((g) => (!filter || g.main.ev === filter) && (!only7 || g.main.imp >= 7));

  const talked = useMemo(() => {
    const by = new Map<string, { name: string; docs: Set<string>; sum: number; n: number }>();
    for (const s of data.signals) {
      const e = by.get(s.eid) ?? { name: s.name, docs: new Set(), sum: 0, n: 0 };
      e.docs.add(s.doc); e.sum += s.s; e.n += 1; by.set(s.eid, e);
    }
    return [...by.values()].sort((a, b) => b.docs.size - a.docs.size).slice(0, 5);
  }, [data.signals]);

  const groups = GROUPS.map(([label, names]) => {
    const src = data.sources.filter((s) => names.includes(s.name));
    if (!src.length) return null;
    const live = src.some((s) => s.active_mode === "live");
    const fallback = src.some((s) => s.active_mode === "fallback" || s.active_mode === "down");
    const status = live && !fallback ? ["Working", "green"] : fallback ? ["Using saved copy", "amber"] : ["Recorded copy", "amber"];
    return { label, status };
  }).filter(Boolean) as { label: string; status: string[] }[];

  if (data.loading) return <main><Skeleton h={600} /></main>;
  return (
    <main>
      <h1>Live news</h1>
      <p className="lead" style={{ marginBottom: 22 }}>Every story is read in its own language and given a Sentiment Score, an Event Classification and an Impact Score.</p>
      <div className="row between" style={{ marginBottom: 18 }}>
        <div className="row">
          {FILTERS.map(([label, ev]) => (
            <button key={label} className={`filter ${filter === ev ? "on" : ""}`} onClick={() => { setFilter(ev); setLimit(30); }}>{label}</button>
          ))}
        </div>
        <label className="row" style={{ fontWeight: 600 }}>
          <input type="checkbox" checked={only7} onChange={(e) => setOnly7(e.target.checked)} style={{ width: 20, height: 20, accentColor: "var(--blue)" }} />
          Impact Score 7+ only
        </label>
      </div>
      <div className="grid main-side">
        <div className="stack" aria-live="polite">
          {shown.length === 0 && <div className="card empty">No stories match. In demo mode recorded news replays offline; the first stories appear once the models have loaded.</div>}
          {shown.slice(0, limit).map(({ main, all: ents }) => {
            const names = [...new Set(ents.map((e) => e.name))];
            return (
              <article key={main.doc} className="story">
                <div className="row between">
                  <div className="row small">
                    <span className="pill">{outletName(main.outlet, main.src)}</span>
                    <span className="pill">{langName(main.lang)}</span>
                    {main.type === "social" && <span className="pill blue">Social post</span>}
                    <span className="muted">{whenLabel(main, data.view)}</span>
                  </div>
                  <span className={`pill ${eventTone(main.ev)}`}>{eventShort(main.ev)}</span>
                </div>
                <div className="title">
                  {main.url ? (
                    <a href={main.url} target="_blank" rel="noreferrer" style={{ color: "inherit", fontWeight: 700, textDecoration: "none" }}><Title text={main.title} names={tickerNames(ents)} /></a>
                  ) : <Title text={main.title} names={tickerNames(ents)} />}
                </div>
                <div className="row small" style={{ gap: 26 }}>
                  <span>Sentiment Score <SentimentText score={main.s} /></span>
                  <span title={main.cap ? `Capped at 5: ${main.cap}` : undefined}>
                    <ImpactDots score={main.imp} low={main.lo} high={main.hi} />{main.cap && <span className="faint"> · capped</span>}
                  </span>
                  <span className="muted">Affects: {names.slice(0, 4).join(", ")}{names.length > 4 ? ` +${names.length - 4}` : ""}</span>
                </div>
              </article>
            );
          })}
          {shown.length > limit && <button className="btn" onClick={() => setLimit(limit + 30)}>Show more ({shown.length - limit} more stories)</button>}
        </div>
        <div className="stack">
          <Card title="Most talked about">
            {talked.map((t) => (
              <div key={t.name} className="kv" style={{ alignItems: "center", fontSize: 16 }}>
                <span>{t.name}</span>
                <span className={`pill ${t.sum / t.n < -0.05 ? "orange" : "blue"}`} title={`average Sentiment Score ${(t.sum / t.n).toFixed(2)}`}>{t.docs.size.toLocaleString()} stories</span>
              </div>
            ))}
          </Card>
          <Card title="Where the news comes from">
            {groups.map((g) => (
              <div key={g.label}>
                <div className="kv" style={{ alignItems: "center", fontSize: 16 }}>
                  <span>{g.label}</span>
                  <span className={`pill ${g.status[1]}`}>{g.status[0]}</span>
                </div>
                {g.label === "Social media" && promo?.total > 0 && (
                  <div className="faint small" style={{ marginTop: -4 }} title={Object.entries<number>(promo.by_source).map(([k, v]) => `${k}: ${v}`).join(", ")}>
                    {promo.total.toLocaleString()} promotional posts filtered today
                  </div>
                )}
              </div>
            ))}
            {data.view === "live" && <button className="link small" style={{ marginTop: 6 }} onClick={() => go("system")}>Source details and failover test →</button>}
          </Card>
          <section className="card soft">
            <h2 style={{ fontSize: 17 }}>How to read a story</h2>
            <p style={{ margin: 0 }}>
              <b>Sentiment Score</b> runs from −1.0 (negative) to +1.0 (positive). <b>Impact Score</b> from 1 to 10 is how much the market could move,
              learned from how prices reacted to past news. Stories in other languages are read in their own language.
            </p>
          </section>
        </div>
      </div>
      <p className="footnote">{data.view === "replay" ? "Recorded GDELT news of 24 Feb 2022 (00:00–12:00 UTC), scored by the real models." : "Live values from the risk engine."}</p>
    </main>
  );
}

/** Ticker -> name for the companies linked in this story (e.g. AAPL -> Apple). */
function tickerNames(ents: { eid: string; name: string }[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const e of ents) if (/^[A-Z0-9.-]+$/.test(e.eid)) out[e.eid.split(".")[0]] = e.name;
  return out;
}

