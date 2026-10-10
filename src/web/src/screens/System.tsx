import { useEffect, useState } from "react";
import { get, onAction, post } from "../api";
import type { ReactNode } from "react";
import { Panel, Skeleton } from "../components/ui";

function Kpi({ label, value, note, tip }: { label: string; value: ReactNode; note?: ReactNode; tip: string }) {
  return (
    <div className="kpi" title={tip}>
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {note && <div className="note">{note}</div>}
    </div>
  );
}

export default function System({ health, refresh }: { health: any; refresh: () => void }) {
  const [bench, setBench] = useState<any>(null);
  useEffect(() => {
    get("/v1/eval/bench").then(setBench).catch(() => undefined);
  }, []);
  const act = async (name: string, what: "kill" | "revive") => {
    await post(`/v1/sources/${name}/${what}`).catch(() => undefined);
    refresh();
  };
  useEffect(() => onAction((a) => a.type === "kill-source" && act(a.arg ?? "rss", "kill")), []);

  if (!health) return <Skeleton h={300} />;
  const p = health.pipeline ?? {};
  return (
    <main>
      <h1>Sources and system</h1>
      <p className="sub">
        Mode <b>{health.mode}</b>: every live source has a recorded replay twin; a source whose circuit breaker opens falls back to its twin. Kill a source to watch it happen.
      </p>
      <div className="kpis" style={{ marginBottom: 14 }}>
        <Kpi label="Uptime" value={`${Math.round(health.uptime_s / 60)} min`} tip="Since the API process started" />
        <Kpi label="Docs ingested" value={health.docs_total} tip="All sources, since start" />
        <Kpi label="Pipeline" value={p.status ?? "–"} note={p.modes ? Object.entries(p.modes).map(([k, v]) => `${k}: ${v}`).join(" · ") : undefined} tip="Model modes: fine-tuned ONNX models, or the documented fallbacks" />
        <Kpi label="Duplicates dropped" value={p.duplicate_ratio != null ? `${(p.duplicate_ratio * 100).toFixed(1)}%` : "–"} tip="MinHash near-duplicates removed before any model runs" />
        <Kpi label="Promotional posts filtered" value={(p.promotional_today?.total ?? 0).toLocaleString()} note={`today (UTC) · ${(p.promotional_filtered ?? 0).toLocaleString()} since start`} tip="Social posts dropped before scoring: more than 4 cashtags, cashtags + hashtags over 40% of the words, or a promo phrase (DECISIONS D71)" />
        <Kpi label="Bus lag" value={Object.values<number>(health.bus?.lag ?? {}).reduce((a, b) => a + b, 0)} note="queued messages" tip="In-process queue depth across subscribers" />
      </div>
      <Panel title="Sources">
        <table>
          <thead><tr><th>Source</th><th>Type</th><th>Configured</th><th>Active</th><th>Breaker</th><th className="num">Docs/min</th><th className="num">Total</th><th>Last error</th><th></th></tr></thead>
          <tbody>
            {health.sources.map((s: any) => (
              <tr key={s.name}>
                <td>{s.name}</td>
                <td className="muted">{s.source_type}</td>
                <td className="muted">{s.configured_mode}</td>
                <td><span className={`chip ${s.active_mode === "live" ? "ok" : s.active_mode === "fallback" ? "warn" : s.active_mode === "down" ? "bad" : ""}`}>{s.active_mode}</span></td>
                <td className="muted">{s.breaker ? `${s.breaker.state}${s.breaker.retry_in_s ? ` (retry ${s.breaker.retry_in_s}s)` : ""}` : "–"}</td>
                <td className="num">{s.docs_per_min}</td>
                <td className="num">{s.docs_total}</td>
                <td className="muted" style={{ maxWidth: 260 }}>{s.last_error ?? s.note ?? ""}</td>
                <td className="row">
                  <button onClick={() => act(s.name, "kill")} title="Simulate an outage of this live source">Kill</button>
                  <button onClick={() => act(s.name, "revive")}>Revive</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <Panel title="Measured speed (make bench)" tip="docs/eval/bench.json; the README scale tables use these numbers">
          {bench ? (
            <table><tbody>{Object.entries<string>(bench.summary).map(([k, v]) => <tr key={k}><td className="muted">{k}</td><td>{v}</td></tr>)}</tbody></table>
          ) : (
            <Skeleton h={160} />
          )}
        </Panel>
        <Panel title="Languages seen">
          <div className="row">{Object.entries<number>(p.languages ?? {}).map(([k, v]) => <span key={k} className="chip">{k} {v}</span>)}</div>
          <h3 style={{ marginTop: 12 }}>Architecture</h3>
          <p className="muted" style={{ fontSize: 13 }}>
            sources (live or replay) → in-process bus → NLP pipeline (language ID → normalise → MinHash dedupe → entity linking → sentiment + event ONNX models → learned impact) →
            signal store + WebSocket → aggregator S<sub>e</sub>(t) → Module A rebalancer and Module B stress trigger → this dashboard.
          </p>
        </Panel>
      </div>
    </main>
  );
}
