import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { get, type Alert, type Signal, type StressResult } from "./api";

/** One signal in the compact shape shared by the live engine and the recorded 24 Feb 2022 replay. */
export interface S {
  id: string; doc: string; t: string; outlet: string; src: string; type: string; lang: string;
  eid: string; name: string; country?: string | null; sector?: string | null;
  s: number; ev: string; evc: number; imp: number; lo: number; hi: number; drv: string[];
  title: string; sent: string; url: string; synd: number;
  orig?: string | null; // replayed items: the recording's own publish time
  cap?: string | null; // why the Impact Score was capped
}

export type View = "live" | "replay";

export interface Dataset {
  view: View;
  loading: boolean;
  error: string | null;
  signals: S[];
  alerts: Alert[];
  stories: number; // documents read
  signalsTotal: number;
  languages: Record<string, number>;
  stressTests: number;
  sources: any[];
  mode: string;
  updatedAt: number;
  window: [Date, Date]; // time span the data covers
  result: (id: string) => Promise<StressResult | null>;
  replay: any | null; // full replay file (results, timeline) in replay view
}

export const toS = (x: Signal): S => ({
  id: x.signal_id, doc: (x as any).doc_id ?? x.signal_id, t: x.published_at, outlet: x.source.outlet, src: x.source.name,
  type: x.source.type, lang: x.language, eid: x.entity.id, name: x.entity.name, country: x.entity.country,
  sector: x.entity.sector, s: x.sentiment.score, ev: x.event.primary, evc: x.event.confidence, imp: x.impact.score,
  lo: x.impact.low, hi: x.impact.high, drv: x.impact.drivers, title: x.evidence.title || x.evidence.sentence,
  sent: x.evidence.sentence, url: x.evidence.url, synd: x.syndication_count,
  orig: (x as any).original_published_at ?? null, cap: (x.impact as any).cap_reason ?? null,
});

interface Ctx { data: Dataset; view: View; setView: (v: View) => void; health: any; tickers: Record<string, string> }
const DataCtx = createContext<Ctx | null>(null);

export function useData(): Ctx {
  const c = useContext(DataCtx);
  if (!c) throw new Error("useData outside DataProvider");
  return c;
}

export function DataProvider({ children }: { children: ReactNode }) {
  // ?view=replay opens the recorded 24 Feb 2022 morning directly (shareable demo link).
  const [view, setView] = useState<View>(() => (new URLSearchParams(location.search).get("view") === "replay" ? "replay" : "live"));
  const [health, setHealth] = useState<any>(null);
  const [live, setLive] = useState<{ signals: S[]; stress: any } | null>(null);
  const [replay, setReplay] = useState<{ meta: any; signals: S[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState(Date.now());
  const cache = useRef(new Map<string, StressResult>());
  const [tickers, setTickers] = useState<Record<string, string>>({});
  useEffect(() => { get<Record<string, string>>("/v1/tickers").then(setTickers).catch(() => undefined); }, []);

  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const [h, sig, st] = await Promise.all([
          get("/v1/health"),
          get<{ items: Signal[] }>("/v1/signals?limit=500"),
          get("/v1/modules/stress").catch(() => null),
        ]);
        if (stop) return;
        setHealth(h);
        setLive({ signals: sig.items.map(toS), stress: st });
        setUpdatedAt(Date.now());
        setError(null);
      } catch (e) {
        if (!stop) setError(String(e));
      }
    };
    tick();
    const id = window.setInterval(tick, 6000);
    return () => { stop = true; window.clearInterval(id); };
  }, []);

  useEffect(() => {
    if (view !== "replay" || replay) return;
    Promise.all([get("/v1/eval/replay_2022-02-24"), get<S[]>("/v1/eval/replay_2022-02-24_signals")])
      .then(([meta, signals]) => setReplay({ meta, signals }))
      .catch((e) => setError(String(e)));
  }, [view, replay]);

  const result = useCallback(
    async (id: string): Promise<StressResult | null> => {
      if (view === "replay") return replay?.meta?.results?.[id] ?? null;
      if (cache.current.has(id)) return cache.current.get(id)!;
      try {
        const r = await get<StressResult>(`/v1/modules/stress/results/${id}`);
        cache.current.set(id, r);
        return r;
      } catch {
        return null;
      }
    },
    [view, replay],
  );

  const data: Dataset = useMemo(() => {
    if (view === "replay") {
      const m = replay?.meta;
      return {
        view, loading: !replay, error, signals: replay?.signals ?? [], alerts: (m?.alert_queue ?? []) as Alert[],
        stories: m?.docs ?? 0, signalsTotal: m?.signals ?? 0, languages: m?.languages ?? {}, stressTests: m?.stress_tests_fired ?? 0,
        sources: [{ name: "gdelt_gkg", source_type: "news", active_mode: "replay", note: "recorded 24 Feb 2022" }],
        mode: "replay", updatedAt, window: [new Date("2022-02-24T00:00:00Z"), new Date("2022-02-24T12:00:00Z")], result, replay: m ?? null,
      };
    }
    const sig = live?.signals ?? [];
    const times = sig.map((s) => new Date(s.t).getTime());
    const end = new Date(Math.max(Date.now(), ...times));
    const start = new Date(Math.min(end.getTime() - 6 * 3600e3, ...times));
    return {
      view, loading: !live, error, signals: sig, alerts: (live?.stress?.alert_queue ?? []) as Alert[],
      stories: health?.docs_total ?? 0, signalsTotal: health?.pipeline?.signals ?? sig.length, languages: health?.pipeline?.languages ?? {},
      stressTests: live?.stress?.results?.length ?? 0, sources: health?.sources ?? [], mode: health?.mode ?? "",
      updatedAt, window: [start, end], result, replay: null,
    };
  }, [view, live, replay, health, error, updatedAt, result]);

  return <DataCtx.Provider value={{ data, view, setView, health, tickers }}>{children}</DataCtx.Provider>;
}

// ------------------------------------------------------------------ derived views

export const TRIGGER_EVENTS = new Set(["Geopolitical", "Macroeconomic", "Credit Event"]);

/** Group signals into stories (one per document), strongest signal first. */
export function stories(signals: S[]): { main: S; all: S[] }[] {
  const by = new Map<string, S[]>();
  for (const s of signals) {
    const k = s.doc || s.id;
    by.set(k, [...(by.get(k) ?? []), s]);
  }
  return [...by.values()]
    .map((all) => ({ main: [...all].sort((a, b) => b.imp - a.imp || Math.abs(b.s) - Math.abs(a.s))[0], all }))
    .sort((a, b) => b.main.t.localeCompare(a.main.t));
}

export function alertCountry(a: Alert, signals: S[]): string | null {
  const [ev, key] = a.cluster.split("|");
  if (ev !== "Credit Event") return key === "Global" ? null : key;
  return signals.find((s) => s.eid === key)?.country ?? (a.entity?.startsWith("SOV:") ? a.entity.slice(4) : null);
}

/**
 * Risk level (DECISIONS D68, a display summary, not a validated model):
 *   High = a stress-test alert with severity >= 100 in the last 6 h; Elevated = any alert in the
 *   last 6 h; Watch = a Geopolitical/Macroeconomic/Credit signal with impact >= 8 in the last 6 h;
 *   Calm otherwise.
 */
export type Level = "High" | "Elevated" | "Watch" | "Calm";
export function levelAt(at: number, alerts: Alert[], signals: S[], country?: string): Level {
  const lo = at - 6 * 3600e3;
  const inWin = (iso: string) => { const t = new Date(iso).getTime(); return t <= at && t > lo; };
  const al = alerts.filter((a) => inWin(a.first_utc) && (!country || alertCountry(a, signals) === country));
  if (al.some((a) => a.severity >= 100)) return "High";
  if (al.length) return "Elevated";
  if (signals.some((s) => TRIGGER_EVENTS.has(s.ev) && s.imp >= 8 && inWin(s.t) && (!country || s.country === country))) return "Watch";
  return "Calm";
}

/** Risk points per hour: sum over stories of max(impact - 6, 0), Geopolitical/Macro/Credit only. */
export function riskSeries(signals: S[], start: Date, end: Date, stepMin = 30): { t: number; v: number }[] {
  const docs = stories(signals.filter((s) => TRIGGER_EVENTS.has(s.ev))).map((g) => ({ t: new Date(g.main.t).getTime(), p: Math.max(g.main.imp - 6, 0) }));
  const out: { t: number; v: number }[] = [];
  for (let t = start.getTime(); t <= end.getTime() + 1; t += stepMin * 60e3) {
    out.push({ t, v: docs.filter((d) => d.t <= t && d.t > t - 3600e3).reduce((a, d) => a + d.p, 0) });
  }
  return out;
}

export const LEVEL_TONE: Record<Level, string> = { High: "#e8590c", Elevated: "#f59e0b", Watch: "#94a3b8", Calm: "#60a5fa" };

/** Honest time label: replayed items show their original date, never "N min ago". */
export function whenLabel(s: { t: string; orig?: string | null }, view: View): string {
  if (s.orig) {
    const d = new Date(s.orig);
    return `Posted ${d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" })} · replay`;
  }
  if (view === "replay") return `${new Date(s.t).toISOString().slice(11, 16)} UTC · 24 Feb 2022 recording`;
  const sec = (Date.now() - new Date(s.t).getTime()) / 1000;
  if (sec < 90) return "just now";
  if (sec < 3600) return `${Math.round(sec / 60)} min ago`;
  if (sec < 86400) return `${Math.round(sec / 3600)} h ago`;
  return new Date(s.t).toISOString().slice(0, 10);
}

/** True when any source is serving a recorded copy (replay mode or a live source's fallback). */
export function servingRecorded(health: any): boolean {
  const src: any[] = health?.sources ?? [];
  return health?.mode === "replay" || src.some((s) => s.active_mode === "replay" || s.active_mode === "fallback");
}

