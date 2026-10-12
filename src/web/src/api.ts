// Thin client for the Sentinel API. Every call is same-origin (served by FastAPI), so the
// dashboard needs no internet: in replay/demo mode all data comes from recorded sources.

export async function get<T = any>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
}

export async function post<T = any>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
}

export function wsUrl(path: string): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}${path}`;
}

// Cross-screen actions (used by the Demo Mode tour to drive screens).
export type Action = { type: "replay-2022" | "analyze-example" | "kill-source" | "custom-scenario"; arg?: string };

export function dispatch(a: Action): void {
  window.dispatchEvent(new CustomEvent<Action>("sentinel:action", { detail: a }));
}

export function onAction(fn: (a: Action) => void): () => void {
  const h = (e: Event) => fn((e as CustomEvent<Action>).detail);
  window.addEventListener("sentinel:action", h);
  return () => window.removeEventListener("sentinel:action", h);
}

// ---- shapes used across screens (subset of the backend schema) ----
export interface Signal {
  signal_id: string;
  doc_id?: string;
  published_at: string;
  language: string;
  source: { name: string; outlet: string; type: string; credibility: number };
  entity: { id: string; name: string; country?: string | null; sector?: string | null };
  sentiment: { score: number; score_human?: number | null; label: string; confidence: number };
  event: { primary: string; confidence: number; secondary: string[] };
  impact: { score: number; low: number; high: number; drivers: string[]; mode: string };
  evidence: { title: string; title_en?: string; sentence: string; url: string };
  syndication_count: number;
  novelty: number;
  attention?: { burst_z: number; outlets_window: number } | null;
}

export interface StressResult {
  id: string;
  cluster: string;
  scenario: { name: string; event: string; severity: number; country?: string | null; sector?: string | null; factors: Record<string, number> };
  book_value_before: number;
  book_value_after: number;
  loss: number;
  mtm_pnl: number;
  delta_expected_loss: number;
  waterfall: Record<string, number>;
  rwa_before: number;
  rwa_after: number;
  delta_rwa: number;
  cet1_ratio_before: number;
  cet1_ratio_after: number;
  delta_cet1_bps: number;
  delta_cet1_from_losses_bps: number;
  delta_cet1_from_rwa_bps: number;
  cet1_capital_before?: number;
  cet1_capital_after?: number;
  loss_pre_tax?: number;
  loss_after_tax?: number;
  tax_rate?: number;
  rwa_after_uncapped?: number;
  rwa_growth_cap?: number;
  rwa_cap_binds?: boolean;
  top_losers: { name: string; pnl: number }[];
  heatmap: { countries: string[]; sectors: string[]; pnl: number[][] };
  monte_carlo: { draws: number; horizon_days: number; var_99: number; es_97_5: number; mean_loss: number; histogram_musd: { counts: number[]; edges: number[] } };
  corroboration?: { outlets: number; max_burst_z: number; tier1_outlets?: string[]; score?: number } | null;
  trigger?: { title: string; url: string; entity: string; event: string; impact: number; published_at: string } | null;
}

export interface Alert {
  id?: string;
  cluster: string;
  event: string;
  first_utc: string;
  last_utc?: string;
  merged_signals: number;
  severity: number;
  headline: string;
  impact: number;
  url?: string;
  entity?: string;
  result_id: string | null;
  loss?: number;
  delta_cet1_bps?: number;
  corroboration?: { outlets: number; max_burst_z: number; tier1_outlets?: string[]; score?: number };
}
