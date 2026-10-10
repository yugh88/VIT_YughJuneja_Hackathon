// Formatting helpers: every number on screen carries a unit.

export const usdM = (v: number | null | undefined, digits = 1): string =>
  v == null
    ? "–"
    : `${v < 0 ? "−" : ""}$${Math.abs(v / 1e6).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits })}m`;

export const usdBn = (v: number | null | undefined, digits = 2): string =>
  v == null ? "–" : `$${(v / 1e9).toFixed(digits)}bn`;

export const pct = (v: number | null | undefined, digits = 1): string =>
  v == null ? "–" : `${(v * 100).toFixed(digits)}%`;

export const bps = (v: number | null | undefined): string => (v == null ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(0)} bp`);

export const num = (v: number | null | undefined, digits = 3): string =>
  v == null || Number.isNaN(v) ? "–" : v.toFixed(digits);

export const ci = (c?: number[] | null, digits = 2): string =>
  c && c.length === 2 ? `[${c[0].toFixed(digits)}, ${c[1].toFixed(digits)}]` : "";

export const timeUtc = (iso: string): string => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toISOString().slice(11, 16) + " UTC";
};

export const cssVar = (name: string): string =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";

export const ASSET_LABELS: Record<string, string> = {
  term_loan: "Term loans",
  revolver: "Revolvers",
  bond_ig: "IG bonds",
  bond_hy: "HY bonds",
  sovereign_bond: "Sovereigns",
  irs: "Swaps",
  fx_forward: "FX forwards",
  cds: "CDS",
  equity_option: "Equity options",
};
