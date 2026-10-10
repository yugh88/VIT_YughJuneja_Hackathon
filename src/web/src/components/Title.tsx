import { useData } from "../data";

const CASHTAG = /(?<![\w$])\$([A-Za-z][A-Za-z.]{0,9})\b/g;

/**
 * A headline with readable cashtags: "Vaxart ($VXRT)" on first use when the company is known,
 * and only the first two cashtags shown ("+N more" in grey) when a post lists many.
 */
export function Title({ text, names = {} }: { text: string; names?: Record<string, string> }) {
  const { tickers } = useData();
  const seen = new Set<string>();
  let kept = 0;
  let hidden = 0;
  const parts: (string | JSX.Element)[] = [];
  let last = 0;
  for (const m of text.matchAll(CASHTAG)) {
    const tk = m[1].toUpperCase();
    parts.push(text.slice(last, m.index));
    last = (m.index ?? 0) + m[0].length;
    if (seen.has(tk)) { parts.push(m[0]); continue; }
    seen.add(tk);
    if (kept >= 2) { hidden += 1; continue; }
    kept += 1;
    const name = names[tk] ?? tickers[tk];
    parts.push(name ? `${name} (${m[0]})` : m[0]);
  }
  parts.push(text.slice(last));
  const out = parts.join("").replace(/\s{2,}/g, " ").trim();
  return (
    <>
      {out}
      {hidden > 0 && <span className="faint" style={{ fontWeight: 500 }}> +{hidden} more</span>}
    </>
  );
}
