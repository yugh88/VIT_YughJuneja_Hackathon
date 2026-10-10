import type { ReactNode } from "react";

export function Kpi({ label, value, sub, dark, tip, tone }: { label: string; value: ReactNode; sub?: ReactNode; dark?: boolean; tip: string; tone?: "pos" | "neg" }) {
  return (
    <div className={`card kpi ${dark ? "dark" : ""}`} title={tip}>
      <div className="lbl">{label}</div>
      <div className={`val ${tone ?? ""}`}>{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export function Card({ title, right, children, className = "", tip }: { title?: ReactNode; right?: ReactNode; children: ReactNode; className?: string; tip?: string }) {
  return (
    <section className={`card ${className}`} title={tip}>
      {(title || right) && (
        <div className="row between" style={{ marginBottom: 10 }}>
          {title && <h2 style={{ margin: 0 }}>{title}</h2>}
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

/** Impact badge (9 orange, 8 amber-brown, 7 and below slate), as in the designs. */
export function ImpactBadge({ score }: { score: number }) {
  const cls = score >= 9 ? "i9" : score >= 8 ? "i8" : "i7";
  return (
    <div className={`badge ${cls}`} title={`Impact Score ${score} of 10`}>
      {score}
      <small>IMPACT</small>
    </div>
  );
}

/** Ten squares, filled up to the impact score. */
export function ImpactDots({ score, low, high }: { score: number; low?: number; high?: number }) {
  const colour = score >= 8 ? "var(--orange-dark)" : score >= 6 ? "#475569" : "var(--blue)";
  return (
    <span className="row" style={{ gap: 8 }} title={`Impact Score ${score}/10${low != null ? `; historically, news scored ${score} moved prices like a ${low}-${high}` : ""}`}>
      <span>Impact Score <b>{score}/10</b></span>
      <span className="dots">{Array.from({ length: 10 }, (_, i) => <i key={i} style={i < score ? { background: colour } : undefined} />)}</span>
    </span>
  );
}

export function SentimentText({ score, label = true }: { score: number; label?: boolean }) {
  const word = score >= 0.05 ? "Positive" : score <= -0.05 ? "Negative" : "Neutral";
  const cls = score >= 0.05 ? "pos" : score <= -0.05 ? "neg" : "muted";
  return (
    <b className={cls} title="Sentiment Score from −1.0 (negative) to +1.0 (positive)">
      {score >= 0 ? "+" : "−"}{Math.abs(score).toFixed(2)}{label ? ` ${word}` : ""}
    </b>
  );
}

/** Centred diverging bar for a sentiment score in [−1, 1]. */
export function SentimentBar({ score, width = 160 }: { score: number; width?: number }) {
  const w = Math.min(1, Math.abs(score)) * 50;
  return (
    <span className="track" style={{ width, display: "inline-block", height: 8 }} aria-hidden="true">
      <span style={{ left: "50%", width: 2, background: "#cbd5e1" }} />
      <span style={{ left: `${score >= 0 ? 50 : 50 - w}%`, width: `${w}%`, background: score >= 0 ? "var(--blue)" : "var(--orange)" }} />
    </span>
  );
}

export function Bar({ value, max, colour }: { value: number; max: number; colour: string }) {
  return (
    <div className="track" style={{ height: 12 }}>
      <span style={{ left: 0, width: `${Math.max(2, (Math.abs(value) / (max || 1)) * 100)}%`, background: colour }} />
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function Skeleton({ h = 120 }: { h?: number }) {
  return <div className="skeleton" style={{ height: h }} aria-busy="true" />;
}

/* Kept for the analyst pages (report, sources). */
export function Panel({ title, children, right, tip }: { title?: string; children: ReactNode; right?: ReactNode; tip?: string }) {
  return (
    <section className="panel">
      {(title || right) && (
        <div className="row between" style={{ marginBottom: 8 }}>
          {title && <h3 title={tip}>{title}</h3>}
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

export function Status({ met, label }: { met: boolean | null; label?: string }) {
  if (met === null) return <span className="chip warn">{label ?? "pending"}</span>;
  return <span className={`chip ${met ? "ok" : "bad"}`}>{label ?? (met ? "met" : "missed")}</span>;
}
