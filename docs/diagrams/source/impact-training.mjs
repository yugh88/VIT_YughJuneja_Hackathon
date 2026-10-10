const C = { blue: "#a5d8ff", green: "#b2f2bb", yellow: "#ffec99", red: "#ffc9c9", violet: "#d0bfff", grey: "#e9ecef", teal: "#96f2d7", orange: "#ffd8a8" };
const B = (id, x, text, fill) => ({ id, x, y: 150, w: 300, h: 180, fill, size: 16, text });
export default {
  name: "impact-training", W: 1700, H: 750,
  title: "How the Impact Score is learned (not hand-written)",
  subtitle: "label = how much the stock or market actually moved after past news · strict time split, no look-ahead",
  nodes: [
    B("g", 30, "**Historical news\nGDELT, 2,000 random\n15-min windows\n2019 – 2026", C.blue),
    B("p", 365, "**Score with the shipped\npipeline\n74,964 labelled signals\n~~(entity, headline, time)", C.green),
    B("l", 700, "**Label from prices\n(yfinance)\nabnormal return, next 2\ntrading days ÷ own 60-day σ\n~~market-model β over 120 days", C.yellow),
    B("f", 1035, "**25 features\nevent · |sentiment| · burst\nsector · $ volume · tier\nearnings window · syndication\n~~all known at publish time", C.violet),
    B("m", 1370, "**Gradient boosting\n→ Impact Score 1 – 10\n+ likely range (IQR)\n+ top drivers", C.orange),
    { id: "tr", x: 30, y: 400, w: 920, h: 80, fill: C.teal, size: 18, text: "**Train 2019 – 2023 · 44,811 rows\n~~hyper-parameters tuned on 2023 only" },
    { id: "te", x: 960, y: 400, w: 710, h: 80, fill: C.red, size: 18, text: "**Test 2024 → 2026 · 30,153 rows · touched once" },
    { id: "r", x: 30, y: 530, w: 1640, h: 185, align: "left", fill: "#f8f9fa", size: 17, text: "**Out-of-sample results (2024+)\n• Spearman ρ 0.115 (95% CI 0.104–0.127, p ≈ 1e-89) vs 0.043 for a hand-written heuristic → 2.7× better ranking\n• Score ≥ 8 lands in the top decile of real moves 1.61× as often as chance (16.8% vs 10.4%)\n• Score 10 → mean move 2.61σ vs 1.13σ at score 1 (2.3×); scores 1–8 are nearly flat\n• Honest caveat: much of ρ is the earnings calendar (0.083 without it, 0.030 outside earnings windows); target ρ ≥ 0.25 missed" },
  ],
  edges: [
    { from: "g", to: "p" }, { from: "p", to: "l" }, { from: "l", to: "f" }, { from: "f", to: "m" },
  ],
  texts: [{ x: 960, y: 390, size: 14, color: "#c92a2a", text: "split by time ↓" }],
};
