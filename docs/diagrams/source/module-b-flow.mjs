const C = { blue: "#a5d8ff", green: "#b2f2bb", yellow: "#ffec99", red: "#ffc9c9", violet: "#d0bfff", grey: "#e9ecef", teal: "#96f2d7", orange: "#ffd8a8" };
const D = (id, x, text) => ({ id, x, y: 135, w: 250, h: 190, shape: "diamond", fill: C.orange, size: 15, text });
export default {
  name: "module-b-flow", W: 1700, H: 940,
  title: "Module B · event-driven stress test",
  subtitle: "subscribes to Event Classification + Impact Score · a qualifying event re-prices a $10 bn synthetic wholesale bank book",
  nodes: [
    { id: "sig", x: 30, y: 170, w: 190, h: 120, fill: C.yellow, text: "**New signal\nevent class\n+ impact score" },
    D("d1", 260, "**Trigger class?\nGeopolitical,\nMacroeconomic\nor Credit Event"),
    D("d2", 550, "**High impact?\nImpact Score > 7"),
    D("d3", 840, "**Corroborated?\n≥ 3 outlets or a\nburst within 6 h\n+ 1 tier-1 outlet"),
    D("d4", 1130, "**Credit gate\n(Credit only)\nconf ≥ 0.7 +\ndistress term"),
    { id: "mrg", x: 1420, y: 160, w: 250, h: 140, fill: C.teal, size: 16, text: "**Merge into alert\none test per\n(country or obligor,\nevent) per 6 h" },
    { id: "no", x: 550, y: 370, w: 830, h: 70, fill: C.grey, size: 16, text: "**No stress test — logged (24 Feb 2022: 26 high-impact signals not corroborated, 1 failed the credit gate)" },
    { id: "shk", x: 1420, y: 490, w: 250, h: 200, fill: C.red, size: 15, text: "**Shock set\nmeasured from a real\nanchor episode:\nUkraine 2022 · 2022 rate\nhikes · SVB 2023\n~~scaled by impact, targeted\n~~by country · region · sector" },
    { id: "px", x: 1110, y: 490, w: 270, h: 200, fill: C.red, size: 15, text: "**Re-price 300 positions\nbonds: duration + convexity\nloans: Vasicek-stressed PD\nderivatives: DV01, Δ, Γ, vega\n~~loans · revolvers · IG/HY/sov\n~~bonds · swaps · FX · CDS · options" },
    { id: "cap", x: 800, y: 490, w: 270, h: 200, fill: C.violet, size: 15, text: "**Capital + tail risk\nBasel IRB → ΔRWA (cap +15%)\nΔCET1 after 25% tax\nMonte Carlo 10,000 draws\n99% VaR · 97.5% ES\n~~0.21 s per run" },
    { id: "q", x: 490, y: 490, w: 270, h: 200, fill: C.violet, size: 15, text: "**Alert queue\nranked by severity\n= impact × corroboration\n~~one click opens the\n~~full stress result" },
    { id: "ui", x: 30, y: 490, w: 420, h: 200, fill: C.blue, size: 16, text: "**Dashboard: before vs after\nportfolio value $10.0 bn → after shock\nloss by asset class · CET1 vs 7% minimum\nshocks applied + anchor episode\nlargest exposures · VaR / ES" },
    { id: "ex", x: 30, y: 740, w: 1640, h: 150, align: "left", fill: "#f8f9fa", size: 17, text: "**Measured results (seeded book, real models, real trigger rule)\n• Macroeconomic, impact 8, US: loss $321 m (book $10.0 bn → $9.68 bn), CET1 12.0% → 6.9% (−510 bp), 99% VaR $428 m\n• Geopolitical, impact 9, Russia: loss $72 m, ΔCET1 −236 bp  ·  Credit Event, impact 10, US banks: loss $22 m, ΔEL $30 m  ·  losses after 25% tax, RWA migration capped at +15%\n• Replay of 24 Feb 2022 (00:00–12:00 UTC): 11,520 articles → 3,107 signals → 23 stress tests; 8 of the top 9 alerts are war clusters (Ukraine, Russia, oil > $100)" },
  ],
  edges: [
    { from: "sig", to: "d1" },
    { from: "d1", to: "d2", label: "yes", size: 15 }, { from: "d2", to: "d3", label: "yes", size: 15 },
    { from: "d3", to: "d4", label: "yes", size: 15 }, { from: "d4", to: "mrg", label: "pass", size: 15 },
    { from: "d1", to: "no", fs: "b", ts: "l", label: "no", size: 15, labelSeg: 0, ly: 360 },
    { from: "d2", to: "no", fs: "b", ts: "t15.06", label: "no", size: 15 },
    { from: "d3", to: "no", fs: "b", ts: "t50", label: "no", size: 15 },
    { from: "d4", to: "no", fs: "b", ts: "t84.94", label: "fail", size: 15 },
    { from: "mrg", to: "shk", fs: "b", ts: "t" },
    { from: "shk", to: "px", fs: "l", ts: "r" }, { from: "px", to: "cap", fs: "l", ts: "r" },
    { from: "cap", to: "q", fs: "l", ts: "r" }, { from: "q", to: "ui", fs: "l", ts: "r" },
  ],
};
