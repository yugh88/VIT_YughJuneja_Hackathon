const C = { blue: "#a5d8ff", green: "#b2f2bb", yellow: "#ffec99", red: "#ffc9c9", violet: "#d0bfff", grey: "#e9ecef", teal: "#96f2d7", orange: "#ffd8a8" };
export default {
  name: "module-a-flow", W: 1700, H: 810,
  title: "Module A · tactical index rebalancer",
  subtitle: "subscribes to the Sentiment Score · positive news → more weight, negative news → less weight, inside hard risk limits",
  nodes: [
    { id: "sig", x: 30, y: 150, w: 250, h: 130, fill: C.yellow, text: "**Sentiment signals\nfor the 20 index stocks\n~~(AAPL, MSFT, JPM, XOM …)" },
    { id: "agg", x: 340, y: 150, w: 270, h: 130, fill: C.teal, text: "**Score per stock S\ntime-decayed mean\n~~half-life 12 h news / 2 h social" },
    { id: "sur", x: 670, y: 150, w: 300, h: 130, fill: C.teal, text: "**Surprise (live default v2)\nS minus the stock's own\n30-day mean\n~~only signals with impact ≥ 6" },
    { id: "d1", x: 1040, y: 125, w: 280, h: 180, shape: "diamond", fill: C.orange, size: 16, text: "**Rebalance now?\nimpact ≥ 5 signal\nor 15-min timer" },
    { id: "wait", x: 1400, y: 165, w: 250, h: 100, fill: C.grey, text: "**Keep weights\n~~wait for next signal / tick" },
    { id: "d2", x: 1040, y: 380, w: 280, h: 180, shape: "diamond", fill: C.orange, size: 16, text: "**Moved enough?\n|ΔS| > 0.15\n~~(hysteresis)" },
    { id: "tilt", x: 670, y: 400, w: 300, h: 140, fill: C.green, text: "**Tilt\nw = base × clip(e^(κ·S), 0.5, 2)\n~~base = float-adjusted cap (≤ 10%)\n~~κ = 0.5 (slider on dashboard)" },
    { id: "lim", x: 340, y: 400, w: 270, h: 140, fill: C.red, text: "**Risk limits\n≤ 15% per stock · ≥ 1% floor\n≤ 35% per sector\n≤ 10% turnover per rebalance" },
    { id: "out", x: 30, y: 400, w: 250, h: 140, fill: C.violet, text: "**New weights\n+ rebalance log\n~~names the signal\n~~behind each move" },
    { id: "ui", x: 30, y: 630, w: 250, h: 120, fill: C.violet, text: "**Dashboard\nstacked weights over time\n~~GET /v1/modules/index" },
    { id: "bt", x: 340, y: 610, w: 1310, h: 160, align: "left", fill: "#f8f9fa", size: 17, text: "**Backtest, measured honestly (10 bp per side, no look-ahead, test 2024 → Aug 2026, run once)\n• Two versions written down before running; v2 (surprise) won on 2020–23 validation\n• Test: net IR −1.43 (v2) / −1.31 (v1); beats 33% / 74% of date-shuffled placebos (bar: 95%) → no proven edge\n• Risk limits held on 1,673 of 1,673 rebalance days · turnover 2.2% per day\n~~The rebalancer is a working, controlled consumer of the signal; the dashboard states the verdict plainly." },
  ],
  edges: [
    { from: "sig", to: "agg" }, { from: "agg", to: "sur" }, { from: "sur", to: "d1" },
    { from: "d1", to: "wait", label: "no", size: 16 },
    { from: "d1", to: "d2", fs: "b", ts: "t", label: "yes", size: 16 },
    { from: "d2", to: "wait", fs: "r", ts: "b", label: "no", size: 16, lx: 1525, ly: 420 },
    { from: "d2", to: "tilt", fs: "l", ts: "r", label: "yes", size: 16 },
    { from: "tilt", to: "lim", fs: "l", ts: "r" },
    { from: "lim", to: "out", fs: "l", ts: "r" },
    { from: "out", to: "ui", fs: "b", ts: "t" },
  ],
};
