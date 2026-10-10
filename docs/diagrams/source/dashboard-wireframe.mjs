const C = { blue: "#a5d8ff", green: "#b2f2bb", yellow: "#ffec99", red: "#ffc9c9", violet: "#d0bfff", grey: "#e9ecef", teal: "#96f2d7", orange: "#ffd8a8", navy: "#1c3d5a" };
const PW = 530, PH = 390, G = 25, X0 = 30, Y0 = 130;
const nodes = [], frames = [];
// panel: [title, subtitle, [ [relx, rely, w, h, text, fill], ... ] ] (relative to the panel body)
const panels = [
  ["Home", "is risk building anywhere right now?", [
    [0, 0, 120, 60, "**stories\n~~read", C.grey], [125, 0, 120, 60, "**risk\n~~signals", C.grey], [250, 0, 120, 60, "**languages", C.grey], [375, 0, 115, 60, "**stress\n~~tests", C.grey],
    [0, 70, 300, 170, "**world map\n~~hover a country →\n~~risk level, headlines,\n~~stress loss", C.blue],
    [310, 70, 180, 80, "**overall risk\n~~Calm / Watch / High", C.orange], [310, 160, 180, 80, "**alerts that\n~~need attention", C.red],
    [0, 250, 240, 60, "**sentiment by company", C.yellow], [250, 250, 240, 60, "**event mix", C.yellow]]],
  ["Live news", "every story, scored", [
    [0, 0, 490, 45, "~~filters: event · Impact 7+ · source · language", C.grey],
    [0, 55, 330, 60, "**headline · source · lang\n~~Sentiment +0.62 · Impact 8 · AAPL", C.white],
    [0, 125, 330, 60, "**headline · source · lang\n~~Sentiment −0.71 · Impact 9 · RU", C.white],
    [0, 195, 330, 60, "**…  (replays show original date)", C.white],
    [340, 55, 150, 120, "**most\n~~talked-about\n~~names", C.yellow], [340, 185, 150, 125, "**source status\n~~+ spam posts\n~~filtered today", C.grey],
    [0, 265, 330, 45, "~~promotional posts dropped and counted", C.grey]]],
  ["Stress test (Module B)", "what would this event cost the bank?", [
    [0, 0, 150, 310, "**alert queue\n~~most serious\n~~first\n\n~~impact ×\n~~corroboration", C.red],
    [160, 0, 330, 120, "**portfolio value\n~~BEFORE  $10.0 bn  →  AFTER\n~~(bar chart, loss highlighted)", C.blue],
    [160, 130, 160, 80, "**ΔCET1 vs\n~~7.0% minimum", C.orange], [330, 130, 160, 80, "**99% VaR\n~~97.5% ES", C.orange],
    [160, 220, 160, 90, "**loss by\n~~asset class", C.yellow], [330, 220, 160, 90, "**replay 24 Feb\n~~2022 · build your\n~~own scenario", C.violet]]],
  ["Index rebalancer (Module A)", "how is sentiment moving the weights?", [
    [0, 0, 490, 170, "**stacked weights over time (20 stocks)\n~~today · last 7 days · last 30 backtest days", C.blue],
    [0, 180, 160, 130, "**tilt strength\n~~κ slider\n~~pause / resume", C.grey],
    [170, 180, 160, 130, "**largest weight\n~~changes + the\n~~signal behind", C.yellow],
    [340, 180, 150, 130, "**verdict\n~~risk limits ✓\n~~no edge after\n~~costs (stated)", C.red]]],
  ["Try it", "type any headline or ask about a company", [
    [0, 0, 490, 55, "~~“tata ki recent news btao” · “Apple unveils new iPhone” · any language", C.grey],
    [0, 65, 160, 120, "**Sentiment\n~~gauge −1 … +1", C.yellow], [170, 65, 160, 120, "**Event\n~~class\n~~probabilities", C.yellow], [340, 65, 150, 120, "**Impact\n~~8 / 10\n~~range 4–8", C.yellow],
    [0, 195, 240, 115, "**words that drove\n~~the sentiment\n~~(highlighted)", C.green], [250, 195, 240, 115, "**plain reasons for\n~~the impact score\n~~+ raw JSON download", C.green]]],
  ["Track record", "can you trust it?", [
    [0, 0, 240, 95, "**credit detection\n~~recall 100% · precision 67%", C.green], [250, 0, 240, 95, "**bigger moves after\n~~higher scores (2.3×)", C.green],
    [0, 105, 240, 95, "**reading quality\n~~vs targets", C.yellow], [250, 105, 240, 95, "**how we tested\n~~time splits, frozen sets", C.grey],
    [0, 210, 490, 100, "**where it falls short\n~~misses listed plainly: latency, multilingual F1, impact ρ, index edge", C.red]]],
];
panels.forEach(([title, sub, items], i) => {
  const px = X0 + (i % 3) * (PW + G), py = Y0 + Math.floor(i / 3) * (PH + G);
  nodes.push({ id: `p${i}`, x: px, y: py, w: PW, h: PH, fill: "#ffffff", rough: 0.6, text: "" });
  nodes.push({ id: `h${i}`, x: px, y: py, w: PW, h: 52, fill: "#d0ebff", rough: 0.6, size: 17, text: `**${i + 1} · ${title}\n~~${sub}` });
  items.forEach(([rx, ry, w, h, text, fill], j) => nodes.push({ id: `p${i}w${j}`, x: px + 20 + rx, y: py + 65 + ry, w, h, fill, size: 14, rough: 0.8, text }));
});
export default {
  name: "dashboard-wireframe", W: 1700, H: 990,
  title: "Dashboard wireframe · 6 screens",
  subtitle: "React + ECharts, served by the API at / · works offline · header toggles Live ↔ Replay of 24 Feb 2022 · 5-min guided tour",
  nodes,
  texts: [{ x: 850, y: 965, size: 16, color: "#495057", text: "screens map to the brief: 2 + 5 = engine output · 4 = Module A weights over time · 3 = Module B value before vs after · 6 = evaluation" }],
};
