const C = { blue: "#a5d8ff", green: "#b2f2bb", yellow: "#ffec99", red: "#ffc9c9", violet: "#d0bfff", orange: "#ffd8a8", grey: "#e9ecef", teal: "#96f2d7" };

const stages = [
  ["1 · Normalise", "Unicode NFKC · strip URLs, handles, outlet tags"],
  ["2 · Language ID", "fastText lid.176 · 176 languages"],
  ["3 · Quality filter", "drop promo spam · short-text language rule"],
  ["4 · De-duplicate", "MinHash LSH, Jaccard ≥ 0.8 → syndication count"],
  ["5 · Entity linking", "284 entities · 9.7k aliases · 20+ languages"],
  ["6 · Sentiment Score  −1 … +1", "FinBERT (English) · XLM-R (other languages)"],
  ["7 · Event Classification (12 classes)", "XLM-R + zero-shot NLI · 8-K / CAMEO overrides"],
  ["8 · Impact Score  1 … 10", "gradient boosting, learned from market reactions"],
];
const S0 = 160, SH = 64, SG = 14;
const stageNodes = stages.map(([a, b], i) => ({
  id: `s${i + 1}`, x: 665, y: S0 + i * (SH + SG), w: 385, h: SH, size: 16,
  text: `**${a}\n~~${b}`, fill: i >= 5 ? C.yellow : C.green,
}));
const stageEdges = stages.slice(1).map((_, i) => ({ from: `s${i + 1}`, to: `s${i + 2}`, fs: "b", ts: "t" }));

export default {
  name: "architecture", W: 1700, H: 900,
  title: "Sentinel · system architecture",
  subtitle: "keyless news + social text  →  AI/NLP Risk Engine  →  structured risk signals  →  two downstream modules + dashboard",
  frames: [
    { x: 30, y: 110, w: 280, h: 725, label: "① Sources (all keyless)" },
    { x: 345, y: 110, w: 260, h: 725, label: "② Ingestion" },
    { x: 640, y: 110, w: 435, h: 725, label: "③ AI / NLP Risk Engine (core)", color: "#2b8a3e" },
    { x: 1110, y: 110, w: 260, h: 725, label: "④ Signal output" },
    { x: 1410, y: 110, w: 270, h: 725, label: "⑤ Consumers" },
  ],
  nodes: [
    { id: "news", x: 50, y: 165, w: 240, h: 115, size: 16, fill: C.blue, text: "**📰 News\nGDELT 2.0 · 65 languages\nGoogle News · Yahoo\nEconomic Times · Mint" },
    { id: "fil", x: 50, y: 310, w: 240, h: 115, size: 16, fill: C.blue, text: "**🏛 Filings + central banks\nSEC EDGAR 8-K feed\nFed · ECB · RBI releases" },
    { id: "soc", x: 50, y: 455, w: 240, h: 115, size: 16, fill: C.blue, text: "**💬 Social\nMastodon · Reddit\nHacker News\nfinance tweets (HF)" },
    { id: "rep", x: 50, y: 640, w: 240, h: 115, size: 16, fill: C.grey, dashed: true, text: "**⏺ Replay twins\nrecorded .jsonl.gz\nper source → the demo\nruns fully offline" },
    { id: "con", x: 365, y: 240, w: 220, h: 110, size: 16, fill: C.violet, text: "**Connectors\n9 sources, async polling\nGDELT every 15 min" },
    { id: "brk", x: 365, y: 420, w: 220, h: 120, size: 16, fill: C.violet, text: "**Circuit breaker\nback-off 1 s → 60 s\nfails over to the\nreplay twin" },
    { id: "raw", x: 365, y: 620, w: 220, h: 95, size: 16, fill: C.violet, text: "**Raw document bus\n/v1/raw · WebSocket" },
    ...stageNodes,
    { id: "sig", x: 1130, y: 165, w: 220, h: 150, size: 16, fill: C.yellow, text: "**RiskSignal (JSON)\nentity · Sentiment Score\nEvent class · Impact Score\n+ range, evidence,\nconfidence, model versions" },
    { id: "agg", x: 1130, y: 360, w: 220, h: 110, size: 16, fill: C.teal, text: "**Per-entity score S(t)\ntime-decayed mean\nnews 12 h · social 2 h" },
    { id: "api", x: 1130, y: 515, w: 220, h: 110, size: 16, fill: C.teal, text: "**FastAPI\nREST /v1/signals …\nWebSocket /v1/stream" },
    { id: "fil2", x: 1130, y: 680, w: 220, h: 110, size: 16, fill: C.teal, shape: "cyl", text: "**Files\nsignals.jsonl\nhourly Parquet" },
    { id: "ma", x: 1430, y: 165, w: 230, h: 135, size: 16, fill: C.blue, text: "**Module A\nIndex rebalancer\n20 S&P 100 stocks\n~~subscribes: Sentiment" },
    { id: "mb", x: 1430, y: 340, w: 230, h: 135, size: 16, fill: C.red, text: "**Module B\nStress test\n$10 bn synthetic book\n~~subscribes: Event + Impact" },
    { id: "ui", x: 1430, y: 515, w: 230, h: 120, size: 16, fill: C.violet, text: "**React dashboard\n6 screens + 5-min\nguided tour, offline" },
    { id: "ext", x: 1430, y: 690, w: 230, h: 95, size: 16, fill: C.grey, text: "**Any other consumer\nDuckDB · notebooks · BI" },
  ],
  edges: [
    { from: "news", to: "con", mid: 325 },
    { from: "fil", to: "con", mid: 325 },
    { from: "soc", to: "con", mid: 325 },
    { from: "rep", to: "raw", mid: 328, dashed: true, label: "replay: offline or on failure", lx: 175, ly: 790, size: 14 },
    { from: "con", to: "brk", fs: "b", ts: "t" },
    { from: "brk", to: "raw", fs: "b", ts: "t" },
    { from: "raw", to: "s1", mid: 622 },
    ...stageEdges,
    { from: "s8", to: "sig", mid: 1090 },
    { from: "sig", to: "agg", fs: "b", ts: "t" },
    { from: "agg", to: "api", fs: "b", ts: "t" },
    { from: "api", to: "fil2", fs: "b", ts: "t", label: "every signal appended", size: 14 },
    { from: "agg", to: "ma", mid: 1380, label: "S(t)", lx: 1380, ly: 330, size: 14 },
    { from: "api", to: "mb", fs: "r30", ts: "l", mid: 1396 },
    { from: "api", to: "ui", fs: "r70", ts: "l" },
    { from: "fil2", to: "ext" },
  ],
  texts: [
    { x: 857, y: 800, size: 15, color: "#495057", text: "green = text processing · yellow = the 3 required outputs\nONNX Runtime on CPU · every model stage has a fallback\n(lexicon / keyword rules / heuristic impact)" },
    { x: 850, y: 868, size: 17, color: "#343a40", text: "Python 3.11 · FastAPI · ONNX Runtime · Hugging Face fine-tunes · scikit-learn · DuckDB/Parquet · React + ECharts · Docker" },
  ],
};
