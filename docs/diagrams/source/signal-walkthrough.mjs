const C = { blue: "#a5d8ff", green: "#b2f2bb", yellow: "#ffec99", red: "#ffc9c9", violet: "#d0bfff", grey: "#e9ecef", teal: "#96f2d7" };
const st = (id, x, text, fill = C.green) => ({ id, x, y: 160, w: 200, h: 175, size: 17, fill, text });
export default {
  name: "signal-walkthrough", W: 1700, H: 790,
  title: "One headline through the engine (real output)",
  subtitle: "POST /v1/analyze on a Hindi headline · values below are exactly what the shipped models returned",
  nodes: [
    { id: "in", x: 30, y: 160, w: 270, h: 175, size: 17, fill: C.blue, text: "**Input (any language)\nटाटा मोटर्स का\nमुनाफा दोगुना हुआ\n~~“Tata Motors' profit doubled”" },
    st("lang", 340, "**Language ID\nHindi (hi)\n~~fastText · conf 0.96"),
    st("ent", 580, "**Entity link\nटाटा मोटर्स\n→ TATAMOTORS.NS\n~~India · Cons. Discretionary"),
    st("sen", 820, "**Sentiment Score\n+0.80\n~~positive · conf 0.93\n~~XLM-R multilingual", C.yellow),
    st("evt", 1060, "**Event class\nEarnings /\nGuidance\n~~conf 0.99", C.yellow),
    st("imp", 1300, "**Impact Score\n8 / 10\n~~likely range 4 – 8\n~~raised by: strong sentiment,\n~~confident event", C.yellow),
    { id: "json", x: 30, y: 430, w: 1000, h: 300, size: 16, align: "left", fill: "#f8f9fa",
      text: "**Structured output (RiskSignal, schema 1.0) → API, WebSocket, signals.jsonl\n{ \"entity\": {\"id\": \"TATAMOTORS.NS\", \"country\": \"IN\", \"sector\": \"Consumer Discretionary\"},\n  \"sentiment\": {\"score\": 0.797, \"label\": \"positive\", \"confidence\": 0.929},\n  \"event\": {\"primary\": \"Earnings/Guidance\", \"confidence\": 0.988},\n  \"impact\": {\"score\": 8, \"low\": 4, \"high\": 8, \"mode\": \"model\",\n             \"effects\": [{\"feature\": \"abs_sentiment\", \"effect\": +0.028}, …]},\n  \"language\": \"hi\", \"novelty\": 1.0, \"syndication_count\": 1,\n  \"model_versions\": {\"sentiment\": \"sentiment-multi-1.0\", \"impact\": \"hgb-impact-20261005\"} }" },
    { id: "ma", x: 1110, y: 430, w: 560, h: 95, size: 16, fill: C.blue, text: "**Module A · index rebalancer\n~~only the 20 S&P 100 index stocks are tilted → not used here" },
    { id: "mb", x: 1110, y: 548, w: 560, h: 95, size: 16, fill: C.red, text: "**Module B · stress test\n~~Earnings is not a trigger class (Geo / Macro / Credit, impact > 7) → no test" },
    { id: "ui", x: 1110, y: 666, w: 560, h: 95, size: 16, fill: C.violet, text: "**Dashboard\n~~shows on Live news + Try it, with the words that drove the score" },
  ],
  edges: [
    { from: "in", to: "lang" }, { from: "lang", to: "ent" }, { from: "ent", to: "sen" },
    { from: "sen", to: "evt" }, { from: "evt", to: "imp" },
    { from: "imp", to: "json", fs: "b", ts: "t", mid: 385 },
    { from: "json", to: "ma", fs: "r", ts: "l", mid: 1070 },
    { from: "json", to: "mb", fs: "r", ts: "l", mid: 1070 },
    { from: "json", to: "ui", fs: "r", ts: "l", mid: 1070 },
  ],
  texts: [{ x: 1600, y: 248, size: 15, color: "#495057", text: "yellow =\nthe 3\nrequired\nfields" }],
};
