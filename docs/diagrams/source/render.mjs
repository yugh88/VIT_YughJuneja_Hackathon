// Hand-drawn (Excalidraw-style) diagram renderer.
// node render.mjs <spec.mjs> <outdir>  ->  <name>.svg, <name>.html (for PNG capture), <name>.excalidraw
import fs from "node:fs";
import path from "node:path";
import rough from "roughjs";

const [, , specPath, outDir] = process.argv;
const { default: spec } = await import(path.resolve(specPath));
const gen = rough.generator();
const FONT_DIR = new URL("./node_modules/@fontsource/kalam/files/", import.meta.url);
const font = (f) => fs.readFileSync(new URL(f, FONT_DIR)).toString("base64");

export const C = {
  blue: "#a5d8ff", green: "#b2f2bb", yellow: "#ffec99", red: "#ffc9c9", violet: "#d0bfff",
  orange: "#ffd8a8", grey: "#e9ecef", teal: "#96f2d7", pink: "#fcc2d7", white: "#ffffff",
};
const STROKE = "#1e1e1e";
let seed = 7;

function port(n, side) {
  const { x, y, w, h } = n;
  switch (side) {
    case "r": return [x + w, y + h / 2];
    case "l": return [x, y + h / 2];
    case "t": return [x + w / 2, y];
    case "b": return [x + w / 2, y + h];
    default: {
      // "r30" style: side + offset percentage along the side
      const s = side[0], p = Number(side.slice(1)) / 100;
      if (s === "r") return [x + w, y + h * p];
      if (s === "l") return [x, y + h * p];
      if (s === "t") return [x + w * p, y];
      return [x + w * p, y + h];
    }
  }
}

function edgePoints(e, nodes) {
  const a = nodes[e.from], b = nodes[e.to];
  const fs_ = e.fs || "r", ts = e.ts || "l";
  const p0 = port(a, fs_), p1 = port(b, ts);
  if (e.via) return [p0, ...e.via, p1];
  const horiz = "rl".includes(fs_[0]) && "rl".includes(ts[0]);
  const vert = "tb".includes(fs_[0]) && "tb".includes(ts[0]);
  if (horiz && Math.abs(p0[1] - p1[1]) > 2) {
    const mx = e.mid ?? (p0[0] + p1[0]) / 2;
    return [p0, [mx, p0[1]], [mx, p1[1]], p1];
  }
  if (vert && Math.abs(p0[0] - p1[0]) > 2) {
    const my = e.mid ?? (p0[1] + p1[1]) / 2;
    return [p0, [p0[0], my], [p1[0], my], p1];
  }
  if ("rl".includes(fs_[0]) && "tb".includes(ts[0])) return [p0, [p1[0], p0[1]], p1];
  if ("tb".includes(fs_[0]) && "rl".includes(ts[0])) return [p0, [p0[0], p1[1]], p1];
  return [p0, p1];
}

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const paths = (d, extra = "") =>
  gen.toPaths(d).map((p) => `<path d="${p.d}" stroke="${p.stroke}" stroke-width="${p.strokeWidth}" fill="${p.fill || "none"}" ${extra}/>`).join("");

function textBlock(lines, cx, cy, size, color = STROKE, anchor = "middle") {
  const lh = size * 1.22;
  const y0 = cy - ((lines.length - 1) * lh) / 2;
  return lines.map((ln, i) => {
    let s = ln, weight = 400, sz = size, fill = color;
    if (s.startsWith("**")) { s = s.slice(2); weight = 700; sz = size * 1.08; }
    if (s.startsWith("~~")) { s = s.slice(2); sz = size * 0.84; fill = "#495057"; }
    return `<text x="${cx}" y="${y0 + i * lh}" font-size="${sz}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}" dominant-baseline="central">${esc(s)}</text>`;
  }).join("");
}

// ---------------------------------------------------------------- SVG
const nodes = Object.fromEntries((spec.nodes || []).map((n) => [n.id, n]));
let svg = "";
for (const f of spec.frames || []) {
  svg += paths(gen.rectangle(f.x, f.y, f.w, f.h, { stroke: f.stroke || "#868e96", strokeWidth: 1.6, roughness: 0.8, seed: seed++, strokeLineDash: [10, 8], fill: f.fill, fillStyle: "solid" }));
  svg += `<text x="${f.x + 14}" y="${f.y + 22}" font-size="${f.size || 20}" font-weight="700" fill="${f.color || "#495057"}">${esc(f.label)}</text>`;
}
for (const e of spec.edges || []) {
  const pts = edgePoints(e, nodes);
  const opt = { stroke: e.color || STROKE, strokeWidth: e.width || 2, roughness: 0.9, seed: seed++, strokeLineDash: e.dashed ? [8, 7] : undefined };
  svg += paths(gen.linearPath(pts, opt));
  const [x1, y1] = pts[pts.length - 2], [x2, y2] = pts[pts.length - 1];
  const ang = Math.atan2(y2 - y1, x2 - x1), L = 14, sp = 0.45;
  for (const s of [-sp, sp]) svg += paths(gen.line(x2, y2, x2 - L * Math.cos(ang + s), y2 - L * Math.sin(ang + s), { ...opt, strokeLineDash: undefined }));
  if (e.label) {
    const i = e.labelSeg ?? Math.floor((pts.length - 1) / 2);
    const [ax, ay] = pts[i], [bx, by] = pts[i + 1];
    const lx = e.lx ?? (ax + bx) / 2, ly = e.ly ?? (ay + by) / 2;
    const lines = e.label.split("\n"), sz = e.size || 15;
    const wpx = Math.max(...lines.map((l) => l.length)) * sz * 0.5 + 12, hpx = lines.length * sz * 1.22 + 6;
    svg += `<rect x="${lx - wpx / 2}" y="${ly - hpx / 2}" width="${wpx}" height="${hpx}" fill="${spec.bg || "#ffffff"}" rx="4"/>`;
    svg += textBlock(lines, lx, ly, sz, e.color || "#343a40");
  }
}
for (const n of spec.nodes || []) {
  const o = { stroke: n.stroke || STROKE, strokeWidth: n.sw || 2, fill: n.fill || C.white, fillStyle: "solid", roughness: n.rough ?? 1, bowing: 1, seed: seed++, strokeLineDash: n.dashed ? [9, 7] : undefined };
  if (n.shape === "diamond") svg += paths(gen.polygon([[n.x + n.w / 2, n.y], [n.x + n.w, n.y + n.h / 2], [n.x + n.w / 2, n.y + n.h], [n.x, n.y + n.h / 2]], o));
  else if (n.shape === "ellipse") svg += paths(gen.ellipse(n.x + n.w / 2, n.y + n.h / 2, n.w, n.h, o));
  else if (n.shape === "cyl") {
    const r = 12;
    svg += paths(gen.path(`M${n.x} ${n.y + r} L${n.x} ${n.y + n.h - r} A${n.w / 2} ${r} 0 0 0 ${n.x + n.w} ${n.y + n.h - r} L${n.x + n.w} ${n.y + r}`, o));
    svg += paths(gen.ellipse(n.x + n.w / 2, n.y + r, n.w, 2 * r, o));
  } else {
    const r = 10, { x, y, w, h } = n;
    svg += paths(gen.path(`M${x + r} ${y} L${x + w - r} ${y} Q${x + w} ${y} ${x + w} ${y + r} L${x + w} ${y + h - r} Q${x + w} ${y + h} ${x + w - r} ${y + h} L${x + r} ${y + h} Q${x} ${y + h} ${x} ${y + h - r} L${x} ${y + r} Q${x} ${y} ${x + r} ${y} Z`, o));
  }
  const lines = n.text.split("\n");
  const cx = n.align === "left" ? n.x + 16 : n.x + n.w / 2;
  svg += textBlock(lines, cx, n.y + n.h / 2, n.size || 17, STROKE, n.align === "left" ? "start" : "middle");
}
for (const t of spec.texts || []) svg += textBlock(t.text.split("\n"), t.x, t.y, t.size || 16, t.color || STROKE, t.anchor || "middle");

const W = spec.W, H = spec.H;
const header = spec.title
  ? `<text x="${W / 2}" y="44" font-size="34" font-weight="700" text-anchor="middle" fill="${STROKE}">${esc(spec.title)}</text>` +
    (spec.subtitle ? `<text x="${W / 2}" y="80" font-size="19" text-anchor="middle" fill="#495057">${esc(spec.subtitle)}</text>` : "")
  : "";
const fontCss = `@font-face{font-family:Hand;font-weight:400;src:url(data:font/woff2;base64,${font("kalam-latin-400-normal.woff2")}) format("woff2");}
@font-face{font-family:Hand;font-weight:700;src:url(data:font/woff2;base64,${font("kalam-latin-700-normal.woff2")}) format("woff2");}
@font-face{font-family:Hand;font-weight:400;unicode-range:U+0900-097F;src:url(data:font/woff2;base64,${font("kalam-devanagari-400-normal.woff2")}) format("woff2");}
@font-face{font-family:Hand;font-weight:700;unicode-range:U+0900-097F;src:url(data:font/woff2;base64,${font("kalam-devanagari-700-normal.woff2")}) format("woff2");}
text{font-family:Hand,"Comic Sans MS",cursive;}`;
const out = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><style>${fontCss}</style><rect width="${W}" height="${H}" fill="${spec.bg || "#ffffff"}"/>${header}${svg}</svg>`;
fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, `${spec.name}.svg`), out);
fs.writeFileSync(path.join(outDir, `${spec.name}.html`), `<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0">${out}</body></html>`);

// ---------------------------------------------------------------- Excalidraw scene (editable source)
let k = 0;
const id = () => `el${++k}`;
const base = (type, x, y, w, h, extra = {}) => ({
  id: id(), type, x, y, width: w, height: h, angle: 0, strokeColor: STROKE, backgroundColor: "transparent",
  fillStyle: "solid", strokeWidth: 2, strokeStyle: "solid", roughness: 1, opacity: 100, groupIds: [], frameId: null,
  roundness: null, seed: 1000 + k, version: 1, versionNonce: 2000 + k, isDeleted: false, boundElements: null,
  updated: 1, link: null, locked: false, ...extra,
});
const clean = (s) => s.replace(/^\*\*|^~~/gm, "");
const xText = (txt, cx, cy, size, align = "center") => {
  const lines = clean(txt).split("\n");
  const w = Math.max(...lines.map((l) => l.length)) * size * 0.55, h = lines.length * size * 1.25;
  return base("text", align === "left" ? cx : cx - w / 2, cy - h / 2, w, h, {
    text: clean(txt), originalText: clean(txt), fontSize: size, fontFamily: 1, textAlign: align, verticalAlign: "middle",
    containerId: null, lineHeight: 1.25, baseline: size, autoResize: true,
  });
};
const els = [];
if (spec.title) els.push(xText(spec.title, W / 2, 40, 32));
if (spec.subtitle) els.push(xText(spec.subtitle, W / 2, 78, 18));
for (const f of spec.frames || []) {
  els.push(base("rectangle", f.x, f.y, f.w, f.h, { strokeColor: "#868e96", strokeStyle: "dashed", backgroundColor: f.fill || "transparent" }));
  els.push(xText(f.label, f.x + 14, f.y + 20, 20, "left"));
}
for (const n of spec.nodes || []) {
  const type = n.shape === "diamond" ? "diamond" : n.shape === "ellipse" ? "ellipse" : "rectangle";
  els.push(base(type, n.x, n.y, n.w, n.h, { backgroundColor: n.fill || "transparent", roundness: type === "rectangle" ? { type: 3 } : null, strokeStyle: n.dashed ? "dashed" : "solid" }));
  els.push(n.align === "left" ? xText(n.text, n.x + 16, n.y + n.h / 2, n.size || 17, "left") : xText(n.text, n.x + n.w / 2, n.y + n.h / 2, n.size || 17));
}
for (const e of spec.edges || []) {
  const pts = edgePoints(e, nodes);
  const [x0, y0] = pts[0];
  const rel = pts.map(([x, y]) => [x - x0, y - y0]);
  const xs = rel.map((p) => p[0]), ys = rel.map((p) => p[1]);
  els.push(base("arrow", x0, y0, Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), {
    points: rel, startArrowhead: null, endArrowhead: "arrow", strokeStyle: e.dashed ? "dashed" : "solid", lastCommittedPoint: null, startBinding: null, endBinding: null,
  }));
  if (e.label) {
    const i = e.labelSeg ?? Math.floor((pts.length - 1) / 2);
    els.push(xText(e.label, e.lx ?? (pts[i][0] + pts[i + 1][0]) / 2, e.ly ?? (pts[i][1] + pts[i + 1][1]) / 2, e.size || 15));
  }
}
for (const t of spec.texts || []) els.push(xText(t.text, t.x, t.y, t.size || 16, t.anchor === "start" ? "left" : "center"));
fs.writeFileSync(path.join(outDir, `${spec.name}.excalidraw`), JSON.stringify({
  type: "excalidraw", version: 2, source: "https://excalidraw.com", elements: els,
  appState: { viewBackgroundColor: spec.bg || "#ffffff", gridSize: null }, files: {},
}));
console.log(`${spec.name}: ${W}x${H}, ${els.length} excalidraw elements`);
