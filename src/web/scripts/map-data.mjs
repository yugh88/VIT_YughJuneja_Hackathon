// Generates the offline map data used by the Home screen (run: node scripts/map-data.mjs).
// Source: world-atlas 110m (Natural Earth, public domain; package ISC). Output is small JSON.
import { readFileSync, writeFileSync } from "node:fs";
import { feature } from "topojson-client";

const land = feature(JSON.parse(readFileSync("node_modules/world-atlas/land-110m.json")), "land");
const countries = feature(JSON.parse(readFileSync("node_modules/world-atlas/countries-110m.json")), "countries");
const codes = JSON.parse(readFileSync("src/data/countries.json")).numeric_to_alpha2;

const polys = (g) => (g.type === "Polygon" ? [g.coordinates] : g.type === "MultiPolygon" ? g.coordinates : []);
const inRing = ([x, y], ring) => {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
};
const inPoly = (p, poly) => inRing(p, poly[0]) && !poly.slice(1).some((h) => inRing(p, h));

// Dotted land grid, 2.4 degrees, Antarctica cut.
const landPolys = land.features.flatMap((f) => polys(f.geometry));
const dots = [];
for (let lat = 80; lat >= -56; lat -= 2.4)
  for (let lon = -178; lon <= 180; lon += 2.4) {
    const p = [lon, lat];
    if (landPolys.some((poly) => inPoly(p, poly))) dots.push([+lon.toFixed(1), +lat.toFixed(1)]);
  }

// Country label points: area-weighted centre of the largest polygon.
const centroids = {};
for (const f of countries.features) {
  const a2 = codes[String(f.id).padStart(3, "0")];
  if (!a2) continue;
  let best = null, bestA = -1;
  for (const poly of polys(f.geometry)) {
    const r = poly[0];
    let A = 0, cx = 0, cy = 0;
    for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
      const c = r[j][0] * r[i][1] - r[i][0] * r[j][1];
      A += c; cx += (r[j][0] + r[i][0]) * c; cy += (r[j][1] + r[i][1]) * c;
    }
    if (Math.abs(A) > bestA) { bestA = Math.abs(A); best = [cx / (3 * A), cy / (3 * A)]; }
  }
  if (best) centroids[a2] = [+best[0].toFixed(2), +best[1].toFixed(2)];
}
writeFileSync("src/data/worldmap.json", JSON.stringify({ dots, centroids }));
console.log(`dots ${dots.length}, centroids ${Object.keys(centroids).length}`);
