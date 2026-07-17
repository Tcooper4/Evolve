/**
 * Post-build assertion: generated service worker embeds live-data exclusions.
 * Invoked at the end of `npm run build`.
 */
import { readFileSync, existsSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const dist = join(dirname(fileURLToPath(import.meta.url)), "..", "dist");

function fail(msg) {
  console.error(`assert-pwa-sw: FAIL — ${msg}`);
  process.exit(1);
}

if (!existsSync(dist)) fail("dist/ missing — run vite build first");

const files = readdirSync(dist);
const candidates = files.filter(
  (f) => f === "sw.js" || f.startsWith("workbox-") || f.endsWith("sw.js"),
);
if (!candidates.length) fail(`no SW artifacts in dist: ${files.join(", ")}`);

let blob = "";
for (const f of candidates) {
  blob += readFileSync(join(dist, f), "utf8");
}

if (!/NetworkOnly/i.test(blob)) fail("built SW missing NetworkOnly");
if (!/\/api\//.test(blob) && !/\\\/api\\\//.test(blob)) fail("built SW missing /api/ pattern");
if (!/\/ws\//.test(blob) && !/\\\/ws\\\//.test(blob)) fail("built SW missing /ws/ pattern");

// Manifest must exist and every declared icon must be a real emitted file.
const manifestName = files.find((f) => f.endsWith("webmanifest") || f === "manifest.webmanifest");
if (!manifestName) fail("manifest.webmanifest missing from dist");

const manifest = JSON.parse(readFileSync(join(dist, manifestName), "utf8"));
const icons = Array.isArray(manifest.icons) ? manifest.icons : [];
if (icons.length < 3) fail(`manifest declares only ${icons.length} icons`);
if (!icons.some((i) => (i.purpose || "").includes("maskable"))) fail("manifest missing maskable icon");
for (const icon of icons) {
  const rel = String(icon.src).replace(/^\//, "");
  if (!existsSync(join(dist, rel))) fail(`manifest icon not in build output: ${icon.src}`);
}

// iOS home-screen icon must be referenced by index.html and emitted.
const indexHtml = readFileSync(join(dist, "index.html"), "utf8");
const appleMatch = /rel="apple-touch-icon"\s+href="([^"]+)"/.exec(indexHtml);
if (!appleMatch) fail("index.html missing apple-touch-icon link");
const appleRel = appleMatch[1].replace(/^\//, "");
if (!existsSync(join(dist, appleRel))) fail(`apple-touch-icon not in build output: ${appleMatch[1]}`);

console.log(`assert-pwa-sw: PASS (${candidates.join(", ")}, ${manifestName}, ${icons.length} icons + apple-touch)`);
