/**
 * Generate PWA icon PNGs from the existing gradient favicon SVG.
 *
 * These outputs are COMMITTED static assets (public/icons/*.png) so the normal
 * `npm run build` / CI needs no render step. Re-run manually after changing the
 * source art:  npm run gen:icons
 *
 * Reuses the browser-tab favicon identity (icons-src/evolve-icon.svg) — same
 * gradient upward-tick glyph — rather than inventing a new mark.
 */
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const srcDir = join(root, "icons-src");
const outDir = join(root, "public", "icons");

mkdirSync(outDir, { recursive: true });

const standard = readFileSync(join(srcDir, "evolve-icon.svg"));
const maskable = readFileSync(join(srcDir, "evolve-icon-maskable.svg"));

/** [source svg buffer, output filename, pixel size] */
const targets = [
  [standard, "pwa-192.png", 192],
  [standard, "pwa-512.png", 512],
  [maskable, "pwa-maskable-512.png", 512],
  // iOS home screen: opaque 180x180, Safari rounds corners itself.
  [maskable, "apple-touch-icon.png", 180],
];

for (const [svg, name, size] of targets) {
  const png = await sharp(svg, { density: 384 })
    .resize(size, size, { fit: "contain", background: { r: 5, g: 7, b: 13, alpha: 1 } })
    .png()
    .toBuffer();
  writeFileSync(join(outDir, name), png);
  console.log(`gen-pwa-icons: wrote public/icons/${name} (${size}x${size})`);
}

console.log("gen-pwa-icons: done");
