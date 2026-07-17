/**
 * Manifest integrity self-test: every icon the manifest declares must exist as
 * a real file in the build output (not merely declared). Also verifies the
 * apple-touch-icon referenced by index.html is emitted.
 *
 * Skips when dist/ has no PWA build yet (CI runs test:unit before build); the
 * authoritative post-build check lives in scripts/assert-pwa-sw.mjs.
 */
import { readFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

function main(): void {
  const here = dirname(fileURLToPath(import.meta.url));
  const dist = join(here, "..", "dist");
  const manifestPath = join(dist, "manifest.webmanifest");

  if (!existsSync(manifestPath)) {
    console.log("pwaManifest.selftest: SKIP (no dist manifest yet)");
    return;
  }

  const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as {
    icons?: { src: string; sizes: string; type: string; purpose?: string }[];
  };
  const icons = manifest.icons ?? [];
  assert(icons.length >= 3, `expected >=3 icons, got ${icons.length}`);

  const hasMaskable = icons.some((i) => (i.purpose ?? "").includes("maskable"));
  assert(hasMaskable, "manifest must declare a maskable icon");
  const has192 = icons.some((i) => i.sizes === "192x192");
  const has512 = icons.some((i) => i.sizes === "512x512");
  assert(has192 && has512, "manifest must declare 192 and 512 icons");

  for (const icon of icons) {
    const rel = icon.src.replace(/^\//, "");
    const p = join(dist, rel);
    assert(existsSync(p), `manifest icon missing from build output: ${icon.src}`);
  }

  // Apple touch icon: referenced by index.html, needed for iOS home screen.
  const indexHtml = readFileSync(join(dist, "index.html"), "utf8");
  const m = /rel="apple-touch-icon"\s+href="([^"]+)"/.exec(indexHtml);
  assert(m, "index.html must include an apple-touch-icon link");
  const appleRel = m[1].replace(/^\//, "");
  assert(
    existsSync(join(dist, appleRel)),
    `apple-touch-icon missing from build output: ${m[1]}`,
  );

  console.log("pwaManifest.selftest: PASS");
}

main();
