/**
 * Config-correctness self-test for PWA live-data exclusions.
 * Run: node --experimental-strip-types src/pwaWorkbox.selftest.ts
 * (from web/frontend; may also be invoked after build to inspect dist SW).
 */
import { readFileSync, existsSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  NAVIGATE_FALLBACK_DENYLIST,
  LIVE_DATA_PATH_PREFIXES,
  denylistMatchesLivePaths,
  isLiveDataPath,
  liveDataNetworkOnlyCaching,
  buildWorkboxOptions,
} from "../pwaWorkbox.ts";

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

function main(): void {
  assert(denylistMatchesLivePaths([...NAVIGATE_FALLBACK_DENYLIST]), "denylist must match /api|/ws only");
  assert(isLiveDataPath("/api/quote/SPY"), "/api must be live");
  assert(isLiveDataPath("/ws/notifications"), "/ws must be live");
  assert(!isLiveDataPath("/assets/index.js"), "assets must not be live");
  assert(!isLiveDataPath("/"), "root must not be live");

  const caching = liveDataNetworkOnlyCaching();
  assert(caching.length > 0, "expected NetworkOnly runtime entries");
  assert(
    caching.every((e) => e.handler === "NetworkOnly"),
    "every live runtime entry must be NetworkOnly",
  );
  for (const prefix of LIVE_DATA_PATH_PREFIXES) {
    assert(
      caching.some((e) => e.urlPattern.test(`${prefix}x`)),
      `missing NetworkOnly pattern for ${prefix}`,
    );
  }

  const wb = buildWorkboxOptions();
  assert(wb.navigateFallback === "/index.html", "SPA navigateFallback");
  assert(
    denylistMatchesLivePaths(wb.navigateFallbackDenylist),
    "workbox denylist must exclude live paths",
  );
  // No CacheFirst / StaleWhileRevalidate for live routes in our config.
  for (const entry of wb.runtimeCaching) {
    assert(entry.handler === "NetworkOnly", `unexpected handler ${entry.handler}`);
    const sampleApi = "/api/health";
    const sampleWs = "/ws/notifications";
    if (entry.urlPattern.test(sampleApi) || entry.urlPattern.test(sampleWs)) {
      assert(entry.handler === "NetworkOnly", "live URL matched non-NetworkOnly");
    }
  }

  // If a PWA build already exists, assert generated SW embeds NetworkOnly + denylist.
  // Skip when dist is absent or from a pre-PWA build (CI runs test:unit before build).
  const here = dirname(fileURLToPath(import.meta.url));
  const dist = join(here, "..", "dist");
  if (existsSync(join(dist, "sw.js"))) {
    const files = readdirSync(dist);
    const candidates = files.filter(
      (f) => f === "sw.js" || f.startsWith("workbox-") || f.endsWith("sw.js"),
    );
    assert(candidates.length > 0, `no service worker artifacts in dist (${files.join(",")})`);
    let blob = "";
    for (const f of candidates) {
      blob += readFileSync(join(dist, f), "utf8");
    }
    assert(/NetworkOnly/i.test(blob), "built SW must mention NetworkOnly");
    assert(/\/api\//.test(blob) || /\\\/api\\\//.test(blob), "built SW must mention /api/ exclusion");
    assert(/\/ws\//.test(blob) || /\\\/ws\\\//.test(blob), "built SW must mention /ws/ exclusion");
    const apiIdx = blob.search(/\\\/api\\\//);
    if (apiIdx >= 0) {
      const window = blob.slice(Math.max(0, apiIdx - 80), apiIdx + 120);
      assert(
        !/CacheFirst|StaleWhileRevalidate|CacheOnly/i.test(window) || /NetworkOnly/i.test(window),
        `built SW /api/ neighborhood must not use a cache strategy: ${window.slice(0, 100)}`,
      );
    }
  }

  console.log("pwaWorkbox.selftest: PASS");
}

main();
