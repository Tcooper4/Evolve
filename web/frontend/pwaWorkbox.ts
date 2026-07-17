/**
 * Shared Workbox options for vite-plugin-pwa.
 *
 * SCOPING DECISION (do not broaden casually):
 * This service worker caches the STATIC APP SHELL (JS/CSS/HTML/icons) for
 * fast repeat loads and installability — NOT offline access to live data.
 * Quotes, portfolio state, chat, and alerts are inherently live; serving a
 * stale cached copy would be actively misleading for a trading research tool.
 * Therefore /api/* and /ws/* must never enter a caching strategy — NetworkOnly
 * only, and navigateFallback must not swallow those paths into index.html.
 */

/** Path prefixes that must never be served from cache or as the HTML shell. */
export const LIVE_DATA_PATH_PREFIXES = ["/api/", "/ws/"] as const;

/**
 * navigateFallbackDenylist — SPA shell must not claim API/WS navigations.
 * Kept as RegExp source strings so tests can assert the built SW embeds them.
 */
export const NAVIGATE_FALLBACK_DENYLIST: RegExp[] = [
  /^\/api\//,
  /^\/ws\//,
];

/** Precache only static shell assets — never JSON API payloads. */
export const PRECACHE_GLOB_PATTERNS = [
  "**/*.{js,css,html,ico,png,svg,webp,woff2}",
];

export type RuntimeCachingEntry = {
  urlPattern: RegExp;
  handler: "NetworkOnly";
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE" | "HEAD";
};

/**
 * Explicit NetworkOnly handlers for live endpoints.
 * Empty/default Workbox already leaves unmatched fetches on the network, but
 * these entries make the exclusion visible in the generated SW and in tests.
 */
export function liveDataNetworkOnlyCaching(): RuntimeCachingEntry[] {
  const entries: RuntimeCachingEntry[] = [];
  for (const prefix of LIVE_DATA_PATH_PREFIXES) {
    const escaped = prefix.replace(/\//g, "\\/");
    const pattern = new RegExp(escaped);
    // Cover common HTTP methods used by Evolve's REST surface.
    for (const method of ["GET", "POST", "PUT", "PATCH", "DELETE"] as const) {
      entries.push({ urlPattern: pattern, handler: "NetworkOnly", method });
    }
  }
  return entries;
}

export function buildWorkboxOptions() {
  return {
    // SCOPING DECISION: precache static shell only — see file header.
    globPatterns: [...PRECACHE_GLOB_PATTERNS],
    navigateFallback: "/index.html",
    // SCOPING DECISION: never fall back HTML shell for live endpoints.
    navigateFallbackDenylist: [...NAVIGATE_FALLBACK_DENYLIST],
    // SCOPING DECISION: runtime-cache nothing for /api or /ws — NetworkOnly.
    runtimeCaching: liveDataNetworkOnlyCaching(),
    cleanupOutdatedCaches: true,
  };
}

/** True if a pathname is a live data route that must not be cached. */
export function isLiveDataPath(pathname: string): boolean {
  const p = pathname.startsWith("/") ? pathname : `/${pathname}`;
  return LIVE_DATA_PATH_PREFIXES.some((prefix) => p.startsWith(prefix));
}

/**
 * Config-correctness helper: a denylist pattern must match live paths and
 * must NOT match static shell paths.
 */
export function denylistMatchesLivePaths(patterns: RegExp[]): boolean {
  const liveSamples = ["/api/quote/SPY", "/api/health", "/ws/notifications", "/ws/quote/AAPL"];
  const shellSamples = ["/", "/index.html", "/assets/index.js", "/manifest.webmanifest"];
  for (const sample of liveSamples) {
    if (!patterns.some((re) => re.test(sample))) return false;
  }
  for (const sample of shellSamples) {
    if (patterns.some((re) => re.test(sample))) return false;
  }
  return true;
}
