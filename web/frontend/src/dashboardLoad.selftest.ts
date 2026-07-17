/**
 * Self-test for dashboardLoad sequencing (Node 22 --experimental-strip-types).
 * Asserts hanging chart-events does not delay critical paint.
 */
import { runDashboardLoad } from "./dashboardLoad.ts";

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

async function main(): Promise<void> {
  let criticalAt = 0;
  let overlayAt = 0;
  let chartEventsStarted = false;

  const hangMs = 250;

  const result = await runDashboardLoad({
    symbol: "GE",
    period: "6mo",
    interval: "",
    soft: false,
    fetchers: {
      getQuote: async () => {
        await sleep(5);
        return { symbol: "GE", price: 100, volume: 1 };
      },
      getHistory: async () => {
        await sleep(5);
        return {
          symbol: "GE",
          candles: [{ time: "2026-01-01", open: 1, high: 1, low: 1, close: 1, volume: 1 }],
        };
      },
      getNews: async () => {
        await sleep(5);
        return { items: [{ title: "t" }] };
      },
      getChartEvents: async () => {
        chartEventsStarted = true;
        await sleep(hangMs);
        return { events: [{ time: "2026-01-01", text: "N" }] };
      },
      getBreakingNews: async () => {
        await sleep(5);
        return { items: [] };
      },
      getMarketState: async () => {
        await sleep(5);
        return { level: "calm" };
      },
    },
    hooks: {
      onCritical: () => {
        criticalAt = performance.now();
      },
      onOverlay: () => {
        overlayAt = performance.now();
      },
    },
  });

  // Give the microtask that starts the overlay a tick to fire
  await sleep(20);

  if (result !== "critical_ok") {
    throw new Error(`expected critical_ok, got ${result}`);
  }
  if (!criticalAt) {
    throw new Error("onCritical never fired");
  }
  if (!chartEventsStarted) {
    throw new Error("overlay chart-events should have started after critical");
  }
  // Critical paint must finish well before the hanging chart-events resolves
  const paintLag = criticalAt; // absolute not useful; check overlay not yet done
  if (overlayAt) {
    throw new Error(
      `onOverlay fired too early (${overlayAt}); chart-events should still be hanging`,
    );
  }
  // Wait for hang to finish and confirm overlay arrives
  await sleep(hangMs + 50);
  if (!overlayAt) {
    throw new Error("onOverlay never fired after chart-events resolved");
  }
  if (overlayAt - criticalAt < hangMs * 0.5) {
    throw new Error(
      `overlay arrived too soon after critical (${overlayAt - criticalAt}ms); ` +
        `expected ~${hangMs}ms hang`,
    );
  }

  // Soft load must not call chart-events
  let softChartCalls = 0;
  await runDashboardLoad({
    symbol: "SPY",
    period: "6mo",
    interval: "",
    soft: true,
    fetchers: {
      getQuote: async () => ({ symbol: "SPY", price: 1 }),
      getHistory: async () => ({ symbol: "SPY", candles: [] }),
      getNews: async () => {
        throw new Error("soft must not fetch news");
      },
      getChartEvents: async () => {
        softChartCalls += 1;
        return { events: [] };
      },
      getBreakingNews: async () => {
        throw new Error("soft must not fetch breaking");
      },
      getMarketState: async () => {
        throw new Error("soft must not fetch market-state");
      },
    },
    hooks: {
      onCritical: () => {},
      onOverlay: () => {
        throw new Error("soft must not call onOverlay");
      },
    },
  });
  if (softChartCalls !== 0) {
    throw new Error("soft refresh must skip chart-events");
  }

  // Silence unused (paintLag kept for clarity if we extend)
  void paintLag;
  console.log("dashboardLoad.selftest: PASS");
}

main().catch((e) => {
  console.error("dashboardLoad.selftest: FAIL", e);
  process.exit(1);
});
