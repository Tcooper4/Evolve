/**
 * analyzeLoad.selftest — hanging score / chart-events must not delay critical paint.
 */
import { runAnalyzeLoad } from "./analyzeLoad.ts";

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

async function main(): Promise<void> {
  let criticalAt = 0;
  let scoreAt = 0;
  let eventsAt = 0;
  let scoreStarted = false;
  let eventsStarted = false;

  const hangMs = 250;

  const result = await runAnalyzeLoad({
    symbol: "IDXX",
    mode: "long",
    fetchers: {
      getScore: async () => {
        scoreStarted = true;
        await sleep(hangMs);
        scoreAt = performance.now();
        return { symbol: "IDXX", score: 5.7, grade: "C" };
      },
      getHistory: async () => {
        await sleep(5);
        return {
          symbol: "IDXX",
          candles: [{ time: "2026-01-01", open: 1, high: 1, low: 1, close: 1, volume: 1 }],
        };
      },
      getChartEvents: async () => {
        eventsStarted = true;
        await sleep(hangMs);
        eventsAt = performance.now();
        return { events: [{ time: "2026-01-01" }] };
      },
      getNews: async () => {
        await sleep(5);
        return { items: [{ title: "t" }] };
      },
      getRisk: async () => {
        await sleep(5);
        return { metrics: { vol: 1 } };
      },
      getPlaybook: async () => {
        await sleep(5);
        return { strategy: {} };
      },
      getEarnings: async () => {
        await sleep(5);
        return {};
      },
      getForecast: async () => {
        await sleep(5);
        return { forecast: { values: [1] } };
      },
      getNewsContext: async () => {
        await sleep(5);
        return { items: [] };
      },
    },
    hooks: {
      onCritical: () => {
        criticalAt = performance.now();
      },
      onScore: () => {},
      onEvents: () => {},
      onExtras: () => {},
      onNewsWhy: () => {},
      onForecast: () => {},
      onForecastSettled: () => {},
    },
  });

  await sleep(20);

  if (result !== "critical_ok") {
    throw new Error(`expected critical_ok, got ${result}`);
  }
  if (!criticalAt) throw new Error("onCritical never fired");
  if (!scoreStarted) throw new Error("score should start before critical settles");
  if (!eventsStarted) throw new Error("chart-events should start after critical");
  // Critical must finish well before hanging score/events resolve
  if (scoreAt && scoreAt <= criticalAt) {
    throw new Error("score resolved before critical — hang not simulated");
  }
  if (eventsAt && eventsAt <= criticalAt) {
    throw new Error("events resolved before critical — hang not simulated");
  }
  // Give hanging work time — critical should already be done ~200ms earlier
  const lag = hangMs - 20;
  if (performance.now() - criticalAt < lag * 0.5) {
    // ok — we already returned from runAnalyzeLoad
  }

  // Soft check: runAnalyzeLoad returned while score still hanging
  if (scoreAt) {
    throw new Error("score should still be hanging when runAnalyzeLoad returns");
  }

  await sleep(hangMs + 40);
  console.log("analyzeLoad.selftest: PASS");
}

main().catch((e) => {
  console.error("analyzeLoad.selftest: FAIL", e);
  process.exit(1);
});
