/**
 * Analyze ticker load sequencing (mirrors dashboardLoad.ts).
 *
 * Critical path (clears main loading / paints chart): history only.
 * Score path: full AI score — started immediately, never blocks first paint.
 * Overlay path (never blocks critical): chart-events, news, risk, playbook,
 * earnings, forecast (+ news context after headlines arrive).
 */

export type AnalyzeHistory = {
  symbol: string;
  candles: unknown[];
};

export type AnalyzeScore = {
  symbol?: string;
  score?: number | null;
  grade?: string | null;
  error?: string | null;
  signals?: Record<string, unknown>;
  summary?: string | null;
  last_price?: number | null;
  short_score?: number | null;
  technical_score?: number | null;
  momentum_score?: number | null;
  sentiment_score?: number | null;
  fundamental_score?: number | null;
  signal_list?: { name?: string; value?: unknown; impact?: string; description?: string }[];
};

export type AnalyzeNewsBundle = { items?: unknown[] };
export type AnalyzeEventsBundle = { events?: unknown[] };
export type AnalyzeRiskBundle = { metrics?: Record<string, unknown> | null };
export type AnalyzeForecastBundle = { forecast?: Record<string, unknown> | null };

export type AnalyzeFetchers = {
  getScore: (sym: string, mode: "long" | "short") => Promise<AnalyzeScore>;
  getHistory: (sym: string, period: string) => Promise<AnalyzeHistory>;
  getChartEvents: (sym: string, period: string) => Promise<AnalyzeEventsBundle>;
  getNews: (sym: string) => Promise<AnalyzeNewsBundle>;
  getRisk: (sym: string) => Promise<AnalyzeRiskBundle>;
  getPlaybook: (sym: string) => Promise<Record<string, unknown> | null>;
  getEarnings: (sym: string) => Promise<Record<string, unknown> | null>;
  getForecast: (sym: string) => Promise<AnalyzeForecastBundle>;
  getNewsContext: (
    titles: string[],
  ) => Promise<{ items?: Array<{ title?: string; why?: string }> }>;
};

export type AnalyzeLoadHooks = {
  onCritical: (payload: { symbol: string; history: AnalyzeHistory }) => void;
  onCriticalError?: (symbol: string) => void;
  onScore: (score: AnalyzeScore) => void;
  onScoreError?: (symbol: string) => void;
  onEvents: (events: unknown[]) => void;
  onExtras: (payload: {
    news: unknown[];
    risk: Record<string, unknown> | null;
    playbook: Record<string, unknown> | null;
    earnings: Record<string, unknown> | null;
  }) => void;
  onNewsWhy: (map: Record<string, string>) => void;
  onForecast: (forecast: Record<string, unknown> | null) => void;
  onForecastSettled: () => void;
};

export type AnalyzeLoadResult = "critical_ok" | "critical_error";

/**
 * Run an Analyze load. Awaits only history for the critical path; score and
 * overlays run without blocking first paint (same contract as Dashboard).
 */
export async function runAnalyzeLoad(opts: {
  symbol: string;
  mode: "long" | "short";
  period?: string;
  fetchers: AnalyzeFetchers;
  hooks: AnalyzeLoadHooks;
}): Promise<AnalyzeLoadResult> {
  const { symbol, mode, fetchers, hooks } = opts;
  const period = opts.period ?? "6mo";

  // Kick score immediately — do not await before painting the chart.
  void fetchers
    .getScore(symbol, mode)
    .then((s) => hooks.onScore(s))
    .catch(() => hooks.onScoreError?.(symbol));

  let critical: AnalyzeLoadResult = "critical_ok";
  try {
    const history = await fetchers.getHistory(symbol, period);
    hooks.onCritical({ symbol, history });
  } catch {
    hooks.onCriticalError?.(symbol);
    critical = "critical_error";
  }

  // Overlays: fire-and-forget so chart-events / forecast cannot block paint.
  void (async () => {
    try {
      void fetchers
        .getChartEvents(symbol, period)
        .then((ev) => hooks.onEvents(ev.events ?? []))
        .catch(() => hooks.onEvents([]));

      const [n, rk, pb, er] = await Promise.all([
        fetchers.getNews(symbol).catch(() => ({ items: [] as unknown[] })),
        fetchers.getRisk(symbol).catch(() => ({ metrics: null })),
        fetchers.getPlaybook(symbol).catch(() => null),
        fetchers.getEarnings(symbol).catch(() => null),
      ]);
      const newsItems = (n.items as unknown[]) ?? [];
      hooks.onExtras({
        news: newsItems,
        risk: (rk.metrics as Record<string, unknown> | null) ?? null,
        playbook: pb,
        earnings: er,
      });

      const titles = newsItems
        .map((it) => {
          const row = it as Record<string, unknown>;
          return String(row.title ?? row.headline ?? "").trim();
        })
        .filter(Boolean)
        .slice(0, 6);
      if (titles.length) {
        void fetchers
          .getNewsContext(titles)
          .then((ctx) => {
            const map: Record<string, string> = {};
            for (const it of ctx.items ?? []) {
              if (it.title && it.why) map[it.title] = it.why;
            }
            hooks.onNewsWhy(map);
          })
          .catch(() => {});
      }

      await fetchers
        .getForecast(symbol)
        .then((f) => hooks.onForecast((f.forecast as Record<string, unknown>) ?? null))
        .catch(() => hooks.onForecast(null));
    } finally {
      hooks.onForecastSettled();
    }
  })();

  return critical;
}
