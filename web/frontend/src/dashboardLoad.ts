/**
 * Dashboard ticker load sequencing.
 *
 * Critical path (blocks the loading spinner): quote + history + news.
 * Overlay path (never blocks first paint): chart-events, breaking, market-state.
 *
 * Soft refresh skips the overlay entirely (same as before — chart-events/GDELT
 * must not re-run on the 20–45s poll).
 */

export type QuoteLike = {
  symbol: string;
  price: number | null;
  prev_close: number | null;
  change_pct: number | null;
  volume: number | null;
};

export type HistoryLike = {
  symbol: string;
  candles: unknown[];
  suggestion?: string | null;
};

export type NewsBundle = { items?: unknown[] };
export type EventsBundle = { events?: unknown[] };

export type DashboardFetchers = {
  getQuote: (sym: string) => Promise<QuoteLike>;
  getHistory: (
    sym: string,
    per: string,
    interval: string,
  ) => Promise<HistoryLike>;
  getNews: (sym: string, n: number) => Promise<NewsBundle>;
  getChartEvents: (sym: string, per: string) => Promise<EventsBundle>;
  getBreakingNews: (n: number) => Promise<NewsBundle>;
  getMarketState: (sym: string) => Promise<unknown | null>;
};

export type CriticalPayload = {
  quote: QuoteLike;
  history: HistoryLike;
  news: NewsBundle;
};

export type OverlayPayload = {
  events: unknown[];
  breaking: unknown[];
  marketState: unknown | null;
};

export type DashboardLoadHooks = {
  /** Critical data ready — clear loading / paint chart here. */
  onCritical: (payload: CriticalPayload) => void;
  /** Overlay data ready — apply markers / side panels. */
  onOverlay: (payload: OverlayPayload) => void;
  onCriticalError?: () => void;
};

export type DashboardLoadResult =
  | "critical_ok"
  | "critical_error"
  | "soft_ok";

/**
 * Run a Dashboard ticker load.
 *
 * Hard load awaits only the critical fan-out, invokes ``onCritical``, then
 * starts the overlay fan-out without awaiting it (so a hanging chart-events
 * call cannot block first paint). Soft load refreshes quote+history only.
 */
export async function runDashboardLoad(opts: {
  symbol: string;
  period: string;
  interval: string;
  soft: boolean;
  fetchers: DashboardFetchers;
  hooks: DashboardLoadHooks;
}): Promise<DashboardLoadResult> {
  const { symbol, period, interval, soft, fetchers, hooks } = opts;
  try {
    if (soft) {
      const [quote, history] = await Promise.all([
        fetchers.getQuote(symbol),
        fetchers.getHistory(symbol, period, interval),
      ]);
      hooks.onCritical({ quote, history, news: { items: [] } });
      return "soft_ok";
    }

    const [quote, history, news] = await Promise.all([
      fetchers.getQuote(symbol),
      fetchers.getHistory(symbol, period, interval),
      fetchers.getNews(symbol, 5).catch(() => ({ items: [] as unknown[] })),
    ]);
    hooks.onCritical({ quote, history, news });

    void (async () => {
      const [ev, br, ms] = await Promise.all([
        fetchers.getChartEvents(symbol, period).catch(() => ({ events: [] })),
        fetchers.getBreakingNews(5).catch(() => ({ items: [] as unknown[] })),
        fetchers.getMarketState(symbol).catch(() => null),
      ]);
      hooks.onOverlay({
        events: ev.events ?? [],
        breaking: (br.items as unknown[]) ?? [],
        marketState: ms,
      });
    })();

    return "critical_ok";
  } catch {
    hooks.onCriticalError?.();
    return "critical_error";
  }
}
