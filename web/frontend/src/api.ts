// Thin typed client for the Evolve API. The JWT lives in memory +
// sessionStorage (survives refresh, cleared on tab close).

export interface Quote {
  symbol: string;
  price: number | null;
  prev_close: number | null;
  change_pct: number | null;
  volume: number | null;
}

export interface Candle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface WatchlistRow {
  symbol: string;
  note?: string | null;
  user_id?: string;
}

let token: string | null = sessionStorage.getItem("evolve_token");

export function setToken(t: string | null) {
  token = t;
  if (t) sessionStorage.setItem("evolve_token", t);
  else sessionStorage.removeItem("evolve_token");
}

export function hasToken(): boolean {
  return !!token;
}

async function req<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    ...(init.headers as Record<string, string> | undefined),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(path, { ...init, headers });
  if (res.status === 401) {
    setToken(null);
    throw new Error("unauthorized");
  }
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export async function login(username: string, password: string) {
  const body = new URLSearchParams({ username, password });
  const res = await fetch("/api/auth/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
  if (!res.ok) throw new Error("Incorrect username or password");
  const data = (await res.json()) as {
    access_token: string;
    display_name: string;
  };
  setToken(data.access_token);
  return data.display_name;
}

export const getQuote = (symbol: string) => req<Quote>(`/api/quote/${symbol}`);

export const getHistory = (symbol: string, period = "6mo", interval = "") =>
  req<{
    symbol: string;
    candles: Candle[];
    interval?: string;
    resolved_from?: string | null;
    suggestion?: string | null;
  }>(
    `/api/history/${symbol}?period=${encodeURIComponent(period)}${interval ? `&interval=${encodeURIComponent(interval)}` : ""}`,
  );

export type ChartEventHeadline = {
  title: string;
  source?: string;
  link_quality?: "same_day" | "fallback_recent" | string;
  date_confirmed?: boolean;
};

export type ChartEvent = {
  time: string;
  title?: string;
  text?: string;
  color?: string;
  shape?: string;
  headlines?: Array<string | ChartEventHeadline>;
  volume_ratio?: number;
  price_change_pct?: number;
  link_quality?: "same_day" | "fallback_recent" | string;
  date_confirmed?: boolean;
  provisional?: boolean;
  tier?: "significant" | "notable" | string;
};

export const getChartEvents = (symbol: string, period = "6mo") =>
  req<{ success: boolean; events?: ChartEvent[]; error?: string }>(
    `/api/chart-events/${encodeURIComponent(symbol)}?period=${encodeURIComponent(period)}`,
  );

export const getDiagnostics = (symbol: string) =>
  req<Record<string, unknown>>(`/api/diagnostics/${encodeURIComponent(symbol)}`);

/** @deprecated Use getDiagnostics — former "Causal" label was a misnomer. */
export const getCausal = getDiagnostics;

export const getPatterns = (symbol: string) =>
  req<Record<string, unknown>>(`/api/patterns/${encodeURIComponent(symbol)}`);

export const getPlaybook = (symbol: string) =>
  req<Record<string, unknown>>(`/api/playbook/${encodeURIComponent(symbol)}`);

export const getEarnings = (symbol: string) =>
  req<Record<string, unknown>>(`/api/earnings/${encodeURIComponent(symbol)}`);

export const runTuneModels = (symbol: string, n_trials = 12) =>
  req<Record<string, unknown>>("/api/tune-models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      symbol,
      n_trials,
      models: ["xgboost", "ridge", "catboost", "prophet", "garch"],
    }),
  });

export const runGnn = (symbols: string[], period = "1y") =>
  req<Record<string, unknown>>("/api/gnn", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbols, period }),
  });

export const getWatchlist = () => req<WatchlistRow[]>("/api/watchlist");

export const addToWatchlist = (symbol: string) =>
  req<{ ok: boolean }>("/api/watchlist", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol }),
  });

export const removeFromWatchlist = (symbol: string) =>
  req<{ ok: boolean }>(`/api/watchlist/${symbol}`, { method: "DELETE" });

// ---- page-parity endpoints ----
export interface ScoreResult {
  symbol: string;
  score: number | null;
  grade: string | null;
  short_score?: number | null;
  short_grade?: string | null;
  technical_score?: number | null;
  momentum_score?: number | null;
  sentiment_score?: number | null;
  fundamental_score?: number | null;
  summary?: string | null;
  last_price?: number | null;
  signals: Record<string, unknown>;
  signal_list?: { name?: string; value?: unknown; impact?: string; description?: string }[];
  error?: string | null;
}
export const getScore = (symbol: string, mode: "long" | "short" = "long") =>
  req<ScoreResult>(`/api/score/${symbol}?mode=${mode}`);

export const runScan = (opts: {
  filters: string[];
  max_results?: number;
  universe?: string;
  min_quick_score?: number;
  custom_tickers?: string[];
}) =>
  req<{
    success: boolean;
    results?: Record<string, unknown>[];
    error?: string;
    scanned?: number;
    passed?: number;
    universe?: string;
    universe_size?: number;
  }>("/api/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filters: opts.filters,
      max_results: opts.max_results ?? 15,
      universe: opts.universe ?? "sp100",
      min_quick_score: opts.min_quick_score ?? 6.0,
      custom_tickers: opts.custom_tickers ?? [],
    }),
  });

export const runPairs = (
  symbols: string[],
  opts?: { universe?: string; max_symbols?: number; max_pairs?: number },
) =>
  req<{
    success: boolean;
    pairs?: Record<string, unknown>[];
    error?: string;
    tested?: number;
    requested?: number;
    note?: string;
    truncated?: boolean;
  }>(
    "/api/pairs",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symbols,
        universe: opts?.universe ?? "",
        max_symbols: opts?.max_symbols ?? 100,
        max_pairs: opts?.max_pairs ?? 15,
      }),
    },
  );

export const runMonteCarlo = (
  symbol: string,
  n_simulations = 400,
  horizon_days = 63,
  method: "iid" | "stationary_block" = "iid",
) =>
  req<{
    success: boolean;
    error?: string;
    final_p5?: number;
    final_p50?: number;
    final_p95?: number;
    mean_path?: number[];
    note?: string;
    n_simulations?: number;
    horizon_days?: number;
    method?: string;
    block?: Record<string, unknown>;
  }>("/api/monte-carlo", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol, n_simulations, horizon_days, method }),
  });

export const getOptions = (symbol: string) =>
  req<Record<string, unknown>>(`/api/options/${encodeURIComponent(symbol)}`);

export const getOptionsContext = (symbol: string, expiry = "") =>
  req<{
    success: boolean;
    symbol?: string;
    sentiment?: Record<string, unknown>;
    gex?: Record<string, unknown>;
    skew?: Record<string, unknown>;
    disclosure?: string;
    error?: string;
  }>(
    `/api/options/context/${encodeURIComponent(symbol)}`
    + (expiry ? `?expiry=${encodeURIComponent(expiry)}` : ""),
  );

export const getSignalIc = (symbol: string) =>
  req<Record<string, unknown>>(`/api/ic/${encodeURIComponent(symbol)}`);

export const runAllocate = (symbols: string[], period = "1y") =>
  req<Record<string, unknown>>("/api/allocate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbols, period }),
  });

export const getCashbook = () =>
  req<{ success: boolean; cash: number; limit_orders: Record<string, unknown>[] }>(
    "/api/cashbook",
  );

export const adjustCash = (amount: number, note = "") =>
  req<{ success: boolean; cash: number }>("/api/cashbook/adjust", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ amount, note }),
  });

export const placeLimit = (symbol: string, side: string, quantity: number, limit_price: number) =>
  req<{ success: boolean; limit_orders: Record<string, unknown>[] }>("/api/cashbook/limit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol, side, quantity, limit_price }),
  });

export const cancelLimit = (id: string) =>
  req<{ success: boolean; limit_orders: Record<string, unknown>[] }>(
    `/api/cashbook/limit/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );

export const getAlerts = () =>
  req<{ success: boolean; alerts: Record<string, unknown>[]; triggered: Record<string, unknown>[] }>(
    "/api/alerts",
  );

export const upsertAlert = (
  symbol: string,
  condition: string,
  threshold: number,
  confirm?: string | null,
  confirmThreshold?: number | null,
  mode?: "watch" | "action" | null,
) =>
  req<{ success: boolean; alerts: Record<string, unknown>[] }>("/api/alerts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      symbol,
      condition,
      threshold,
      mode: mode || "watch",
      ...(confirm ? { confirm, confirm_threshold: confirmThreshold ?? undefined } : {}),
    }),
  });

export const rearmAlert = (id: string) =>
  req<{ success: boolean; alerts: Record<string, unknown>[]; error?: string }>(
    `/api/alerts/${encodeURIComponent(id)}/rearm`,
    { method: "POST" },
  );

export const deleteAlert = (id: string) =>
  req<{ success: boolean; alerts: Record<string, unknown>[] }>(
    `/api/alerts/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );

export const getStrategies = () =>
  req<{ success: boolean; strategies: string[] }>("/api/strategies");

export type StrategyOverlayMarker = {
  time: string;
  position?: "aboveBar" | "belowBar";
  color?: string;
  shape?: "circle" | "square" | "arrowUp" | "arrowDown";
  text?: string;
  title?: string;
  side?: string;
  label?: string;
  price?: number;
  gamma_tag?: string | null;
};

export type StrategyOverlay = {
  success: boolean;
  symbol?: string;
  strategy?: string;
  markers?: StrategyOverlayMarker[];
  n_markers?: number;
  overlay_series?: Array<{
    id: string;
    label?: string;
    color: string;
    style?: "solid" | "dashed" | "dotted";
    points: Array<{ time: string; value: number }>;
  }>;
  reference_levels?: {
    levels?: Array<{ key: string; label: string; value: number; price_scale?: boolean }>;
    note?: string;
  };
  gamma_context?: Record<string, unknown> | null;
  disclosure?: string;
  default_on?: boolean;
  framing?: string;
  last_bar?: string;
  error?: string;
};

export const getStrategyOverlay = (
  symbol: string,
  strategy: string,
  period = "6mo",
) =>
  req<StrategyOverlay>(
    `/api/strategy-overlay/${encodeURIComponent(symbol)}`
    + `?strategy=${encodeURIComponent(strategy)}`
    + `&period=${encodeURIComponent(period)}`,
  );

export type OptionsStructureOverlay = StrategyOverlay & {
  pick?: {
    structure?: string;
    label?: string;
    mark_text?: string;
    rationale?: string;
    wing_pct_guide?: number | null;
    alternate?: string | null;
  };
  gex?: Record<string, unknown> | null;
  skew?: Record<string, unknown> | null;
};

export const getOptionsStructureOverlay = (symbol: string) =>
  req<OptionsStructureOverlay>(
    `/api/options-structure-overlay/${encodeURIComponent(symbol)}`,
  );

export const runBacktest = (symbol: string, strategy: string,
                            params: Record<string, unknown> = {},
                            period = "1y",
                            opts?: { force_defaults?: boolean; cost_bps?: number }) =>
  req<Record<string, unknown>>("/api/backtest", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      symbol, strategy, params, period,
      force_defaults: opts?.force_defaults ?? false,
      cost_bps: opts?.cost_bps ?? 5,
    }),
  });

export const runOptionsStructureBacktest = (body: {
  symbol: string;
  strategy?: string;
  period?: string;
  dte?: number;
  short_delta?: number;
  wing_pct?: number;
  profit_take?: number;
  max_loss_mult?: number;
  exit_dte_floor?: number;
  sweep?: boolean;
}) =>
  req<Record<string, unknown>>("/api/backtest/options-structure", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

export const runModelBacktest = (symbol: string, model: string, period = "1y") =>
  req<Record<string, unknown>>("/api/backtest/model", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol, model, period, horizon: 5 }),
  });

export const runOptimize = (
  strategy: string,
  symbol: string,
  max_evaluations = 24,
  period = "2y",
  metric = "sharpe_ratio",
) =>
  req<Record<string, unknown>>("/api/optimize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ strategy, symbol, max_evaluations, period, metric }),
  });

export const getStrategyParams = (strategy: string, symbol: string) =>
  req<{
    success: boolean;
    adopted: Record<string, unknown> | null;
    defaults: Record<string, unknown>;
    using_saved: boolean;
  }>(`/api/strategy-params/${encodeURIComponent(strategy)}/${encodeURIComponent(symbol)}`);

export const clearStrategyParams = (strategy: string, symbol: string) =>
  req<{ success: boolean; cleared: boolean }>(
    `/api/strategy-params/${encodeURIComponent(strategy)}/${encodeURIComponent(symbol)}`,
    { method: "DELETE" },
  );

export const sendChat = (message: string) =>
  req<{ success: boolean; reply?: string; error?: string; tool_captions?: string[] }>("/api/chat", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

export const getPulse = () =>
  req<{ success: boolean; items: { symbol: string; label: string; price: number | null; change_pct: number | null }[] }>(
    "/api/pulse",
  );

export const refreshPulseSymbol = (symbol: string) =>
  req<{ success: boolean; symbol: string; price: number | null; change_pct: number | null }>(
    `/api/pulse/${encodeURIComponent(symbol)}`,
  );

export const getNews = (symbol: string, max_items = 8) =>
  req<{ success: boolean; items?: Record<string, unknown>[]; error?: string }>(
    `/api/news/${symbol}?max_items=${max_items}`,
  );

export const getBreakingNews = (max_items = 8) =>
  req<{ success: boolean; items?: Record<string, unknown>[]; source?: string; error?: string }>(
    `/api/news/breaking?max_items=${max_items}`,
  );

export type MarketState = {
  success: boolean;
  symbol?: string;
  level?: string;
  label?: string;
  disclosure?: string;
  predicts_direction?: boolean;
  push_priority?: boolean;
  components?: {
    gex_regime?: string;
    event_severity?: number;
    volatility_regime?: string;
  };
  error?: string;
};

export const getMarketState = (symbol: string, max_headlines = 8) =>
  req<MarketState>(
    `/api/market-state/${encodeURIComponent(symbol)}?max_headlines=${max_headlines}`,
  );

export const getNewsContext = (titles: string[]) =>
  req<{
    success: boolean;
    items?: { title: string; why: string; hedged?: boolean; note?: string }[];
    error?: string;
  }>("/api/news/context", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ titles }),
  });

export const getForecast = (symbol: string, horizon = 7) =>
  req<{ success: boolean; forecast?: Record<string, unknown>; error?: string }>(
    `/api/forecast/${symbol}?horizon=${horizon}`,
  );

export const getRisk = (symbol: string) =>
  req<{ success: boolean; metrics?: Record<string, string | number>; error?: string }>(
    `/api/risk/${symbol}`,
  );

export const runBriefing = (universe = "sp100", min_ai_score = 6.0) =>
  req<{
    success: boolean;
    error?: string;
    markdown?: string;
    market_regime?: Record<string, unknown>;
    top_opportunities?: Record<string, unknown>[];
    short_opportunities?: Record<string, unknown>[];
  }>("/api/briefing", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ universe, min_ai_score, max_positions: 3 }),
  });

export const getKeys = () =>
  req<{ anthropic: boolean; openai: boolean; news: boolean; reddit?: boolean; twitter?: boolean }>(
    "/api/settings/keys",
  );

export const saveKeys = (k: {
  anthropic?: string; openai?: string; news?: string;
  reddit_client_id?: string; reddit_client_secret?: string;
  twitter_bearer?: string;
}) =>
  req<{ ok: boolean }>("/api/settings/keys", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(k),
  });

export const getPrefs = () =>
  req<{ success: boolean; prefs: Record<string, unknown> }>("/api/settings/prefs");

export const savePrefs = (prefs: Record<string, unknown>) =>
  req<{ ok: boolean }>("/api/settings/prefs", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(prefs),
  });

/** App.tsx page ids — keep in sync with backend TOUR_PAGE_IDS. */
export type TourPageId =
  | "dashboard"
  | "analyze"
  | "scanner"
  | "portfolio"
  | "backtest"
  | "chat"
  | "settings";

export const getTours = () =>
  req<{ success: boolean; tours_seen: Record<string, boolean>; error?: string }>(
    "/api/settings/tours",
  );

export const markTourSeen = (page_id: TourPageId) =>
  req<{ success: boolean; tours_seen: Record<string, boolean>; error?: string }>(
    "/api/settings/tours/seen",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ page_id }),
    },
  );

export const resetTours = () =>
  req<{ success: boolean; tours_seen: Record<string, boolean>; error?: string }>(
    "/api/settings/tours/reset",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" },
  );

export interface GprSignal {
  current: number;
  level?: string;
  trend?: string;
  percentile?: number;
  description?: string;
  source?: string;
}
export interface RevisionBreadth {
  success?: boolean;
  pct_up: number;
  pct_down: number;
  pct_neutral?: number;
  signal?: string;
  sample_size?: number;
  description?: string;
  breadth_score?: number;
}
export const getMarketSignals = () =>
  req<{ success: boolean; gpr: GprSignal | null; revision_breadth: RevisionBreadth | null }>(
    "/api/market-signals",
  );
export const loadGpr = () =>
  req<{ success: boolean; gpr?: GprSignal; error?: string }>("/api/market-signals/gpr", {
    method: "POST",
  });
export const loadRevisionBreadth = (sample_size = 150) =>
  req<{ success: boolean; revision_breadth?: RevisionBreadth; error?: string }>(
    `/api/market-signals/revision-breadth?sample_size=${sample_size}`,
    { method: "POST" },
  );

export function quoteSocket(
  symbol: string,
  onMsg: (q: {
    price?: number | null;
    change_pct?: number | null;
    type?: string;
    [key: string]: unknown;
  }) => void,
) {
  const t = sessionStorage.getItem("evolve_token") ?? "";
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/quote/${symbol}?token=${t}`);
  ws.onmessage = (e) => { try { onMsg(JSON.parse(e.data)); } catch { /* skip */ } };
  return ws;
}

export function notificationsSocket(
  onMsg: (n: { type?: string; message?: string; [key: string]: unknown }) => void,
) {
  const t = sessionStorage.getItem("evolve_token") ?? "";
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/notifications?token=${t}`);
  ws.onmessage = (e) => { try { onMsg(JSON.parse(e.data)); } catch { /* skip */ } };
  // Server waits on receive_text for keepalive; ping occasionally
  const ping = window.setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      try { ws.send("ping"); } catch { /* skip */ }
    }
  }, 25000);
  const prevClose = ws.close.bind(ws);
  ws.close = (...args: Parameters<WebSocket["close"]>) => {
    window.clearInterval(ping);
    return prevClose(...args);
  };
  return ws;
}

// ---- paper portfolio ----
export interface Position {
  symbol: string;
  quantity: number;
  avg_cost: number;
  last_price: number | null;
  market_value: number | null;
  unrealized_pnl: number | null;
  unrealized_pct: number | null;
}
export interface PortfolioSummary {
  success: boolean;
  positions: Position[];
  cash: number;
  total_cost_basis: number;
  total_market_value: number;
  total_equity: number;
  total_unrealized_pnl: number;
  realized_pnl: number;
  all_prices_live: boolean;
}
export const getPortfolio = () => req<PortfolioSummary>("/api/portfolio");
export const recordTrade = (symbol: string, side: "buy" | "sell",
                            quantity: number, price?: number) =>
  req<{
    success: boolean; error?: string; realized_pnl?: number;
    recommendation?: { symbol?: string; status?: string };
  }>(
    "/api/portfolio/trade",
    { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol, side, quantity, price }) },
  );
export const getPortfolioTrades = () =>
  req<Record<string, unknown>[]>("/api/portfolio/trades");

// ---- account risk / tracked ideas / filings (2026-07 depth pass) ----
export interface StressEntry { daily_return_pct: number; dollar_impact: number }
export interface AccountRisk {
  success: boolean;
  positions: number;
  trade_stats?: { closed_trades: number; win_rate: number | null; avg_win_loss_ratio: number | null };
  kelly?: {
    success: boolean; full_kelly_fraction?: number; half_kelly_fraction?: number;
    half_kelly_dollars?: number; note?: string;
    vol_multiplier?: number; vol_scaled_down?: boolean;
    half_kelly_dollars_vol_adjusted?: number; vol_adjustment_reason?: string;
    options_vix_multiplier?: number; options_vix_scaled_down?: boolean;
    half_kelly_dollars_options_vix_adjusted?: number;
    options_vix_reason?: string; options_vix?: number;
    options_vix_live_wired?: boolean;
    n_closed_trades?: number | null;
    sample_size_flag?: string;
    sample_size_caveat?: string | null;
    recommended_fraction?: number;
    recommended_dollars?: number;
    recommended_basis?: string;
    quarter_kelly_fraction?: number;
    quarter_kelly_dollars?: number;
  } | null;
  kelly_note?: string;
  portfolio_metrics?: Record<string, string | number> | null;
  stress?: Record<string, StressEntry> | null;
  stress_note?: string;
  concentration?: {
    success?: boolean;
    threshold?: number;
    threshold_note?: string;
    high_pairs?: Array<{
      symbol_a: string;
      symbol_b: string;
      correlation: number;
      abs_correlation?: number;
      message: string;
    }>;
    note?: string;
    error?: string | null;
    n_symbols?: number;
  };
  error?: string;
}
export const getAccountRisk = () => req<AccountRisk>("/api/portfolio/risk");

export interface TrackedRec {
  id: string; symbol: string; source: string; score: number | null;
  price_at_rec: number | null; note: string; created_at: string;
  status?: "open" | "acted" | "closed" | string;
  acted_at?: string | null; acted_price?: number | null;
  closed_at?: string | null; closed_price?: number | null;
  gex_regime?: string | null;
  structure_suggestion?: string | null;
  kelly_recommended_fraction?: number | null;
  kelly_recommended_dollars?: number | null;
  real_acted?: boolean | null;
  real_strategy?: string | null;
  real_entry_price?: number | null;
  real_entry_date?: string | null;
  real_exit_price?: number | null;
  real_exit_date?: string | null;
  real_pnl?: number | null;
  real_notes?: string | null;
  real_outcome_at?: string | null;
  real_won?: boolean | null;
  last_price: number | null; change_pct: number | null;
  change_since_acted_pct?: number | null;
}
export interface RealOutcomeSummary {
  success?: boolean;
  n_with_outcome?: number;
  overall?: { n: number; wins: number; win_rate: number | null; avg_pnl: number | null };
  matched_structure?: { n: number; wins: number; win_rate: number | null; avg_pnl: number | null };
  mismatched_structure?: { n: number; wins: number; win_rate: number | null; avg_pnl: number | null };
  sample_size_flag?: string;
  sample_size_caveat?: string | null;
  small_sample_threshold?: number;
  note?: string;
}
export const getRecs = () =>
  req<{
    success: boolean;
    recommendations: TrackedRec[];
    real_outcome_summary?: RealOutcomeSummary;
  }>("/api/recs");
export const trackRec = (symbol: string, score?: number | null,
                         price_at_rec?: number | null, note = "",
                         source = "analyze") =>
  req<{ success: boolean; id?: string; error?: string }>("/api/recs", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol, score, price_at_rec, note, source }),
  });
export const recordRecOutcome = (
  recId: string,
  body: {
    real_pnl: number;
    real_acted?: boolean;
    real_strategy?: string;
    real_entry_price?: number | null;
    real_entry_date?: string;
    real_exit_price?: number | null;
    real_exit_date?: string;
    real_notes?: string;
  },
) =>
  req<{ success: boolean; id?: string; error?: string; won?: boolean | null }>(
    `/api/recs/${encodeURIComponent(recId)}/outcome`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
export const deleteRec = (id: string) =>
  req<{ success: boolean }>(`/api/recs/${encodeURIComponent(id)}`, { method: "DELETE" });

export interface EdgarFiling { form: string; label: string; date: string | null; url: string | null }
export const getEdgar = (symbol: string) =>
  req<{ success: boolean; symbol?: string; filings?: EdgarFiling[];
        signal?: Record<string, unknown> | null; note?: string; error?: string }>(
    `/api/edgar/${encodeURIComponent(symbol)}`,
  );

export interface WalkForwardFold {
  window: number; train_start: string; train_end: string;
  test_start: string; test_end: string;
  mae: number | null; mape: number | null; directional_accuracy: number | null;
}
