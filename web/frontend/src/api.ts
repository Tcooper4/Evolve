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
  req<{ symbol: string; candles: Candle[]; interval?: string }>(
    `/api/history/${symbol}?period=${encodeURIComponent(period)}${interval ? `&interval=${encodeURIComponent(interval)}` : ""}`,
  );

export type ChartEvent = {
  time: string;
  title?: string;
  text?: string;
  color?: string;
  headlines?: string[];
  volume_ratio?: number;
  price_change_pct?: number;
};

export const getChartEvents = (symbol: string, period = "6mo") =>
  req<{ success: boolean; events?: ChartEvent[]; error?: string }>(
    `/api/chart-events/${encodeURIComponent(symbol)}?period=${encodeURIComponent(period)}`,
  );

export const getCausal = (symbol: string) =>
  req<Record<string, unknown>>(`/api/causal/${encodeURIComponent(symbol)}`);

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

export const runMonteCarlo = (symbol: string, n_simulations = 400, horizon_days = 63) =>
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
  }>("/api/monte-carlo", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol, n_simulations, horizon_days }),
  });

export const getOptions = (symbol: string) =>
  req<Record<string, unknown>>(`/api/options/${encodeURIComponent(symbol)}`);

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

export const upsertAlert = (symbol: string, condition: string, threshold: number) =>
  req<{ success: boolean; alerts: Record<string, unknown>[] }>("/api/alerts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol, condition, threshold }),
  });

export const deleteAlert = (id: string) =>
  req<{ success: boolean; alerts: Record<string, unknown>[] }>(
    `/api/alerts/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );

export const getStrategies = () =>
  req<{ success: boolean; strategies: string[] }>("/api/strategies");

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
  req<{ anthropic: boolean; openai: boolean; news: boolean; reddit?: boolean }>(
    "/api/settings/keys",
  );

export const saveKeys = (k: {
  anthropic?: string; openai?: string; news?: string;
  reddit_client_id?: string; reddit_client_secret?: string;
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

export function quoteSocket(symbol: string,
                            onQuote: (q: { price: number | null; change_pct: number | null }) => void) {
  const t = sessionStorage.getItem("evolve_token") ?? "";
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/quote/${symbol}?token=${t}`);
  ws.onmessage = (e) => { try { onQuote(JSON.parse(e.data)); } catch { /* skip */ } };
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
  total_cost_basis: number;
  total_market_value: number;
  total_unrealized_pnl: number;
  realized_pnl: number;
  all_prices_live: boolean;
}
export const getPortfolio = () => req<PortfolioSummary>("/api/portfolio");
export const recordTrade = (symbol: string, side: "buy" | "sell",
                            quantity: number, price?: number) =>
  req<{ success: boolean; error?: string; realized_pnl?: number }>(
    "/api/portfolio/trade",
    { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol, side, quantity, price }) },
  );
export const getPortfolioTrades = () =>
  req<Record<string, unknown>[]>("/api/portfolio/trades");
