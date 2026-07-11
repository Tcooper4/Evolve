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

export const getHistory = (symbol: string, period = "6mo") =>
  req<{ symbol: string; candles: Candle[] }>(
    `/api/history/${symbol}?period=${period}`,
  );

export const getWatchlist = () => req<WatchlistRow[]>("/api/watchlist");

export const addToWatchlist = (symbol: string) =>
  req<{ ok: boolean }>("/api/watchlist", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol }),
  });

export const removeFromWatchlist = (symbol: string) =>
  req<{ ok: boolean }>(`/api/watchlist/${symbol}`, { method: "DELETE" });
