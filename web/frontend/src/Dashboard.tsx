import { useCallback, useEffect, useState } from "react";
import {
  addToWatchlist,
  getHistory,
  getQuote,
  getWatchlist,
  removeFromWatchlist,
  type Candle,
  type Quote,
  type WatchlistRow,
} from "./api";
import Chart from "./Chart";

export default function Dashboard({
  displayName,
  onLogout,
}: {
  displayName: string;
  onLogout: () => void;
}) {
  const [symbol, setSymbol] = useState("SPY");
  const [input, setInput] = useState("SPY");
  const [quote, setQuote] = useState<Quote | null>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [watchlist, setWatchlist] = useState<WatchlistRow[]>([]);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async (sym: string) => {
    setLoading(true);
    try {
      const [q, h] = await Promise.all([getQuote(sym), getHistory(sym)]);
      setQuote(q);
      setCandles(h.candles);
      setSymbol(h.symbol);
    } catch {
      setQuote(null);
      setCandles([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshWatchlist = useCallback(async () => {
    try {
      setWatchlist(await getWatchlist());
    } catch {
      /* token expiry handled by api client */
    }
  }, []);

  useEffect(() => {
    load(symbol);
    refreshWatchlist();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const changeCls =
    quote?.change_pct == null ? "" : quote.change_pct >= 0 ? "up" : "down";

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">EVOLVE</div>
        <div className="dim">Signed in as {displayName}</div>
        <button className="ghost" onClick={onLogout}>
          Log out
        </button>
        <div className="dim" style={{ marginTop: 16 }}>
          Watchlist
        </div>
        {watchlist.map((w) => (
          <div key={w.symbol} className="wl-item" onClick={() => load(w.symbol)}>
            <span>{w.symbol}</span>
            <span
              className="dim"
              onClick={async (e) => {
                e.stopPropagation();
                await removeFromWatchlist(w.symbol);
                refreshWatchlist();
              }}
            >
              ✕
            </span>
          </div>
        ))}
        {watchlist.length === 0 && <div className="dim">Empty</div>}
      </aside>

      <main className="main">
        <div className="row" style={{ marginBottom: 16 }}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value.toUpperCase())}
            onKeyDown={(e) => e.key === "Enter" && load(input)}
            placeholder="Symbol (SPX, AAPL, BTC…)"
          />
          <button className="primary" onClick={() => load(input)}>
            {loading ? "Loading…" : "Load"}
          </button>
          <button
            onClick={async () => {
              await addToWatchlist(symbol);
              refreshWatchlist();
            }}
          >
            + Watchlist
          </button>
        </div>

        <div className="card" style={{ marginBottom: 16 }}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <div>
              <div className="dim">{symbol}</div>
              <div className="quote-price">
                {quote?.price != null ? quote.price.toFixed(2) : "—"}
              </div>
            </div>
            <div className={changeCls} style={{ fontSize: 20 }}>
              {quote?.change_pct != null
                ? `${quote.change_pct >= 0 ? "+" : ""}${quote.change_pct.toFixed(2)}%`
                : ""}
            </div>
          </div>
        </div>

        <div className="card">
          {candles.length > 0 ? (
            <Chart candles={candles} />
          ) : (
            <div className="dim">
              No chart data{loading ? "…" : " — check the symbol or network."}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
