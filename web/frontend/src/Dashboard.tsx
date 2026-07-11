import { useCallback, useEffect, useRef, useState } from "react";
import {
  addToWatchlist,
  getHistory,
  getQuote,
  getWatchlist,
  removeFromWatchlist,
  type Candle,
  type Quote,
} from "./api";
import Chart from "./Chart";
import Sparkline from "./Sparkline";

const PERIODS = ["1mo", "3mo", "6mo", "1y"] as const;
type Period = (typeof PERIODS)[number];
const PERIOD_LABEL: Record<Period, string> = {
  "1mo": "1M", "3mo": "3M", "6mo": "6M", "1y": "1Y",
};

interface WlEntry {
  symbol: string;
  spark: number[];
  last: number | null;
}

export default function Dashboard({
  displayName,
  onLogout,
}: {
  displayName: string;
  onLogout: () => void;
}) {
  const [symbol, setSymbol] = useState("SPY");
  const [input, setInput] = useState("SPY");
  const [period, setPeriod] = useState<Period>("6mo");
  const [quote, setQuote] = useState<Quote | null>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [watchlist, setWatchlist] = useState<WlEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [flash, setFlash] = useState("");
  const searchRef = useRef<HTMLInputElement>(null);
  const prevPrice = useRef<number | null>(null);

  const load = useCallback(async (sym: string, per: Period) => {
    setLoading(true);
    try {
      const [q, h] = await Promise.all([getQuote(sym), getHistory(sym, per)]);
      setQuote(q);
      setCandles(h.candles);
      setSymbol(h.symbol);
      setInput(h.symbol);
      if (q.price != null && prevPrice.current != null && q.price !== prevPrice.current) {
        setFlash(q.price > prevPrice.current ? "flash-up" : "flash-down");
        setTimeout(() => setFlash(""), 700);
      }
      prevPrice.current = q.price;
    } catch {
      setQuote(null);
      setCandles([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshWatchlist = useCallback(async () => {
    try {
      const rows = await getWatchlist();
      const entries: WlEntry[] = await Promise.all(
        rows.map(async (r) => {
          try {
            const h = await getHistory(r.symbol, "1mo");
            const closes = h.candles.map((c) => c.close);
            return { symbol: r.symbol, spark: closes, last: closes.at(-1) ?? null };
          } catch {
            return { symbol: r.symbol, spark: [], last: null };
          }
        }),
      );
      setWatchlist(entries);
    } catch { /* handled by api client */ }
  }, []);

  useEffect(() => {
    load(symbol, period);
    refreshWatchlist();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "/" && document.activeElement?.tagName !== "INPUT") {
        e.preventDefault();
        searchRef.current?.focus();
        searchRef.current?.select();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const range = candles.length
    ? { lo: Math.min(...candles.map((c) => c.low)), hi: Math.max(...candles.map((c) => c.high)) }
    : null;
  const periodChange = candles.length > 1
    ? ((candles[candles.length - 1].close - candles[0].close) / candles[0].close) * 100
    : null;
  const up = (quote?.change_pct ?? 0) >= 0;

  return (
    <div className="app">
      <aside className="rail">
        <div className="brand">
          <span className="dot" /> EVOLVE <small>terminal</small>
        </div>
        <div className="user-chip">
          <span>{displayName}</span>
          <button className="ghost" onClick={onLogout}>Log out</button>
        </div>
        <div className="rail-label">Watchlist</div>
        <div className="wl">
          {watchlist.map((w) => (
            <div
              key={w.symbol}
              className={`wl-card fade-in ${w.symbol === symbol ? "active" : ""}`}
              onClick={() => load(w.symbol, period)}
            >
              <span className="wl-sym">{w.symbol}</span>
              <Sparkline values={w.spark} />
              <span className="wl-px num">
                {w.last != null ? w.last.toFixed(2) : "—"}
                <span
                  className="wl-x"
                  style={{ marginLeft: 8 }}
                  onClick={async (e) => {
                    e.stopPropagation();
                    await removeFromWatchlist(w.symbol);
                    refreshWatchlist();
                  }}
                >
                  ✕
                </span>
              </span>
            </div>
          ))}
          {watchlist.length === 0 && (
            <div className="empty" style={{ padding: "24px 0" }}>
              No symbols yet
            </div>
          )}
        </div>
      </aside>

      <main className="main">
        <div className="topbar">
          <div className="search">
            <span className="icon">⌕</span>
            <input
              ref={searchRef}
              value={input}
              onChange={(e) => setInput(e.target.value.toUpperCase())}
              onKeyDown={(e) => e.key === "Enter" && load(input, period)}
              placeholder="Search symbol — SPX, AAPL, BTC, ES…"
            />
            <kbd>/</kbd>
          </div>
          <button className="primary" onClick={() => load(input, period)}>
            Load
          </button>
          <button
            onClick={async () => {
              await addToWatchlist(symbol);
              refreshWatchlist();
            }}
          >
            + Watch
          </button>
        </div>

        <div className="kpis">
          <div className="card kpi fade-in">
            <div className="label">{symbol}</div>
            {loading ? (
              <div className="skeleton" style={{ height: 40, width: 140 }} />
            ) : (
              <div className={`hero-price num ${flash}`}>
                {quote?.price != null ? quote.price.toFixed(2) : "—"}
              </div>
            )}
            {quote?.change_pct != null && (
              <div style={{ marginTop: 8 }}>
                <span className={`delta num ${up ? "up" : "down"}`}>
                  {up ? "▲" : "▼"} {Math.abs(quote.change_pct).toFixed(2)}%
                </span>
              </div>
            )}
          </div>
          <div className="card kpi fade-in">
            <div className="label">Prev close</div>
            <div className="value num">
              {quote?.prev_close != null ? quote.prev_close.toFixed(2) : "—"}
            </div>
            <div className="sub">last session</div>
          </div>
          <div className="card kpi fade-in">
            <div className="label">{PERIOD_LABEL[period]} range</div>
            <div className="value num" style={{ fontSize: 18 }}>
              {range ? `${range.lo.toFixed(2)} – ${range.hi.toFixed(2)}` : "—"}
            </div>
            <div className="sub">low – high</div>
          </div>
          <div className="card kpi fade-in">
            <div className="label">{PERIOD_LABEL[period]} change</div>
            <div
              className="value num"
              style={{
                color: periodChange == null ? undefined
                  : periodChange >= 0 ? "var(--up)" : "var(--down)",
              }}
            >
              {periodChange != null
                ? `${periodChange >= 0 ? "+" : ""}${periodChange.toFixed(2)}%`
                : "—"}
            </div>
            <div className="sub">close vs close</div>
          </div>
        </div>

        <div className="card fade-in">
          <div className="chart-head">
            <div className="legend"><b>{symbol}</b> · daily</div>
            <div className="seg">
              {PERIODS.map((p) => (
                <button
                  key={p}
                  className={p === period ? "active" : ""}
                  onClick={() => {
                    setPeriod(p);
                    load(symbol, p);
                  }}
                >
                  {PERIOD_LABEL[p]}
                </button>
              ))}
            </div>
          </div>
          {loading ? (
            <div className="skeleton" style={{ height: 460, margin: 18 }} />
          ) : candles.length > 0 ? (
            <Chart candles={candles} />
          ) : (
            <div className="empty">
              No chart data — check the symbol or your connection.
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
