import { useCallback, useEffect, useRef, useState } from "react";
import {
  addToWatchlist,
  getBreakingNews,
  getChartEvents,
  getHistory,
  getNews,
  getQuote,
  getWatchlist,
  quoteSocket,
  removeFromWatchlist,
  runBriefing,
  type Candle,
  type ChartEvent,
  type Quote,
} from "./api";
import Chart from "./Chart";
import Sparkline from "./Sparkline";

const PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "max"] as const;
type Period = (typeof PERIODS)[number];
const PERIOD_LABEL: Record<Period, string> = {
  "1d": "1D", "5d": "1W", "1mo": "1M", "3mo": "3M",
  "6mo": "6M", "1y": "1Y", "max": "ALL",
};

const DAY_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "1h"] as const;
type DayInterval = (typeof DAY_INTERVALS)[number];
const INTERVAL_LABEL: Record<DayInterval, string> = {
  "1m": "1m", "2m": "2m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h",
};

interface WlEntry {
  symbol: string;
  spark: number[];
  last: number | null;
}

export default function Dashboard({
  displayName,
  onAnalyze,
}: {
  displayName: string;
  onAnalyze?: (symbol: string) => void;
}) {
  const [symbol, setSymbol] = useState("SPY");
  const [input, setInput] = useState("SPY");
  const [period, setPeriod] = useState<Period>("6mo");
  const [dayInterval, setDayInterval] = useState<DayInterval>("5m");
  const [quote, setQuote] = useState<Quote | null>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [events, setEvents] = useState<ChartEvent[]>([]);
  const [watchlist, setWatchlist] = useState<WlEntry[]>([]);
  const [news, setNews] = useState<Record<string, unknown>[]>([]);
  const [breaking, setBreaking] = useState<Record<string, unknown>[]>([]);
  const [brief, setBrief] = useState<Record<string, unknown> | null>(null);
  const [briefBusy, setBriefBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [flash, setFlash] = useState("");
  const searchRef = useRef<HTMLInputElement>(null);
  const prevPrice = useRef<number | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const load = useCallback(async (sym: string, per: Period, iv?: DayInterval) => {
    setLoading(true);
    try {
      const interval = per === "1d" ? (iv ?? dayInterval) : "";
      const [q, h, n, ev, br] = await Promise.all([
        getQuote(sym),
        getHistory(sym, per, interval),
        getNews(sym, 5).catch(() => ({ items: [] })),
        getChartEvents(sym, per).catch(() => ({ events: [] })),
        getBreakingNews(5).catch(() => ({ items: [] })),
      ]);
      setQuote(q);
      setCandles(h.candles);
      setEvents(ev.events ?? []);
      setSymbol(h.symbol);
      setInput(h.symbol);
      setNews((n.items as Record<string, unknown>[]) ?? []);
      setBreaking((br.items as Record<string, unknown>[]) ?? []);
      if (q.price != null && prevPrice.current != null && q.price !== prevPrice.current) {
        setFlash(q.price > prevPrice.current ? "flash-up" : "flash-down");
        setTimeout(() => setFlash(""), 700);
      }
      prevPrice.current = q.price;
      wsRef.current?.close();
      wsRef.current = quoteSocket(h.symbol, (wq) => {
        if (wq.price == null) return;
        setQuote((prev) => prev ? { ...prev, price: wq.price, change_pct: wq.change_pct } : prev);
        if (prevPrice.current != null && wq.price !== prevPrice.current) {
          setFlash(wq.price > prevPrice.current ? "flash-up" : "flash-down");
          setTimeout(() => setFlash(""), 700);
        }
        prevPrice.current = wq.price;
      });
    } catch {
      setQuote(null);
      setCandles([]);
    } finally {
      setLoading(false);
    }
  }, [dayInterval]);

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
    return () => {
      window.removeEventListener("keydown", onKey);
      wsRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function generateBriefing() {
    setBriefBusy(true);
    try {
      const r = await runBriefing("sp100", 6.0);
      setBrief(r as unknown as Record<string, unknown>);
    } catch (e) {
      setBrief({ success: false, error: e instanceof Error ? e.message : "failed" });
    } finally {
      setBriefBusy(false);
    }
  }

  const range = candles.length
    ? { lo: Math.min(...candles.map((c) => c.low)), hi: Math.max(...candles.map((c) => c.high)) }
    : null;
  const periodChange = candles.length > 1
    ? ((candles[candles.length - 1].close - candles[0].close) / candles[0].close) * 100
    : null;
  const up = (quote?.change_pct ?? 0) >= 0;
  const hour = new Date().getHours();
  const greet = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
  const opps = (brief?.top_opportunities as Record<string, unknown>[] | undefined) ?? [];

  return (
    <div className="fade-in">
      <div className="greeting">{greet}, {displayName} <small>markets at a glance — live</small></div>

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
        <button className="primary" onClick={() => load(input, period)}>Load</button>
        <button onClick={async () => { await addToWatchlist(symbol); refreshWatchlist(); }}>+ Watch</button>
        {onAnalyze && (
          <button onClick={() => onAnalyze(symbol)}>Analyze</button>
        )}
        <button onClick={generateBriefing} disabled={briefBusy}>
          {briefBusy ? "Briefing…" : "Morning briefing"}
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
          <div className="value num" style={{
            color: periodChange == null ? undefined
              : periodChange >= 0 ? "var(--up)" : "var(--down)",
          }}>
            {periodChange != null
              ? `${periodChange >= 0 ? "+" : ""}${periodChange.toFixed(2)}%`
              : "—"}
          </div>
          <div className="sub">close vs close</div>
        </div>
      </div>

      <div className="card fade-in">
        <div className="chart-head">
          <div className="legend">
            <b>{symbol}</b>
            {" · "}
            {period === "1d" || period === "5d" || period === "1mo" || period === "3mo"
              ? (period === "1d" ? dayInterval : "intraday")
              : "daily"}
            {events.length > 0 ? ` · ${events.length} news marks` : ""}
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            {period === "1d" && (
              <div className="seg">
                {DAY_INTERVALS.map((iv) => (
                  <button key={iv} className={iv === dayInterval ? "active" : ""}
                    onClick={() => { setDayInterval(iv); load(symbol, "1d", iv); }}>
                    {INTERVAL_LABEL[iv]}
                  </button>
                ))}
              </div>
            )}
            <div className="seg">
              {PERIODS.map((p) => (
                <button key={p} className={p === period ? "active" : ""}
                  onClick={() => { setPeriod(p); load(symbol, p); }}>
                  {PERIOD_LABEL[p]}
                </button>
              ))}
            </div>
          </div>
        </div>
        {loading ? (
          <div className="skeleton" style={{ height: 460, margin: 18 }} />
        ) : candles.length > 0 ? (
          <Chart candles={candles} markers={events.map((e) => ({
            time: e.time,
            title: e.title,
            text: e.text,
            color: e.color,
          }))} />
        ) : (
          <div className="empty">No chart data — check the symbol or your connection.</div>
        )}
      </div>

      <div className="form-grid" style={{ marginTop: 16 }}>
        <div className="card card-pad">
          <div className="rail-label" style={{ marginTop: 0 }}>Headlines · {symbol}</div>
          {news.length === 0 && <div className="dim">No headlines yet.</div>}
          {news.slice(0, 5).map((n, i) => (
            <div key={i} style={{ padding: "8px 0", borderTop: i ? "1px solid var(--border)" : "none" }}>
              <div style={{ fontWeight: 600, fontSize: 13.5 }}>
                {n.url ? (
                  <a href={String(n.url)} target="_blank" rel="noreferrer" style={{ color: "var(--text)", textDecoration: "none" }}>
                    {String(n.title ?? n.headline ?? "Untitled")}
                  </a>
                ) : String(n.title ?? n.headline ?? "Untitled")}
              </div>
              <div className="dim" style={{ fontSize: 11.5, marginTop: 3 }}>
                {String(n.source ?? "")}
                {n.source_type ? ` · ${String(n.source_type)}` : ""}
                {n.url ? <> · <a href={String(n.url)} target="_blank" rel="noreferrer" style={{ color: "var(--accent)" }}>open</a></> : null}
              </div>
            </div>
          ))}
          {breaking.length > 0 && (
            <>
              <div className="rail-label" style={{ marginTop: 16 }}>Breaking</div>
              {breaking.slice(0, 4).map((n, i) => (
                <div key={`b-${i}`} style={{ padding: "6px 0", borderTop: i ? "1px solid var(--border)" : "none" }}>
                  <div style={{ fontSize: 13 }}>
                    {n.url ? (
                      <a href={String(n.url)} target="_blank" rel="noreferrer" style={{ color: "var(--text)", textDecoration: "none" }}>
                        {String(n.title ?? "").slice(0, 160)}
                      </a>
                    ) : String(n.title ?? "").slice(0, 160)}
                  </div>
                  <div className="dim" style={{ fontSize: 11, marginTop: 2 }}>{String(n.source ?? "wire")}</div>
                </div>
              ))}
            </>
          )}
        </div>
        <div className="card card-pad">
          <div className="rail-label" style={{ marginTop: 0 }}>Morning briefing</div>
          {!brief && !briefBusy && (
            <div className="dim">Generate a briefing to surface top long/short candidates from the S&P 100.</div>
          )}
          {briefBusy && <div className="skeleton" style={{ height: 120 }} />}
          {brief && !briefBusy && (
            <>
              {brief.error && <div className="dim">⚠ {String(brief.error)}</div>}
              {opps.length === 0 && !brief.error && <div className="dim">No opportunities above threshold.</div>}
              {opps.slice(0, 5).map((o, i) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "8px 0",
                  borderTop: i ? "1px solid var(--border)" : "none", cursor: "pointer" }}
                  onClick={() => load(String(o.symbol || ""), period)}>
                  <span style={{ fontWeight: 650 }}>{String(o.symbol)}</span>
                  <span className="num dim">
                    {o.ai_score != null ? Number(o.ai_score).toFixed(1) : o.score != null ? Number(o.score).toFixed(1) : "—"}
                  </span>
                </div>
              ))}
            </>
          )}
        </div>
      </div>

      <div className="rail-label" style={{ margin: "18px 0 8px" }}>Watchlist</div>
      <div className="wl" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 10 }}>
        {watchlist.map((w) => (
          <div key={w.symbol}
            className={`wl-card fade-in ${w.symbol === symbol ? "active" : ""}`}
            onClick={() => load(w.symbol, period)}>
            <span className="wl-sym">{w.symbol}</span>
            <Sparkline values={w.spark} baseline="start" />
            <span className="wl-px num">
              {w.last != null ? w.last.toFixed(2) : "—"}
              <span className="wl-x" style={{ marginLeft: 8 }}
                onClick={async (e) => {
                  e.stopPropagation();
                  await removeFromWatchlist(w.symbol);
                  refreshWatchlist();
                }}>✕</span>
            </span>
          </div>
        ))}
        {watchlist.length === 0 && <div className="dim">No symbols yet — search above and hit + Watch.</div>}
      </div>
    </div>
  );
}
