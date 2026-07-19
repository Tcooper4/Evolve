import { useCallback, useEffect, useRef, useState } from "react";
import {
  addToWatchlist,
  getBreakingNews,
  getChartEvents,
  getOptionsStructureOverlay,
  getHistory,
  getMarketSignals,
  getMarketState,
  getNews,
  getNewsContext,
  getQuote,
  getScore,
  getWatchlist,
  getPrefs,
  getStrategies,
  getStrategyOverlay,
  quoteSocket,
  removeFromWatchlist,
  runBriefing,
  type Candle,
  type ChartEvent,
  type GprSignal,
  type MarketState,
  type Quote,
  type RevisionBreadth,
  type OptionsStructureOverlay,
  type StrategyOverlay,
} from "./api";
import Chart from "./Chart";
import Sparkline from "./Sparkline";
import {
  alignOverlaySeriesToCandles,
  describeEventMark,
  filterMarkersToCandles,
} from "./chartMarkers";
import { loadCachedChartTimezone, cacheChartTimezone } from "./chartTime";
import { runDashboardLoad } from "./dashboardLoad";
import PageTour from "./PageTour";

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
  score: number | null;
  grade: string | null;
  scoreLoading?: boolean;
}

/** Yahoo often leaves Volume=0 on the open bar; infer from session volume. */
function withLiveBarVolume(
  candles: Candle[],
  sessionVol: number | null | undefined,
): Candle[] {
  if (!candles.length || sessionVol == null || !(Number(sessionVol) > 0)) {
    return candles;
  }
  const last = candles[candles.length - 1];
  if (Number(last.volume) > 0) return candles;
  const day = last.time.slice(0, 10);
  let prior = 0;
  for (let i = 0; i < candles.length - 1; i++) {
    if (candles[i].time.slice(0, 10) === day) {
      prior += Number(candles[i].volume) || 0;
    }
  }
  const inferred = Math.max(0, Number(sessionVol) - prior);
  if (!(inferred > 0)) return candles;
  const next = candles.slice();
  next[next.length - 1] = { ...last, volume: inferred };
  return next;
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
  const [chartHint, setChartHint] = useState<string | null>(null);
  const [events, setEvents] = useState<ChartEvent[]>([]);
  const [liveSpike, setLiveSpike] = useState<ChartEvent | null>(null);
  const [watchlist, setWatchlist] = useState<WlEntry[]>([]);
  const [news, setNews] = useState<Record<string, unknown>[]>([]);
  const [newsWhy, setNewsWhy] = useState<Record<string, string>>({});
  const [breaking, setBreaking] = useState<Record<string, unknown>[]>([]);
  const [brief, setBrief] = useState<Record<string, unknown> | null>(null);
  const [briefBusy, setBriefBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [flash, setFlash] = useState("");
  const [gpr, setGpr] = useState<GprSignal | null>(null);
  const [rb, setRb] = useState<RevisionBreadth | null>(null);
  const [marketState, setMarketState] = useState<MarketState | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const prevPrice = useRef<number | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const loadGenRef = useRef(0);
  const symbolRef = useRef(symbol);
  const periodRef = useRef(period);
  const dayIntervalRef = useRef(dayInterval);
  const [chartLive, setChartLive] = useState(false);
  const [chartTimezone, setChartTimezone] = useState(loadCachedChartTimezone);
  // Strategy overlay — OFF by default (no OOS default-on)
  const [overlayOn, setOverlayOn] = useState(false);
  const [overlayStrategy, setOverlayStrategy] = useState("RSIStrategy");
  const [strategies, setStrategies] = useState<string[]>([
    "RSIStrategy", "MACDStrategy", "BollingerStrategy", "SMAStrategy",
  ]);
  const [overlay, setOverlay] = useState<StrategyOverlay | null>(null);
  const [overlayBusy, setOverlayBusy] = useState(false);
  // Options structure guide — also OFF by default (research only)
  const [optOverlayOn, setOptOverlayOn] = useState(false);
  const [optOverlay, setOptOverlay] = useState<OptionsStructureOverlay | null>(null);
  const [optOverlayBusy, setOptOverlayBusy] = useState(false);

  useEffect(() => { symbolRef.current = symbol; }, [symbol]);
  useEffect(() => { periodRef.current = period; }, [period]);
  useEffect(() => { dayIntervalRef.current = dayInterval; }, [dayInterval]);

  useEffect(() => {
    getPrefs().then((r) => {
      const tz = r.prefs?.chart_timezone;
      if (typeof tz === "string" && tz) {
        setChartTimezone(tz);
        cacheChartTimezone(tz);
      }
    }).catch(() => {});
    getStrategies().then((r) => {
      if (r.strategies?.length) setStrategies(r.strategies);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!overlayOn) {
      setOverlay(null);
      return;
    }
    let cancelled = false;
    setOverlayBusy(true);
    const per = period === "1d" || period === "5d" ? "3mo" : period;
    getStrategyOverlay(symbol, overlayStrategy, per)
      .then((r) => { if (!cancelled) setOverlay(r); })
      .catch((e) => {
        if (!cancelled) {
          setOverlay({
            success: false,
            error: e instanceof Error ? e.message : "overlay failed",
          });
        }
      })
      .finally(() => { if (!cancelled) setOverlayBusy(false); });
    return () => { cancelled = true; };
  }, [overlayOn, overlayStrategy, symbol, period]);

  useEffect(() => {
    if (!optOverlayOn) {
      setOptOverlay(null);
      setOptOverlayBusy(false);
      return;
    }
    let cancelled = false;
    setOptOverlayBusy(true);
    getOptionsStructureOverlay(symbol)
      .then((r) => { if (!cancelled) setOptOverlay(r); })
      .catch((e) => {
        if (!cancelled) {
          setOptOverlay({
            success: false,
            error: e instanceof Error ? e.message : "options overlay failed",
          });
        }
      })
      .finally(() => { if (!cancelled) setOptOverlayBusy(false); });
    return () => { cancelled = true; };
  }, [optOverlayOn, symbol]);

  const load = useCallback(async (sym: string, per: Period, iv?: DayInterval, soft = false) => {
    const gen = ++loadGenRef.current;
    if (!soft) setLoading(true);
    const interval = per === "1d" ? (iv ?? dayIntervalRef.current) : "";

    // Critical path (quote+history+news) clears the spinner; chart-events /
    // breaking / market-state overlay in a second pass so a cold GDELT fan-out
    // cannot block first paint. Soft polls still skip that overlay entirely.
    await runDashboardLoad({
      symbol: sym,
      period: per,
      interval,
      soft,
      fetchers: {
        getQuote,
        getHistory,
        getNews,
        getChartEvents,
        getBreakingNews,
        getMarketState,
      },
      hooks: {
        onCritical: ({ quote: q, history: h, news: n }) => {
          if (gen !== loadGenRef.current) return;
          setQuote(q);
          setCandles(withLiveBarVolume(h.candles as Candle[], q.volume));
          if (!soft) {
            // Clear stale marks until overlay arrives for this symbol
            setEvents([]);
            setLiveSpike(null);
            if (!h.candles?.length) {
              const sug = h.suggestion ? String(h.suggestion) : null;
              setChartHint(
                sug
                  ? `No chart data for ${h.symbol}. Did you mean ${sug}?`
                  : `No chart data for ${h.symbol} — check the symbol or your connection.`,
              );
            } else {
              setChartHint(null);
            }
          }
          setSymbol(h.symbol);
          setInput(h.symbol);

          if (soft) {
            if (q.price != null) prevPrice.current = q.price;
            // Keep existing websocket; only refresh bar history (+ live volume)
            return;
          }

          setNews((n.items as Record<string, unknown>[]) ?? []);
          setNewsWhy({});
          const titles = ((n.items as Record<string, unknown>[]) ?? [])
            .map((it) => String(it.title ?? it.headline ?? "").trim())
            .filter(Boolean)
            .slice(0, 5);
          if (titles.length) {
            getNewsContext(titles)
              .then((ctx) => {
                if (gen !== loadGenRef.current) return;
                const map: Record<string, string> = {};
                for (const it of ctx.items ?? []) {
                  if (it.title && it.why) map[it.title] = it.why;
                }
                setNewsWhy(map);
              })
              .catch(() => {});
          }
          if (q.price != null && prevPrice.current != null && q.price !== prevPrice.current) {
            setFlash(q.price > prevPrice.current ? "flash-up" : "flash-down");
            setTimeout(() => setFlash(""), 700);
          }
          prevPrice.current = q.price;
          wsRef.current?.close();
          setChartLive(true);
          wsRef.current = quoteSocket(h.symbol, (wq) => {
            if (wq.type === "volume_spike") {
              if (wq.active === false) {
                setLiveSpike(null);
                return;
              }
              const fallback = wq.link_quality === "fallback_recent";
              const honesty = fallback ? " · may not be same-day headline" : "";
              const baseTitle = String(wq.title ?? "Provisional live volume spike");
              setLiveSpike({
                time: String(wq.time ?? new Date().toISOString().slice(0, 10)),
                title: `${baseTitle}${honesty.includes("may not") && !baseTitle.includes("may not") ? honesty : ""}`,
                text: String(wq.text ?? "LIVE"),
                color: String(wq.color ?? "#F5A623"),
                shape: typeof wq.shape === "string" ? wq.shape : "circle",
                provisional: true,
                link_quality: typeof wq.link_quality === "string" ? wq.link_quality : undefined,
                date_confirmed: Boolean(wq.date_confirmed),
                volume_ratio: typeof wq.volume_ratio === "number" ? wq.volume_ratio : undefined,
                price_change_pct: typeof wq.price_change_pct === "number" ? wq.price_change_pct : undefined,
              });
              return;
            }
            if (wq.price == null) return;
            setQuote((prev) => prev ? { ...prev, price: wq.price as number, change_pct: wq.change_pct as number | null } : prev);
            setCandles((prev) => {
              if (!prev.length) return prev;
              const next = prev.slice();
              const last = { ...next[next.length - 1] };
              const px = Number(wq.price);
              last.close = px;
              last.high = Math.max(last.high, px);
              last.low = Math.min(last.low, px);
              next[next.length - 1] = last;
              return next;
            });
            if (prevPrice.current != null && wq.price !== prevPrice.current) {
              setFlash(wq.price > prevPrice.current ? "flash-up" : "flash-down");
              setTimeout(() => setFlash(""), 700);
            }
            prevPrice.current = wq.price as number;
          });
          setLoading(false);
        },
        onOverlay: ({ events: ev, breaking: br, marketState: ms }) => {
          if (gen !== loadGenRef.current) return;
          setEvents(ev as ChartEvent[]);
          setBreaking((br as Record<string, unknown>[]) ?? []);
          setMarketState(ms as MarketState | null);
        },
        onCriticalError: () => {
          if (gen !== loadGenRef.current) return;
          if (!soft) {
            setQuote(null);
            setCandles([]);
            setChartHint("No chart data — check the symbol or your connection.");
            setLoading(false);
          }
          setChartLive(false);
        },
      },
    });
  }, []);

  const refreshWatchlist = useCallback(async () => {
    try {
      const rows = await getWatchlist();
      const entries: WlEntry[] = await Promise.all(
        rows.map(async (r) => {
          try {
            const h = await getHistory(r.symbol, "1mo");
            const closes = h.candles.map((c) => c.close);
            return {
              symbol: r.symbol, spark: closes, last: closes.at(-1) ?? null,
              score: null, grade: null, scoreLoading: true,
            };
          } catch {
            return {
              symbol: r.symbol, spark: [], last: null,
              score: null, grade: null, scoreLoading: true,
            };
          }
        }),
      );
      setWatchlist(entries);
      // Scores load after sparks so the board paints fast
      void Promise.all(
        entries.map(async (e) => {
          try {
            const s = await getScore(e.symbol, "long");
            setWatchlist((prev) => prev.map((w) =>
              w.symbol === e.symbol
                ? {
                    ...w,
                    score: s.score != null ? Number(s.score) : null,
                    grade: s.grade != null ? String(s.grade) : null,
                    scoreLoading: false,
                  }
                : w,
            ));
          } catch {
            setWatchlist((prev) => prev.map((w) =>
              w.symbol === e.symbol ? { ...w, scoreLoading: false } : w,
            ));
          }
        }),
      );
    } catch { /* handled by api client */ }
  }, []);

  useEffect(() => {
    load(symbol, period);
    refreshWatchlist();
    getMarketSignals()
      .then((s) => { setGpr(s.gpr); setRb(s.revision_breadth); })
      .catch(() => {});
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

  // Soft-refresh candles so new bars appear without flipping the period
  useEffect(() => {
    const intraday = period === "1d" || period === "5d" || period === "1mo" || period === "3mo";
    const ms = period === "1d" ? 20_000 : intraday ? 45_000 : 120_000;
    const id = window.setInterval(() => {
      void load(symbolRef.current, periodRef.current, dayIntervalRef.current, true);
    }, ms);
    return () => window.clearInterval(id);
  }, [period, load]);

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

  // Only count/plot marks that land on days actually present in this chart.
  // (API may use a longer lookback for the 20d volume baseline.)
  const visibleNews = filterMarkersToCandles(
    events.map((e) => {
      const legend = describeEventMark(e);
      const fallback = e.link_quality === "fallback_recent";
      const timedOut = e.link_quality === "news_lookup_timeout";
      const honesty = timedOut
        ? " · headlines timed out (volume mark only)"
        : fallback
          ? " · may not be same-day headline"
          : "";
      const headline = e.title ? `${e.title}${honesty}` : "";
      return {
        time: e.time,
        // Legend first so hover panel always explains letter + color
        title: headline ? `${legend} · ${headline}` : legend,
        text: e.text ?? (e.tier === "notable" ? "n" : e.tier === "event_move" ? "E" : e.tier === "provisional" ? "LIVE" : "N"),
        color: e.color,
        shape: "circle" as const,
      };
    }),
    candles,
  );
  const visibleStrategy = filterMarkersToCandles(
    (overlayOn && overlay?.success && overlay.markers) ? overlay.markers : [],
    candles,
  );
  const visibleOptMarks = filterMarkersToCandles(
    (optOverlayOn && optOverlay?.success && optOverlay.markers) ? optOverlay.markers : [],
    candles,
  );
  const visibleOverlays = alignOverlaySeriesToCandles(
    [
      ...((overlayOn && overlay?.success && overlay.overlay_series)
        ? overlay.overlay_series : []),
      ...((optOverlayOn && optOverlay?.success && optOverlay.overlay_series)
        ? optOverlay.overlay_series : []),
    ],
    candles,
  );

  const up = (quote?.change_pct ?? 0) >= 0;
  const hour = new Date().getHours();
  const greet = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
  const opps = (brief?.top_opportunities as Record<string, unknown>[] | undefined) ?? [];

  return (
    <div className="fade-in">
      <PageTour pageId="dashboard" />
      <div className="greeting">{greet}, {displayName} <small>markets at a glance — live</small></div>

      <div className="topbar" data-tour="dashboard-controls">
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

      <div className="kpis" data-tour="dashboard-pulse">
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
        <div className="card kpi fade-in" title={marketState?.disclosure || "Situational awareness — not a price prediction"}>
          <div className="label">Market state</div>
          <div className="value num" style={{
            fontSize: 16,
            color:
              marketState?.level === "critical" || marketState?.level === "elevated"
                ? "var(--down)"
                : marketState?.level === "calm"
                  ? "var(--up)"
                  : undefined,
          }}>
            {marketState?.level
              ? marketState.level.toUpperCase()
              : loading ? "…" : "—"}
          </div>
          <div className="sub" style={{ maxWidth: 220 }}>
            {marketState?.label
              || (marketState?.error ? "unavailable" : "GEX · news · vol")}
          </div>
        </div>
        <div className="card kpi fade-in" title={gpr?.description || "Load in Settings → Market signals"}>
          <div className="label">Geopolitical risk</div>
          <div className="value num" style={{
            fontSize: 18,
            color: gpr?.level === "HIGH" || gpr?.level === "ELEVATED" ? "var(--down)" : undefined,
          }}>
            {gpr?.current != null ? `${Number(gpr.current).toFixed(0)}` : "—"}
            {gpr?.level ? <span className="dim" style={{ fontSize: 12, marginLeft: 6 }}>{gpr.level}</span> : null}
          </div>
          <div className="sub">{gpr?.trend || "Caldara & Iacoviello · Settings to load"}</div>
        </div>
        <div className="card kpi fade-in" title={rb?.description || "Load in Settings → Market signals"}>
          <div className="label">EPS revision breadth</div>
          <div className="value num" style={{
            fontSize: 16,
            color: rb?.signal === "POSITIVE" ? "var(--up)"
              : rb?.signal === "NEGATIVE" ? "var(--down)" : undefined,
          }}>
            {rb ? `${rb.pct_up.toFixed(0)}% ↑ / ${rb.pct_down.toFixed(0)}% ↓` : "—"}
          </div>
          <div className="sub">{rb?.signal || "Settings to compute"}</div>
        </div>
      </div>

      <div className="card fade-in" data-tour="dashboard-chart">
        <div className="chart-head">
          <div className="legend">
            <b>{symbol}</b>
            {" · "}
            {period === "1d" || period === "5d" || period === "1mo" || period === "3mo"
              ? (period === "1d" ? dayInterval : "intraday")
              : "daily"}
            {visibleNews.length > 0
              ? ` · ${visibleNews.length} news mark${visibleNews.length === 1 ? "" : "s"}`
              : (period === "1d" || period === "5d")
                ? " · no news marks in this session window"
                : " · no news marks"}
            {overlayOn && overlay?.success
              ? ` · ${visibleStrategy.length} backtest signal${visibleStrategy.length === 1 ? "" : "s"}`
              : ""}
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            <label className="dim" style={{ fontSize: 12, display: "inline-flex", gap: 6, alignItems: "center" }}>
              <input
                type="checkbox"
                checked={overlayOn}
                onChange={(e) => setOverlayOn(e.target.checked)}
              />
              Strategy overlay
            </label>
            {overlayOn && (
              <select
                value={overlayStrategy}
                onChange={(e) => setOverlayStrategy(e.target.value)}
                style={{ fontSize: 12 }}
              >
                {strategies.map((s) => (
                  <option key={s} value={s}>{s.replace(/Strategy$/, "")}</option>
                ))}
              </select>
            )}
            <label
              className="dim"
              style={{ fontSize: 12, display: "inline-flex", gap: 6, alignItems: "center" }}
              title="Research guide from delayed options data — marks land on the day’s last bar"
            >
              <input
                type="checkbox"
                checked={optOverlayOn}
                onChange={(e) => setOptOverlayOn(e.target.checked)}
              />
              Options structure
            </label>
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
          <Chart
            live={chartLive && (period === "1d" || period === "5d")}
            timeZone={chartTimezone}
            candles={candles}
            overlays={visibleOverlays}
            markers={[
              ...visibleNews,
              ...filterMarkersToCandles(
                liveSpike ? [{
                  time: liveSpike.time,
                  title: describeEventMark({ ...liveSpike, tier: "provisional" }),
                  text: liveSpike.text ?? "LIVE",
                  color: liveSpike.color ?? "#F5A623",
                  shape: (liveSpike.shape as "circle" | "square" | "arrowUp" | "arrowDown" | undefined) ?? "circle",
                }] : [],
                candles,
              ),
              ...visibleStrategy,
              ...visibleOptMarks,
            ]}
          />
        ) : (
          <div className="empty">
            {chartHint ?? "No chart data — check the symbol or your connection."}
            {chartHint?.includes("Did you mean") && (() => {
              const m = /Did you mean ([A-Z0-9.^=-]+)\?/.exec(chartHint);
              const sug = m?.[1];
              if (!sug) return null;
              return (
                <div style={{ marginTop: 10 }}>
                  <button
                    type="button"
                    className="btn"
                    onClick={() => load(sug, period)}
                  >
                    Load {sug}
                  </button>
                </div>
              );
            })()}
          </div>
        )}
        {optOverlayOn && (
          <div style={{ padding: "0 18px 10px" }}>
            <div className="dim" style={{ fontSize: 11.5, lineHeight: 1.45, marginBottom: 6 }}>
              {optOverlay?.summary
                ?? "Research guide only — not a trade order. Orange line = gamma flip."}
            </div>
            {optOverlay?.timing_note && (
              <div className="dim" style={{ fontSize: 11, lineHeight: 1.4, marginBottom: 6 }}>
                {optOverlay.timing_note}
              </div>
            )}
            {optOverlayBusy && <div className="dim" style={{ fontSize: 12 }}>Loading options structure…</div>}
            {!optOverlayBusy && optOverlay && optOverlay.success === false && (
              <div className="dim" style={{ fontSize: 12 }}>
                Options guide unavailable{optOverlay.error ? ` — ${optOverlay.error}` : ""}
              </div>
            )}
            {!optOverlayBusy && optOverlay?.success && (
              <div className="dim" style={{ fontSize: 12.5, lineHeight: 1.45 }}>
                <b>{String(optOverlay.pick?.label ?? optOverlay.markers?.[0]?.text ?? "Structure")}</b>
                {" — "}
                {String(optOverlay.pick?.rationale ?? optOverlay.markers?.[0]?.title ?? "")}
                {(optOverlay.reference_levels?.levels?.length ?? 0) > 0 && (
                  <div style={{ marginTop: 4 }}>
                    Guides:{" "}
                    {optOverlay.reference_levels!.levels!.map((l) => (
                      <span key={l.key} style={{ marginRight: 10 }}>
                        {l.label} <b className="num">{l.value}</b>
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
        {overlayOn && (
          <div style={{ padding: "0 18px 14px" }}>
            <div className="dim" style={{ fontSize: 11.5, lineHeight: 1.45, marginBottom: 8 }}>
              {overlay?.disclosure
                ?? "Backtest signals — research only, not trade orders."}
            </div>
            {overlayBusy && <div className="dim" style={{ fontSize: 12 }}>Loading strategy markers…</div>}
            {!overlayBusy && overlay && overlay.success === false && (
              <div className="dim" style={{ fontSize: 12 }}>
                Overlay unavailable{overlay.error ? ` — ${overlay.error}` : ""}
              </div>
            )}
            {!overlayBusy && overlay?.success && (
              <>
                {overlay.gamma_context && (
                  <div className="dim" style={{ fontSize: 12, marginBottom: 6 }}>
                    {overlay.gamma_context.available
                      ? <>Today’s GEX snapshot: <b>{String(overlay.gamma_context.regime_short ?? "—").replace(/_/g, " ")}</b>
                          {" — "}{String(overlay.gamma_context.historical_note ?? "display only")}</>
                      : <>GEX context unavailable{overlay.gamma_context.reason ? ` — ${String(overlay.gamma_context.reason)}` : ""}</>}
                  </div>
                )}
                {(overlay.reference_levels?.levels?.length ?? 0) > 0 && (
                  <div className="dim" style={{ fontSize: 12 }}>
                    Last-bar levels:{" "}
                    {overlay.reference_levels!.levels!.map((l) => (
                      <span key={l.key} style={{ marginRight: 10 }}>
                        {l.label} <b className="num">{l.value}</b>
                      </span>
                    ))}
                    <div style={{ marginTop: 4 }}>{overlay.reference_levels?.note}</div>
                    {(overlay.overlay_series?.length ?? 0) > 0 && (
                      <div style={{ marginTop: 4 }}>
                        Chart lines: {overlay.overlay_series!.map((s) => s.label || s.id).join(" · ")}
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>

      <div className="form-grid" style={{ marginTop: 16 }}>
        <div className="card card-pad" data-tour="dashboard-headlines">
          <div className="rail-label" style={{ marginTop: 0 }}>Headlines · {symbol}</div>
          <div className="dim" style={{ fontSize: 11.5, marginBottom: 6 }}>
            LLM blurbs (when keyed) are context only — not a call.
          </div>
          {news.length === 0 && <div className="dim">No headlines yet.</div>}
          {news.slice(0, 5).map((n, i) => {
            const title = String(n.title ?? n.headline ?? "Untitled");
            const why = newsWhy[title];
            return (
            <div key={i} style={{ padding: "8px 0", borderTop: i ? "1px solid var(--border)" : "none" }}>
              <div style={{ fontWeight: 600, fontSize: 13.5 }}>
                {n.url ? (
                  <a href={String(n.url)} target="_blank" rel="noreferrer" style={{ color: "var(--text)", textDecoration: "none" }}>
                    {title}
                  </a>
                ) : title}
              </div>
              {why && (
                <div style={{ fontSize: 12, marginTop: 3, color: "var(--text-2)", lineHeight: 1.4 }}>
                  <span className="dim">Context: </span>{why}
                </div>
              )}
              <div className="dim" style={{ fontSize: 11.5, marginTop: 3 }}>
                {String(n.source ?? "")}
                {n.source_type ? ` · ${String(n.source_type)}` : ""}
                {n.url ? <> · <a href={String(n.url)} target="_blank" rel="noreferrer" style={{ color: "var(--accent)" }}>open</a></> : null}
              </div>
            </div>
            );
          })}
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
        <div className="card card-pad" data-tour="dashboard-briefing">
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
      <div
        className="wl"
        data-tour="dashboard-watchlist"
        style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 10 }}
      >
        {watchlist.map((w) => (
          <div key={w.symbol}
            className={`wl-card fade-in ${w.symbol === symbol ? "active" : ""}`}
            onClick={() => load(w.symbol, period)}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8 }}>
              <span className="wl-sym">{w.symbol}</span>
              <span className="num" style={{
                fontSize: 12.5, fontWeight: 650,
                color: w.score == null ? "var(--text-3)"
                  : w.score >= 7 ? "var(--up)"
                    : w.score <= 4 ? "var(--down)" : "var(--text-2)",
              }}>
                {w.scoreLoading ? "…" : w.score != null ? w.score.toFixed(1) : "—"}
                {!w.scoreLoading && w.grade ? ` ${w.grade}` : ""}
              </span>
            </div>
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
