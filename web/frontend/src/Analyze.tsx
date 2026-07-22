import { useEffect, useState } from "react";
import {
  getDiagnostics, getChartEvents, getEarnings, getEdgar, getForecast, getHistory,
  getNews, getNewsContext, getOptionsContext, getPatterns, getPlaybook, getPrefs, getRisk, getScore,
  getSignalIc, runGnn, runMonteCarlo, trackRec,
  type Candle, type ChartEvent, type EdgarFiling, type ScoreResult,
} from "./api";
import { runAnalyzeLoad } from "./analyzeLoad";
import Chart, { type ChartMarker } from "./Chart";
import Sparkline from "./Sparkline";
import { cacheChartTimezone, loadCachedChartTimezone } from "./chartTime";
import PageTour from "./PageTour";

function ScoreRing({ score }: { score: number }) {
  const pct = Math.max(0, Math.min(1, score / 10));
  const R = 52, C = 2 * Math.PI * R;
  return (
    <svg width="130" height="130" viewBox="0 0 130 130">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#00d4ff" />
          <stop offset="55%" stopColor="#7b6cff" />
          <stop offset="100%" stopColor="#ff7ad9" />
        </linearGradient>
      </defs>
      <circle cx="65" cy="65" r={R} fill="none" stroke="var(--surface-3)" strokeWidth="10" />
      <circle cx="65" cy="65" r={R} fill="none" stroke="url(#g)" strokeWidth="10"
        strokeLinecap="round" strokeDasharray={`${C * pct} ${C}`}
        transform="rotate(-90 65 65)"
        style={{ transition: "stroke-dasharray 0.8s cubic-bezier(0.2,0.8,0.2,1)" }} />
      <text x="65" y="60" textAnchor="middle" fill="var(--text)"
        fontSize="30" fontWeight="800" className="num">{score.toFixed(1)}</text>
      <text x="65" y="82" textAnchor="middle" fill="var(--text-3)" fontSize="11">/ 10</text>
    </svg>
  );
}

function stripEmoji(s: string): string {
  return s.replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu, "").trim();
}

function signalLabel(k: string): string {
  return k
    .replace(/^Pattern:\s*/i, "")
    .replace(/^Factor:\s*/i, "")
    .replace(/_/g, " ")
    .trim();
}

function friendlyStrategy(name: string): string {
  return name.replace(/Strategy$/i, "").replace(/([a-z])([A-Z])/g, "$1 $2");
}

function formatSignalValue(name: string, value: unknown): string {
  if (value == null || (typeof value === "number" && !Number.isFinite(value))) return "—";
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return String(value);
  const lower = name.toLowerCase();
  if (lower.includes("bollinger")) return `${n.toFixed(0)}% of band`;
  if (lower === "rsi" || lower.startsWith("rsi ")) return n.toFixed(1);
  if (lower.includes("momentum") || lower.includes("vs sma") || lower.includes("price vs")) {
    return `${n >= 0 ? "+" : ""}${n.toFixed(1)}%`;
  }
  if (lower.startsWith("pattern")) return `${(n <= 1 ? n * 100 : n).toFixed(0)}% conf`;
  if (Math.abs(n) <= 1 && lower.includes("factor")) return n.toFixed(2);
  return n.toFixed(1);
}

function impactRank(impact?: string): number {
  if (impact === "positive") return 2;
  if (impact === "negative") return 1;
  return 0;
}


type Mode = "long" | "short";
type ToolTab = "main" | "monte" | "options" | "labs" | "filings";
type LabTab = "ic" | "diagnostics" | "patterns" | "gnn";

export default function Analyze({
  initialSymbol = "SPY",
  onOpenBacktest,
}: {
  initialSymbol?: string;
  onOpenBacktest?: (symbol: string, strategy: string) => void;
}) {
  const [input, setInput] = useState(initialSymbol);
  const [mode, setMode] = useState<Mode>("long");
  const [tool, setTool] = useState<ToolTab>("main");
  const [lab, setLab] = useState<LabTab>("ic");
  const [res, setRes] = useState<ScoreResult | null>(null);
  const [runSym, setRunSym] = useState<string | null>(null);
  const [news, setNews] = useState<Record<string, unknown>[]>([]);
  const [newsWhy, setNewsWhy] = useState<Record<string, string>>({});
  const [forecast, setForecast] = useState<Record<string, unknown> | null>(null);
  const [showModels, setShowModels] = useState(false);
  const [risk, setRisk] = useState<Record<string, string | number> | null>(null);
  const [mc, setMc] = useState<Record<string, unknown> | null>(null);
  const [opts, setOpts] = useState<Record<string, unknown> | null>(null);
  const [ic, setIc] = useState<Record<string, unknown> | null>(null);
  const [diagnostics, setDiagnostics] = useState<Record<string, unknown> | null>(null);
  const [patterns, setPatterns] = useState<Record<string, unknown> | null>(null);
  const [gnn, setGnn] = useState<Record<string, unknown> | null>(null);
  const [playbook, setPlaybook] = useState<Record<string, unknown> | null>(null);
  const [earnings, setEarnings] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [scoreLoading, setScoreLoading] = useState(false);
  const [extrasLoading, setExtrasLoading] = useState(false);
  const [toolBusy, setToolBusy] = useState(false);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [events, setEvents] = useState<ChartEvent[]>([]);
  const [chartLoading, setChartLoading] = useState(false);
  const [edgar, setEdgar] = useState<{ filings?: EdgarFiling[]; signal?: Record<string, unknown> | null; note?: string; success?: boolean; error?: string } | null>(null);
  const [tracked, setTracked] = useState<"idle" | "saving" | "done">("idle");
  const [chartTimezone, setChartTimezone] = useState(loadCachedChartTimezone);

  useEffect(() => {
    getPrefs().then((r) => {
      const tz = r.prefs?.chart_timezone;
      if (typeof tz === "string" && tz) {
        setChartTimezone(tz);
        cacheChartTimezone(tz);
      }
    }).catch(() => {});
  }, []);

  async function run() {
    const sym = (input || "SPY").trim().toUpperCase();
    setInput(sym);
    setLoading(true);
    setScoreLoading(true);
    setExtrasLoading(true);
    setChartLoading(true);
    setRunSym(sym);
    setNews([]);
    setNewsWhy({});
    setForecast(null);
    setShowModels(false);
    setRisk(null);
    setMc(null);
    setOpts(null);
    setIc(null);
    setDiagnostics(null);
    setPatterns(null);
    setGnn(null);
    setPlaybook(null);
    setEarnings(null);
    setTool("main");
    setCandles([]);
    setEvents([]);
    setEdgar(null);
    setTracked("idle");
    setRes(null);

    await runAnalyzeLoad({
      symbol: sym,
      mode,
      fetchers: {
        getScore,
        getHistory,
        getChartEvents,
        getNews,
        getRisk,
        getPlaybook,
        getEarnings,
        getForecast,
        getNewsContext,
      },
      hooks: {
        onCritical: ({ history }) => {
          setCandles((history.candles as Candle[]) ?? []);
          setChartLoading(false);
          setLoading(false);
        },
        onCriticalError: () => {
          setCandles([]);
          setChartLoading(false);
          setLoading(false);
        },
        onScore: (score) => {
          setRes(score as unknown as ScoreResult);
          setScoreLoading(false);
        },
        onScoreError: () => {
          setRes({
            symbol: sym,
            score: null,
            grade: null,
            signals: {},
            error: "Score unavailable",
          });
          setScoreLoading(false);
        },
        onEvents: (ev) => setEvents(ev as ChartEvent[]),
        onExtras: ({ news: items, risk: rk, playbook: pb, earnings: er }) => {
          setNews(items as Record<string, unknown>[]);
          setRisk((rk as Record<string, string | number> | null) ?? null);
          setPlaybook(pb);
          setEarnings(er);
        },
        onNewsWhy: (map) => setNewsWhy(map),
        onForecast: (f) => setForecast(f),
        onForecastSettled: () => setExtrasLoading(false),
      },
    });
  }

  async function loadMonte() {
    setToolBusy(true);
    try { setMc(await runMonteCarlo(input)); }
    catch (e) { setMc({ success: false, error: String(e) }); }
    finally { setToolBusy(false); }
  }

  async function loadOptions() {
    setToolBusy(true);
    try { setOpts(await getOptionsContext(input) as unknown as Record<string, unknown>); }
    catch (e) { setOpts({ success: false, error: String(e) }); }
    finally { setToolBusy(false); }
  }

  async function loadIc() {
    setToolBusy(true);
    try { setIc(await getSignalIc(input)); }
    catch (e) { setIc({ success: false, error: String(e) }); }
    finally { setToolBusy(false); }
  }

  async function loadDiagnostics() {
    setToolBusy(true);
    try { setDiagnostics(await getDiagnostics(input)); }
    catch (e) { setDiagnostics({ success: false, error: String(e) }); }
    finally { setToolBusy(false); }
  }

  async function loadFilings() {
    setToolBusy(true);
    try { setEdgar(await getEdgar(input)); }
    catch (e) { setEdgar({ success: false, error: String(e) }); }
    finally { setToolBusy(false); }
  }

  async function trackIdea() {
    if (!res || tracked === "saving") return;
    setTracked("saving");
    try {
      const r = await trackRec(res.symbol, res.score != null ? Number(res.score) : null,
                               res.last_price != null ? Number(res.last_price) : null,
                               "", "analyze");
      setTracked(r.success ? "done" : "idle");
    } catch { setTracked("idle"); }
  }

  async function loadPatterns() {
    setToolBusy(true);
    try { setPatterns(await getPatterns(input)); }
    catch (e) { setPatterns({ success: false, error: String(e) }); }
    finally { setToolBusy(false); }
  }

  async function loadGnn() {
    setToolBusy(true);
    try {
      setGnn(await runGnn([input, "SPY", "QQQ", "AAPL", "MSFT"].filter((v, i, a) => a.indexOf(v) === i)));
    } catch (e) { setGnn({ success: false, error: String(e) }); }
    finally { setToolBusy(false); }
  }

  function openLab(next: LabTab) {
    setLab(next);
    setTool("labs");
    if (next === "ic" && !ic) void loadIc();
    if (next === "diagnostics" && !diagnostics) void loadDiagnostics();
    if (next === "patterns" && !patterns) void loadPatterns();
    if (next === "gnn" && !gnn) void loadGnn();
  }

  // Prefer structured signal_list (raw readings + impact). Values are NOT a shared 0–100 scale.
  const topSignals = (() => {
    const list = Array.isArray(res?.signal_list) ? res!.signal_list! : [];
    if (list.length > 0) {
      return [...list]
        .filter((s) => s?.name)
        .sort((a, b) => impactRank(a.impact) - impactRank(b.impact) || 0)
        .reverse()
        .slice(0, 5)
        .map((s) => ({
          name: String(s.name),
          display: formatSignalValue(String(s.name), s.value),
          impact: s.impact,
          hint: s.description ? String(s.description) : undefined,
        }));
    }
    // Legacy dict fallback
    return Object.entries(res?.signals ?? {})
      .slice(0, 5)
      .map(([k, v]) => ({
        name: k,
        display: formatSignalValue(k, typeof v === "number" ? v : (v as { value?: unknown })?.value),
        impact: undefined as string | undefined,
        hint: undefined as string | undefined,
      }));
  })();

  const dims = [
    ["Technical", res?.technical_score],
    ["Momentum", res?.momentum_score],
    ["Sentiment", res?.sentiment_score],
    ["Fundamental", res?.fundamental_score],
  ] as const;

  const fcPath = Array.isArray(forecast?.consensus_forecast)
    ? (forecast!.consensus_forecast as number[])
    : Array.isArray(forecast?.consensus_path)
      ? (forecast!.consensus_path as number[])
      : Array.isArray(forecast?.values) ? (forecast!.values as number[]) : [];

  const riskEntries = risk ? Object.entries(risk).slice(0, 8) : [];
  const agreement = forecast?.model_agreement != null ? Number(forecast.model_agreement) : null;
  const conviction = forecast?.conviction != null ? String(forecast.conviction) : null;
  const lastPx = forecast?.last_price != null ? Number(forecast.last_price) : null;
  const priceTargets = (forecast?.price_targets && typeof forecast.price_targets === "object")
    ? forecast.price_targets as Record<string, unknown>
    : {};
  const modelsUsed = Array.isArray(forecast?.models_used) ? (forecast!.models_used as string[]) : [];
  const modelsFailed = Array.isArray(forecast?.models_failed) ? (forecast!.models_failed as string[]) : [];
  const modelsExcluded = Array.isArray(forecast?.models_excluded) ? (forecast!.models_excluded as string[]) : [];
  const modelRows = modelsUsed.map((name) => {
    const target = priceTargets[name];
    const t = typeof target === "number" ? target : Number(target);
    let dir = "flat";
    if (lastPx != null && Number.isFinite(t) && lastPx > 0) {
      if (t > lastPx * 1.005) dir = "up";
      else if (t < lastPx * 0.995) dir = "down";
    }
    const chg = (lastPx != null && Number.isFinite(t) && lastPx > 0)
      ? ((t / lastPx) - 1) * 100 : null;
    return { name, target: Number.isFinite(t) ? t : null, dir, chg };
  });
  const mcPath = Array.isArray(mc?.mean_path) ? (mc!.mean_path as number[]) : [];

  const strat = (playbook?.strategy && typeof playbook.strategy === "object")
    ? playbook.strategy as Record<string, unknown>
    : null;
  const primaryStrategy = String(strat?.primary_strategy ?? "");
  const alts = Array.isArray(strat?.recommended_strategies)
    ? (strat!.recommended_strategies as string[]).filter((s) => s !== primaryStrategy).slice(0, 2)
    : [];
  const diagFlags = Array.isArray(diagnostics?.flags) ? (diagnostics!.flags as string[]) : [];
  const diagRecs = Array.isArray(diagnostics?.recommendations)
    ? (diagnostics!.recommendations as string[]) : [];
  const nextEarn = (earnings?.next_earnings && typeof earnings.next_earnings === "object")
    ? earnings.next_earnings as Record<string, unknown>
    : null;

  const regime = String(strat?.market_regime ?? "");
  const regimeHint =
    regime === "bull" ? "trending up"
      : regime === "bear" ? "trending down"
        : regime === "volatile" ? "choppy / big price swings"
          : regime === "sideways" ? "range-bound"
            : regime || "unclear";

  const displaySym = (res?.symbol || runSym || input || "SPY").toUpperCase();
  const showResults = !loading && !!runSym;

  return (
    <div className="fade-in">
      <PageTour pageId="analyze" />
      <div className="greeting">
        Analyze <small>Overall rating, price outlook, news & bumpiness</small>
      </div>
      <div className="topbar" data-tour="analyze-controls">
        <div className="search">
          <span className="icon">⌕</span>
          <input value={input}
            onChange={(e) => setInput(e.target.value.toUpperCase())}
            onKeyDown={(e) => e.key === "Enter" && void run()} />
        </div>
        <div className="seg">
          <button className={mode === "long" ? "active" : ""} onClick={() => setMode("long")}>Buy / long</button>
          <button className={mode === "short" ? "active" : ""} onClick={() => setMode("short")}>Bet it falls</button>
        </div>
        <button className="primary" onClick={() => void run()}>
          {loading ? "Loading chart…" : scoreLoading ? "Scoring…" : "Analyze"}
        </button>
      </div>

      {!loading && !runSym && (
        <>
          <div className="card card-pad empty" data-tour="analyze-score">
            Enter a ticker (like AAPL) for rating, outlook, headlines, and risk.
          </div>
          <div className="card card-pad empty" data-tour="analyze-chart" style={{ marginTop: 12 }}>
            Chart and news marks appear here after Analyze.
          </div>
          <div className="seg" data-tour="analyze-tools" style={{ marginTop: 12, flexWrap: "wrap", opacity: 0.75 }}>
            <button type="button" disabled>Overview</button>
            <button type="button" disabled>What-if ranges</button>
            <button type="button" disabled>Options</button>
            <button type="button" disabled>Filings</button>
            <button type="button" disabled>Labs</button>
          </div>
          <div className="form-grid" data-tour="analyze-forecast-risk" style={{ marginTop: 12 }}>
            <div className="card card-pad empty">Forecast and risk cards appear here after Analyze.</div>
            <div className="card card-pad empty">Risk metrics appear here after Analyze.</div>
          </div>
          <div className="card card-pad empty" data-tour="analyze-news" style={{ marginTop: 12 }}>
            Symbol headlines and context appear here after Analyze.
          </div>
        </>
      )}

      {loading && <div className="skeleton" style={{ height: 220, marginBottom: 16 }} />}

      {showResults && (
        <>
          <div className="card card-pad" data-tour="analyze-score" style={{ marginBottom: 16 }}>
            {scoreLoading ? (
              <div style={{ display: "flex", gap: 28, alignItems: "center", flexWrap: "wrap" }}>
                <div className="skeleton" style={{ width: 130, height: 130, borderRadius: "50%" }} />
                <div style={{ flex: 1, minWidth: 200 }}>
                  <div style={{ fontSize: 22, fontWeight: 700 }}>{displaySym}</div>
                  <div className="dim" style={{ marginTop: 8 }}>
                    Computing AI score… chart and headlines load in parallel.
                  </div>
                </div>
              </div>
            ) : res && res.score != null ? (
            <div style={{ display: "flex", gap: 28, alignItems: "center", flexWrap: "wrap" }}>
              <ScoreRing score={Number(res.score)} />
              <div style={{ flex: 1, minWidth: 200 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <div style={{ fontSize: 22, fontWeight: 700 }}>{displaySym}</div>
                  <button className="ghost" style={{ fontSize: 12, border: "1px solid var(--border)" }}
                    onClick={trackIdea} disabled={tracked !== "idle"}
                    title="Save this idea (no purchase) — record a practice buy in Portfolio when you try it">
                    {tracked === "done" ? "✓ Tracking" : tracked === "saving" ? "Saving…" : "☆ Track idea"}
                  </button>
                </div>
                <div className="dim" style={{ marginTop: 4 }}>
                  Grade {res.grade ?? "—"}
                  {res.last_price != null && <> · ${Number(res.last_price).toFixed(2)}</>}
                  {mode === "short" && res.short_score != null && (
                    <> · short {Number(res.short_score).toFixed(1)}</>
                  )}
                  {tracked === "done" && (
                    <> · saved idea — record a practice buy in Portfolio when you try it</>
                  )}
                </div>
                {res.summary && (
                  <div style={{ marginTop: 10, fontSize: 13.5, lineHeight: 1.45, maxWidth: 520 }}>
                    {String(res.summary)}
                  </div>
                )}
              </div>
              <div className="kpis" style={{ flex: 1, minWidth: 240 }}>
                {dims.map(([label, v]) => (
                  <div className="card kpi" key={label} style={{ padding: "10px 12px" }}>
                    <div className="label">{label}</div>
                    <div className="value num" style={{ fontSize: 18 }}>
                      {v != null ? Number(v).toFixed(1) : "—"}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            ) : (
              <div className="dim">
                Couldn&apos;t score {displaySym}
                {res?.error ? ` — ${res.error}` : ""}. Live data required.
              </div>
            )}

            {topSignals.length > 0 && (
              <div style={{ marginTop: 18, borderTop: "1px solid var(--border)", paddingTop: 14 }}>
                <div className="rail-label" style={{ marginTop: 0 }}>Key readings</div>
                <div className="dim" style={{ fontSize: 11.5, marginBottom: 8 }}>
                  What the score is reacting to — plain English under each name.
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "10px 28px", maxWidth: 520 }}>
                  {topSignals.map((s) => (
                    <div key={s.name} style={{ display: "contents" }}>
                      <div>
                        <div style={{ fontSize: 13.5, color: "var(--text-2)" }}>
                          {signalLabel(s.name)}
                        </div>
                        {s.hint && (
                          <div className="dim" style={{ fontSize: 11.5, lineHeight: 1.35, marginTop: 2 }}>
                            {stripEmoji(s.hint)}
                          </div>
                        )}
                      </div>
                      <div className="num" style={{
                        fontSize: 13.5, fontWeight: 650, textAlign: "right",
                        color: s.impact === "positive" ? "var(--up)"
                          : s.impact === "negative" ? "var(--down)" : "var(--text)",
                        alignSelf: "start",
                      }}>
                        {s.display}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="card" data-tour="analyze-chart" style={{ marginBottom: 16 }}>
            <div className="chart-head">
              <div className="legend"><b>{displaySym}</b> · 6 months, daily bars</div>
            </div>
            {chartLoading ? (
              <div className="skeleton" style={{ height: 320, margin: 18 }} />
            ) : candles.length > 0 ? (
              <Chart timeZone={chartTimezone} candles={candles} markers={[
                ...events.map((e): ChartMarker => {
                  const fallback = e.link_quality === "fallback_recent";
                  const honesty = fallback ? " · may not be same-day headline" : "";
                  return {
                    time: e.time,
                    title: e.title ? `${e.title}${honesty}` : e.title,
                    text: e.text,
                    color: e.color,
                  };
                }),
                // Pattern markers: PLOTTED, not just described - the
                // detector already computes start_date/type/confidence
                // per pattern (agent_tools.get_pattern_analysis used to
                // discard this down to a text summary). Appears once the
                // Patterns tool below has been opened.
                ...(Array.isArray(patterns?.patterns)
                  ? (patterns!.patterns as Record<string, unknown>[])
                      .filter((p) => p.start_date)
                      .map((p): ChartMarker => ({
                        time: String(p.start_date),
                        title: String(p.name ?? "Pattern"),
                        text: p.description ? String(p.description).slice(0, 120) : undefined,
                        color: p.type === "bullish" ? "var(--up)"
                          : p.type === "bearish" ? "var(--down)" : "var(--violet)",
                      }))
                  : []),
              ]} />
            ) : (
              <div className="empty" style={{ padding: "40px 0" }}>
                No chart data for {displaySym}.
              </div>
            )}
          </div>

          <div className="seg" data-tour="analyze-tools" style={{ marginBottom: 12, flexWrap: "wrap" }}>
            <button className={tool === "main" ? "active" : ""} onClick={() => setTool("main")}>Overview</button>
            <button className={tool === "monte" ? "active" : ""} onClick={() => { setTool("monte"); if (!mc) void loadMonte(); }}>What-if ranges</button>
            <button className={tool === "options" ? "active" : ""} onClick={() => { setTool("options"); if (!opts) void loadOptions(); }}>Options</button>
            <button className={tool === "filings" ? "active" : ""} onClick={() => { setTool("filings"); if (!edgar) void loadFilings(); }}>Filings</button>
            <button className={tool === "labs" ? "active" : ""} onClick={() => openLab(lab)}>Labs</button>
          </div>

          {tool === "main" && (
            <>
              {strat && primaryStrategy && (
                <div className="card card-pad" style={{ marginBottom: 16 }}>
                  <div style={{ display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap", justifyContent: "space-between" }}>
                    <div style={{ flex: 1, minWidth: 260 }}>
                      <div className="rail-label" style={{ marginTop: 0 }}>Suggested trading style</div>
                      <div style={{ fontSize: 16, fontWeight: 650, marginBottom: 6 }}>
                        {friendlyStrategy(primaryStrategy)}
                      </div>
                      <div style={{ fontSize: 13.5, lineHeight: 1.5, color: "var(--text-2)", maxWidth: 520 }}>
                        Recent price action looks <b style={{ color: "var(--text)" }}>{regimeHint}</b>
                        {strat.reason ? ` — ${String(strat.reason)}` : ""}.
                        {" "}We suggest trying this playbook style next (not a guaranteed winner).
                        {alts.length > 0 && (
                          <> Also worth trying: {alts.map(friendlyStrategy).join(", ")}.</>
                        )}
                      </div>
                    </div>
                    {onOpenBacktest && (
                      <button className="primary" style={{ marginTop: 4 }}
                        onClick={() => onOpenBacktest(input, primaryStrategy)}>
                        Backtest {friendlyStrategy(primaryStrategy)}
                      </button>
                    )}
                  </div>
                </div>
              )}

              {earnings && earnings.success !== false && (
                (earnings.avg_move_1d != null || nextEarn) && (
                  <div className="card card-pad" style={{ marginBottom: 16 }}>
                    <div className="rail-label" style={{ marginTop: 0 }}>Earnings</div>
                    <div className="kpis">
                      {nextEarn && (nextEarn.next_earnings_date != null || nextEarn.date != null) ? (
                        <div className="card kpi" style={{ padding: "10px 12px" }}>
                          <div className="label">Next</div>
                          <div className="value num" style={{ fontSize: 15 }}>
                            {String(nextEarn.next_earnings_date ?? nextEarn.date)}
                          </div>
                          {nextEarn.days_until != null && (
                            <div className="sub">{Number(nextEarn.days_until)}d away</div>
                          )}
                        </div>
                      ) : null}
                      {earnings.avg_move_1d != null && (
                        <div className="card kpi" style={{ padding: "10px 12px" }}>
                          <div className="label">Avg one-day move</div>
                          <div className="value num" style={{ fontSize: 16 }}>
                            ±{Number(earnings.avg_move_1d).toFixed(1)}%
                          </div>
                        </div>
                      )}
                      {earnings.beat_rate != null && (
                        <div className="card kpi" style={{ padding: "10px 12px" }}>
                          <div className="label">Profit beat rate</div>
                          <div className="value num" style={{ fontSize: 16 }}>
                            {Number(earnings.beat_rate).toFixed(0)}%
                          </div>
                        </div>
                      )}
                      {earnings.positive_reaction_rate != null && (
                        <div className="card kpi" style={{ padding: "10px 12px" }}>
                          <div className="label">Stock usually rose</div>
                          <div className="value num" style={{ fontSize: 16 }}>
                            {Number(earnings.positive_reaction_rate).toFixed(0)}%
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )
              )}

              <div className="form-grid" data-tour="analyze-forecast-risk" style={{ marginBottom: 16 }}>
                <div className="card card-pad">
                  <div className="rail-label" style={{ marginTop: 0 }}>Forecast</div>
                  {extrasLoading && !forecast && (
                    <div className="dim">Loading combined price outlook…</div>
                  )}
                  {!extrasLoading && fcPath.length === 0 && (
                    <div className="dim">Forecast unavailable for this symbol right now.</div>
                  )}
                  {fcPath.length > 0 && (
                    <>
                      <div style={{ fontSize: 15, fontWeight: 650, marginBottom: 8 }}>
                        {String(forecast?.direction ?? "—")}
                        {forecast?.consensus_7d_change_pct != null && (
                          <span className="num" style={{ marginLeft: 10, color: Number(forecast.consensus_7d_change_pct) >= 0 ? "var(--up)" : "var(--down)" }}>
                            {Number(forecast.consensus_7d_change_pct) >= 0 ? "+" : ""}
                            {Number(forecast.consensus_7d_change_pct).toFixed(2)}% / 7d
                          </span>
                        )}
                      </div>
                      <Sparkline values={fcPath} width={420} height={64} baseline="start" />
                      <div className="dim" style={{ marginTop: 8, fontSize: 12.5 }}>
                        Consensus to{" "}
                        {forecast?.consensus_price != null
                          ? `$${Number(forecast.consensus_price).toFixed(2)}`
                          : "—"}
                        {conviction && <> · {conviction.toLowerCase()} confidence</>}
                        {agreement != null && Number.isFinite(agreement) && (
                          <> · agreement {(agreement * 100).toFixed(0)}%</>
                        )}
                      </div>
                      {(() => {
                        const routing = (forecast?.routing && typeof forecast.routing === "object")
                          ? forecast.routing as Record<string, unknown>
                          : null;
                        if (!routing) return null;
                        const just = routing.justification != null ? String(routing.justification) : "";
                        const applied = Boolean(routing.feature_routing_applied);
                        const rule = routing.active_rule != null ? String(routing.active_rule) : "";
                        return (
                          <div className="dim" style={{ marginTop: 8, fontSize: 11.5, lineHeight: 1.45, maxWidth: 480 }}>
                            {applied && rule
                              ? <>Using a tested rule for this stock: <span style={{ color: "var(--text)" }}>{rule}</span>. </>
                              : <>Using the default model mix. </>}
                            {just}
                          </div>
                        );
                      })()}
                      {modelRows.length > 0 && (
                        <div style={{ marginTop: 10 }}>
                          <button className="ghost" style={{ fontSize: 12, border: "1px solid var(--border)", padding: "4px 10px" }}
                            onClick={() => setShowModels((v) => !v)}>
                            {showModels ? "Hide models" : "Show models"}
                          </button>
                          {showModels && (
                            <div style={{ marginTop: 10 }}>
                              <div className="dim" style={{ fontSize: 11.5, marginBottom: 8, lineHeight: 1.4 }}>
                                What each model thinks the price could be — disagreement is useful info, not a vote to trade.
                              </div>
                              <table className="tbl">
                                <thead>
                                  <tr><th>Model</th><th>Target</th><th>vs current</th><th>Direction</th></tr>
                                </thead>
                                <tbody>
                                  {modelRows.map((m) => (
                                    <tr key={m.name}>
                                      <td style={{ fontWeight: 600, textTransform: "capitalize" }}>{m.name}</td>
                                      <td className="num">{m.target != null ? `$${m.target.toFixed(2)}` : "—"}</td>
                                      <td className="num" style={{
                                        color: m.chg == null ? undefined
                                          : m.chg > 0 ? "var(--up)" : m.chg < 0 ? "var(--down)" : undefined,
                                      }}>
                                        {m.chg != null ? `${m.chg >= 0 ? "+" : ""}${m.chg.toFixed(2)}%` : "—"}
                                      </td>
                                      <td style={{
                                        color: m.dir === "up" ? "var(--up)"
                                          : m.dir === "down" ? "var(--down)" : "var(--text-2)",
                                      }}>
                                        {m.dir === "up" ? "up" : m.dir === "down" ? "down" : "flat"}
                                      </td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                              {(modelsFailed.length > 0 || modelsExcluded.length > 0) && (
                                <div className="dim" style={{ fontSize: 11.5, marginTop: 8 }}>
                                  {modelsFailed.length > 0 && <>Failed: {modelsFailed.join(", ")}. </>}
                                  {modelsExcluded.length > 0 && <>Excluded as outliers: {modelsExcluded.join(", ")}.</>}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </>
                  )}
                </div>
                <div className="card card-pad">
                  <div className="rail-label" style={{ marginTop: 0 }}>Risk · {input}</div>
                  <div className="dim" style={{ fontSize: 11.5, marginBottom: 8 }}>
                    How rough the ride can get — higher risk means bigger swings, not a prediction of direction.
                  </div>
                  {riskEntries.length > 0 ? (
                    <div className="kpis">
                      {riskEntries.map(([k, v]) => (
                        <div className="card kpi" key={k} style={{ padding: "10px 12px" }}>
                          <div className="label">{k}</div>
                          <div className="value num" style={{ fontSize: 16 }}>{String(v)}</div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="dim">{extrasLoading ? "Loading risk…" : "Risk metrics unavailable."}</div>
                  )}
                </div>
              </div>

              <div className="card card-pad" data-tour="analyze-news">
                <div className="rail-label" style={{ marginTop: 0 }}>News · {input}</div>
                <div className="dim" style={{ fontSize: 11.5, marginBottom: 8 }}>
                  Headlines for context only — not a buy/sell call.
                </div>
                {news.length === 0 && <div className="dim">No headlines — add a News API key in Settings for more coverage.</div>}
                {news.slice(0, 6).map((n, i) => {
                  const title = String(n.title ?? n.headline ?? "Untitled");
                  const why = newsWhy[title];
                  return (
                    <div key={i} style={{ padding: "10px 0", borderTop: i ? "1px solid var(--border)" : "none" }}>
                      <div style={{ fontWeight: 600 }}>{title}</div>
                      {why && (
                        <div style={{ fontSize: 12.5, marginTop: 4, color: "var(--text-2)", lineHeight: 1.4 }}>
                          <span className="dim">Context: </span>{why}
                        </div>
                      )}
                      <div className="dim" style={{ fontSize: 12, marginTop: 4 }}>
                        {String(n.source ?? n.publisher ?? "")}
                        {n.url ? <> · <a href={String(n.url)} target="_blank" rel="noreferrer" style={{ color: "var(--accent)" }}>open</a></> : null}
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          )}

          {tool === "monte" && (
            <div className="card card-pad">
              <div className="rail-label" style={{ marginTop: 0 }}>What-if price paths · {input}</div>
              <div className="dim" style={{ fontSize: 11.5, marginBottom: 10 }}>
                Many random “what if” price paths. Median = typical outcome; 5th/95th = rough bad/good range.
              </div>
              {toolBusy && <div className="dim">Simulating…</div>}
              {!toolBusy && mc?.success === false && (
                <div className="dim">{String(mc.error ?? "Unavailable")}</div>
              )}
              {!toolBusy && mc?.success !== false && mc?.final_p50 != null && (
                <>
                  <div className="kpis" style={{ marginBottom: 12 }}>
                    <div className="card kpi"><div className="label">Bad-case outcome</div>
                      <div className="value num" style={{ fontSize: 18 }}>${Number(mc.final_p5).toLocaleString()}</div></div>
                    <div className="card kpi"><div className="label">Median</div>
                      <div className="value num" style={{ fontSize: 18 }}>${Number(mc.final_p50).toLocaleString()}</div></div>
                    <div className="card kpi"><div className="label">Good-case outcome</div>
                      <div className="value num" style={{ fontSize: 18 }}>${Number(mc.final_p95).toLocaleString()}</div></div>
                  </div>
                  {mcPath.length > 1 && <Sparkline values={mcPath.map(Number)} width={520} height={56} baseline="start" />}
                </>
              )}
            </div>
          )}

          {tool === "options" && (
            <div className="card card-pad">
              <div className="rail-label" style={{ marginTop: 0 }}>Options context · {input}</div>
              <div className="dim" style={{ fontSize: 11.5, marginBottom: 10 }}>
                {String(
                  (opts?.summary as string | undefined)
                  ?? "Delayed options data — context only, not a trade signal.",
                )}
              </div>
              {toolBusy && <div className="skeleton" style={{ height: 140 }} />}
              {!toolBusy && opts && opts.success === false && (
                <div className="dim">{String(opts.error ?? "Unavailable")}</div>
              )}
              {!toolBusy && opts && opts.success !== false && (
                <>
                  {(() => {
                    const gex = (opts.gex || {}) as Record<string, unknown>;
                    const skew = (opts.skew || {}) as Record<string, unknown>;
                    const sent = (opts.sentiment || {}) as Record<string, unknown>;
                    const pins = Array.isArray(gex.pin_candidates)
                      ? (gex.pin_candidates as Array<Record<string, unknown>>).slice(0, 5)
                      : [];
                    return (
                      <>
                        <div className="kpis" style={{ marginBottom: 12 }}>
                          <div className="card kpi">
                            <div className="label">Options positioning</div>
                            <div className="value" style={{ fontSize: 15 }}>
                              {gex.success
                                ? String(gex.regime_short ?? "—").replace(/_/g, " ")
                                : "—"}
                            </div>
                            <div className="sub">
                              {gex.success
                                ? String(gex.plain_language ?? gex.regime_plain ?? gex.regime ?? "").slice(0, 160)
                                : String(gex.error ?? "Options positioning unavailable")}
                            </div>
                          </div>
                          <div className="card kpi">
                            <div className="label">Key options price level</div>
                            <div className="value num" style={{ fontSize: 18 }}>
                              {gex.gamma_flip != null
                                ? Number(gex.gamma_flip).toFixed(2)
                                : "—"}
                            </div>
                            <div className="sub">
                              spot {gex.spot != null ? Number(gex.spot).toFixed(2) : "—"}
                            </div>
                          </div>
                          <div className="card kpi">
                            <div className="label">Fear vs greed in options</div>
                            <div className="value" style={{ fontSize: 15 }}>
                              {skew.success
                                ? String(skew.shape ?? "—").replace(/_/g, " ")
                                : "—"}
                            </div>
                            <div className="sub">
                              {skew.success
                                ? String(
                                  skew.plain_language
                                    ?? (skew.event_context as Record<string, unknown> | undefined)
                                      ?.plain_language
                                    ?? skew.detail
                                    ?? "",
                                ).slice(0, 160)
                                : String(skew.error ?? "Options pricing unavailable")}
                            </div>
                          </div>
                          <div className="card kpi">
                            <div className="label">Flow</div>
                            <div className="value" style={{ fontSize: 15 }}>
                              {sent.success ? String(sent.net_flow ?? "—") : "—"}
                            </div>
                            <div className="sub">
                              {sent.success
                                ? `Put vs call activity ${Number(sent.put_call_ratio ?? 0).toFixed(2)} · crowd price ${Number(sent.max_pain ?? 0).toFixed(2)}`
                                : String(sent.error ?? "Flow unavailable")}
                            </div>
                          </div>
                        </div>
                        {pins.length > 0 && (
                          <div style={{ marginBottom: 12 }}>
                            <div className="rail-label">Price levels options traders watch</div>
                            <table className="tbl">
                              <thead>
                                <tr><th>Strike</th><th className="num">Strength</th></tr>
                              </thead>
                              <tbody>
                                {pins.map((p) => (
                                  <tr key={String(p.strike)}>
                                    <td className="num">{Number(p.strike).toFixed(2)}</td>
                                    <td className="num">{Number(p.gex).toExponential(2)}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}
                        {skew.plain_language != null && (
                          <div className="dim" style={{ fontSize: 12.5, lineHeight: 1.45 }}>
                            {String(skew.plain_language)}
                          </div>
                        )}
                        {skew.plain_language == null && skew.framing != null && (
                          <div className="dim" style={{ fontSize: 12.5, lineHeight: 1.45 }}>
                            {String(skew.framing)}
                          </div>
                        )}
                      </>
                    );
                  })()}
                </>
              )}
            </div>
          )}

          {tool === "filings" && (
            <div className="card card-pad" style={{ marginBottom: 16 }}>
              <div className="rail-label" style={{ marginTop: 0 }}>SEC filings · {displaySym}</div>
              {toolBusy && !edgar && <div className="skeleton" style={{ height: 100 }} />}
              {edgar?.success === false && (
                <div className="dim">Couldn't reach SEC EDGAR right now{edgar.error ? ` — ${edgar.error}` : ""}.</div>
              )}
              {edgar?.success && (edgar.filings?.length ?? 0) === 0 && (
                <div className="dim">No recent filings found — common for ETFs and non-US listings.</div>
              )}
              {edgar?.success && (edgar.filings?.length ?? 0) > 0 && (
                <>
                  <table className="tbl">
                    <thead><tr><th>What it is</th><th>Filed</th><th /></tr></thead>
                    <tbody>
                      {edgar.filings!.map((f) => (
                        <tr key={f.form}>
                          <td>
                            <b>{f.form}</b>
                            <div className="dim" style={{ fontSize: 12 }}>{f.label}</div>
                          </td>
                          <td className="num">{f.date ?? "—"}</td>
                          <td>
                            {f.url && (
                              <a href={f.url} target="_blank" rel="noreferrer" className="ghost"
                                style={{ fontSize: 12.5 }}>Read on sec.gov ↗</a>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {edgar.signal != null && (edgar.signal as Record<string, unknown>).summary != null && (
                    <div className="dim" style={{ fontSize: 12.5, marginTop: 10 }}>
                      Filing tone: {String((edgar.signal as Record<string, unknown>).summary)}
                    </div>
                  )}
                  {edgar.note && <div className="dim" style={{ fontSize: 11.5, marginTop: 8 }}>{edgar.note}</div>}
                </>
              )}
            </div>
          )}
          {tool === "labs" && (
            <div className="card card-pad">
              <div className="seg" style={{ marginBottom: 12, flexWrap: "wrap" }}>
                <button className={lab === "ic" ? "active" : ""} onClick={() => openLab("ic")}>Score accuracy</button>
                <button className={lab === "diagnostics" ? "active" : ""} onClick={() => openLab("diagnostics")}>Diagnostics</button>
                <button className={lab === "patterns" ? "active" : ""} onClick={() => openLab("patterns")}>Patterns</button>
                <button className={lab === "gnn" ? "active" : ""} onClick={() => openLab("gnn")}>Connected stocks</button>
              </div>
              {toolBusy && <div className="dim">Computing…</div>}

              {!toolBusy && lab === "ic" && ic && (
                ic.success === false
                  ? <div className="dim">{String(ic.error ?? "IC unavailable")}</div>
                  : (
                    <>
                      <div className="dim" style={{ fontSize: 12.5, marginBottom: 10 }}>
                        Has this score historically lined up with later returns? Above 0.05 means it has been a useful hint.
                      </div>
                      <div className="kpis">
                        {Object.entries(ic)
                          .filter(([k, v]) => k !== "success" && k !== "symbol" && (typeof v === "number" || typeof v === "string"))
                          .slice(0, 8)
                          .map(([k, v]) => (
                            <div className="card kpi" key={k}>
                              <div className="label">{k.replace(/_/g, " ")}</div>
                              <div className="value num" style={{ fontSize: 16 }}>
                                {typeof v === "number" ? v.toFixed(3) : String(v)}
                              </div>
                            </div>
                          ))}
                      </div>
                    </>
                  )
              )}

              {!toolBusy && lab === "diagnostics" && diagnostics && (
                diagnostics.success === false
                  ? <div className="dim">{String(diagnostics.error ?? "Unavailable")}</div>
                  : (
                    <>
                      <div className="dim" style={{ fontSize: 12.5, marginBottom: 12 }}>
                        Checks whether price patterns are stable over time — helps pick the right forecast style.
                        {diagnostics.complexity != null && (
                          <> Complexity: <b style={{ color: "var(--text)" }}>{String(diagnostics.complexity)}</b>.</>
                        )}
                        {diagnostics.n_observations != null && (
                          <> · {Number(diagnostics.n_observations)} observations</>
                        )}
                      </div>
                      {typeof diagnostics.disclosure === "string" && diagnostics.disclosure && (
                        <div className="dim" style={{ fontSize: 11.5, marginBottom: 10 }}>
                          {String(diagnostics.disclosure)}
                        </div>
                      )}
                      {diagFlags.length > 0 && (
                        <>
                          <div className="rail-label" style={{ marginTop: 0 }}>Findings</div>
                          <ul style={{ margin: "0 0 14px", paddingLeft: 18, fontSize: 13.5, lineHeight: 1.55 }}>
                            {diagFlags.map((f, i) => (
                              <li key={i}>{stripEmoji(String(f))}</li>
                            ))}
                          </ul>
                        </>
                      )}
                      {diagRecs.length > 0 && (
                        <>
                          <div className="rail-label">Recommendations</div>
                          <ul style={{ margin: "0 0 14px", paddingLeft: 18, fontSize: 13.5, lineHeight: 1.55 }}>
                            {diagRecs.map((f, i) => (
                              <li key={i}>{stripEmoji(String(f))}</li>
                            ))}
                          </ul>
                        </>
                      )}
                      {diagFlags.length === 0 && diagRecs.length === 0 && (
                        <div className="dim">No structured findings returned for this symbol.</div>
                      )}
                    </>
                  )
              )}

              {!toolBusy && lab === "patterns" && patterns && (
                patterns.success === false
                  ? <div className="dim">{String(patterns.error ?? "Unavailable")}</div>
                  : (
                    <>
                      <div style={{ fontWeight: 650, marginBottom: 8 }}>
                        {String(patterns.pattern_count ?? 0)} pattern(s)
                      </div>
                      <pre style={{ whiteSpace: "pre-wrap", fontSize: 12.5, color: "var(--text-2)", margin: 0 }}>
                        {String(patterns.summary ?? "")}
                      </pre>
                    </>
                  )
              )}

              {!toolBusy && lab === "gnn" && gnn && (
                gnn.success === false
                  ? <div className="dim">{String(gnn.error ?? "Unavailable")}</div>
                  : (
                    <>
                      <div className="dim" style={{ fontSize: 12.5, marginBottom: 8 }}>{String(gnn.note ?? "")}</div>
                      <div style={{ fontSize: 13.5, marginBottom: 8 }}>
                        Target {String(gnn.target)} · assets {(gnn.assets as string[] | undefined)?.join(", ")}
                      </div>
                      {Array.isArray(gnn.forecast) && (gnn.forecast as number[]).length > 0 && (
                        <Sparkline values={(gnn.forecast as number[]).map(Number)} width={420} height={56} baseline="start" />
                      )}
                    </>
                  )
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
