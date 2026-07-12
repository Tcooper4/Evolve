import { useState } from "react";
import {
  getCausal, getChartEvents, getEarnings, getEdgar, getForecast, getHistory,
  getNews, getOptions, getPatterns, getPlaybook, getRisk, getScore,
  getSignalIc, runGnn, runMonteCarlo, trackRec,
  type Candle, type ChartEvent, type EdgarFiling, type ScoreResult,
} from "./api";
import Chart, { type ChartMarker } from "./Chart";
import Sparkline from "./Sparkline";

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
type LabTab = "ic" | "causal" | "patterns" | "gnn";

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
  const [news, setNews] = useState<Record<string, unknown>[]>([]);
  const [forecast, setForecast] = useState<Record<string, unknown> | null>(null);
  const [risk, setRisk] = useState<Record<string, string | number> | null>(null);
  const [mc, setMc] = useState<Record<string, unknown> | null>(null);
  const [opts, setOpts] = useState<Record<string, unknown> | null>(null);
  const [ic, setIc] = useState<Record<string, unknown> | null>(null);
  const [causal, setCausal] = useState<Record<string, unknown> | null>(null);
  const [patterns, setPatterns] = useState<Record<string, unknown> | null>(null);
  const [gnn, setGnn] = useState<Record<string, unknown> | null>(null);
  const [playbook, setPlaybook] = useState<Record<string, unknown> | null>(null);
  const [earnings, setEarnings] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [extrasLoading, setExtrasLoading] = useState(false);
  const [toolBusy, setToolBusy] = useState(false);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [events, setEvents] = useState<ChartEvent[]>([]);
  const [chartLoading, setChartLoading] = useState(false);
  const [edgar, setEdgar] = useState<{ filings?: EdgarFiling[]; signal?: Record<string, unknown> | null; note?: string; success?: boolean; error?: string } | null>(null);
  const [tracked, setTracked] = useState<"idle" | "saving" | "done">("idle");

  async function run() {
    setLoading(true);
    setNews([]);
    setForecast(null);
    setRisk(null);
    setMc(null);
    setOpts(null);
    setIc(null);
    setCausal(null);
    setPatterns(null);
    setGnn(null);
    setPlaybook(null);
    setEarnings(null);
    setTool("main");
    setCandles([]);
    setEvents([]);
    setEdgar(null);
    setTracked("idle");
    try {
      const score = await getScore(input, mode);
      setRes(score);
      setLoading(false);

      setExtrasLoading(true);
      setChartLoading(true);
      getHistory(input, "6mo")
        .then((h) => setCandles(h.candles))
        .catch(() => setCandles([]))
        .finally(() => setChartLoading(false));
      getChartEvents(input, "6mo")
        .then((ev) => setEvents(ev.events ?? []))
        .catch(() => setEvents([]));

      const [n, rk, pb, er] = await Promise.all([
        getNews(input).catch(() => ({ items: [] })),
        getRisk(input).catch(() => ({ metrics: null })),
        getPlaybook(input).catch(() => null),
        getEarnings(input).catch(() => null),
      ]);
      setNews((n.items as Record<string, unknown>[]) ?? []);
      setRisk((rk.metrics as Record<string, string | number>) ?? null);
      setPlaybook(pb);
      setEarnings(er);

      getForecast(input)
        .then((f) => setForecast((f.forecast as Record<string, unknown>) ?? null))
        .catch(() => setForecast(null))
        .finally(() => setExtrasLoading(false));
    } catch {
      setRes(null);
      setLoading(false);
      setExtrasLoading(false);
    }
  }

  async function loadMonte() {
    setToolBusy(true);
    try { setMc(await runMonteCarlo(input)); }
    catch (e) { setMc({ success: false, error: String(e) }); }
    finally { setToolBusy(false); }
  }

  async function loadOptions() {
    setToolBusy(true);
    try { setOpts(await getOptions(input)); }
    catch (e) { setOpts({ success: false, error: String(e) }); }
    finally { setToolBusy(false); }
  }

  async function loadIc() {
    setToolBusy(true);
    try { setIc(await getSignalIc(input)); }
    catch (e) { setIc({ success: false, error: String(e) }); }
    finally { setToolBusy(false); }
  }

  async function loadCausal() {
    setToolBusy(true);
    try { setCausal(await getCausal(input)); }
    catch (e) { setCausal({ success: false, error: String(e) }); }
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
    if (next === "causal" && !causal) void loadCausal();
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
  const mcPath = Array.isArray(mc?.mean_path) ? (mc!.mean_path as number[]) : [];

  const strat = (playbook?.strategy && typeof playbook.strategy === "object")
    ? playbook.strategy as Record<string, unknown>
    : null;
  const primaryStrategy = String(strat?.primary_strategy ?? "");
  const alts = Array.isArray(strat?.recommended_strategies)
    ? (strat!.recommended_strategies as string[]).filter((s) => s !== primaryStrategy).slice(0, 2)
    : [];
  const causalFlags = Array.isArray(causal?.flags) ? (causal!.flags as string[]) : [];
  const causalRecs = Array.isArray(causal?.recommendations)
    ? (causal!.recommendations as string[]) : [];
  const nextEarn = (earnings?.next_earnings && typeof earnings.next_earnings === "object")
    ? earnings.next_earnings as Record<string, unknown>
    : null;

  const regime = String(strat?.market_regime ?? "");
  const regimeHint =
    regime === "bull" ? "trending up"
      : regime === "bear" ? "trending down"
        : regime === "volatile" ? "choppy / high vol"
          : regime === "sideways" ? "range-bound"
            : regime || "unclear";

  return (
    <div className="fade-in">
      <div className="greeting">
        Analyze <small>AI Score, forecast, news & risk</small>
      </div>
      <div className="topbar">
        <div className="search">
          <span className="icon">⌕</span>
          <input value={input}
            onChange={(e) => setInput(e.target.value.toUpperCase())}
            onKeyDown={(e) => e.key === "Enter" && run()} />
        </div>
        <div className="seg">
          <button className={mode === "long" ? "active" : ""} onClick={() => setMode("long")}>Buy / long</button>
          <button className={mode === "short" ? "active" : ""} onClick={() => setMode("short")}>Short</button>
        </div>
        <button className="primary" onClick={run}>
          {loading ? "Analyzing…" : "Analyze"}
        </button>
      </div>

      {!loading && !res && (
        <div className="card card-pad empty">
          Enter a symbol for score, forecast, headlines, and risk.
        </div>
      )}

      {loading && <div className="skeleton" style={{ height: 220, marginBottom: 16 }} />}

      {!loading && res && res.score != null && (
        <>
          <div className="card card-pad" style={{ marginBottom: 16 }}>
            <div style={{ display: "flex", gap: 28, alignItems: "center", flexWrap: "wrap" }}>
              <ScoreRing score={Number(res.score)} />
              <div style={{ flex: 1, minWidth: 200 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <div style={{ fontSize: 22, fontWeight: 700 }}>{res.symbol}</div>
                  <button className="ghost" style={{ fontSize: 12, border: "1px solid var(--border)" }}
                    onClick={trackIdea} disabled={tracked !== "idle"}
                    title="Save this idea (no purchase) — paper-buying later marks it Bought">
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
                    <> · open idea — buy in Portfolio to mark acted</>
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

            {topSignals.length > 0 && (
              <div style={{ marginTop: 18, borderTop: "1px solid var(--border)", paddingTop: 14 }}>
                <div className="rail-label" style={{ marginTop: 0 }}>Key readings</div>
                <div className="dim" style={{ fontSize: 11.5, marginBottom: 8 }}>
                  Raw indicators (mixed units) — not a shared 0–100 score. They feed the 0–10 dimension scores above.
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "7px 28px", maxWidth: 420 }}>
                  {topSignals.map((s) => (
                    <div key={s.name} style={{ display: "contents" }}>
                      <div style={{ fontSize: 13.5, color: "var(--text-2)" }} title={s.hint}>
                        {signalLabel(s.name)}
                      </div>
                      <div className="num" style={{
                        fontSize: 13.5, fontWeight: 650, textAlign: "right",
                        color: s.impact === "positive" ? "var(--up)"
                          : s.impact === "negative" ? "var(--down)" : "var(--text)",
                      }}>
                        {s.display}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="card" style={{ marginBottom: 16 }}>
            <div className="chart-head">
              <div className="legend"><b>{res.symbol}</b> · 6mo daily</div>
            </div>
            {chartLoading ? (
              <div className="skeleton" style={{ height: 320, margin: 18 }} />
            ) : candles.length > 0 ? (
              <Chart candles={candles} markers={[
                ...events.map((e): ChartMarker => ({
                  time: e.time, title: e.title, text: e.text, color: e.color,
                })),
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
                No chart data for {res.symbol}.
              </div>
            )}
          </div>

          <div className="seg" style={{ marginBottom: 12 }}>
            <button className={tool === "main" ? "active" : ""} onClick={() => setTool("main")}>Overview</button>
            <button className={tool === "monte" ? "active" : ""} onClick={() => { setTool("monte"); if (!mc) void loadMonte(); }}>Monte Carlo</button>
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
                      <div className="rail-label" style={{ marginTop: 0 }}>Suggested strategy</div>
                      <div style={{ fontSize: 16, fontWeight: 650, marginBottom: 6 }}>
                        {friendlyStrategy(primaryStrategy)}
                      </div>
                      <div style={{ fontSize: 13.5, lineHeight: 1.5, color: "var(--text-2)", maxWidth: 520 }}>
                        Recent tape looks <b style={{ color: "var(--text)" }}>{regimeHint}</b>
                        {strat.reason ? ` (${String(strat.reason)})` : ""}.
                        {" "}This is a regime-based suggestion — not a ranked backtest winner.
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
                          <div className="label">Avg 1d move</div>
                          <div className="value num" style={{ fontSize: 16 }}>
                            ±{Number(earnings.avg_move_1d).toFixed(1)}%
                          </div>
                        </div>
                      )}
                      {earnings.beat_rate != null && (
                        <div className="card kpi" style={{ padding: "10px 12px" }}>
                          <div className="label">EPS beat rate</div>
                          <div className="value num" style={{ fontSize: 16 }}>
                            {Number(earnings.beat_rate).toFixed(0)}%
                          </div>
                        </div>
                      )}
                      {earnings.positive_reaction_rate != null && (
                        <div className="card kpi" style={{ padding: "10px 12px" }}>
                          <div className="label">Positive reaction</div>
                          <div className="value num" style={{ fontSize: 16 }}>
                            {Number(earnings.positive_reaction_rate).toFixed(0)}%
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )
              )}

              <div className="form-grid" style={{ marginBottom: 16 }}>
                <div className="card card-pad">
                  <div className="rail-label" style={{ marginTop: 0 }}>Forecast</div>
                  {extrasLoading && !forecast && (
                    <div className="dim">Loading consensus forecast…</div>
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
                        {conviction && <> · {conviction.toLowerCase()} conviction</>}
                        {agreement != null && Number.isFinite(agreement) && (
                          <> · agreement {(agreement * 100).toFixed(0)}%</>
                        )}
                      </div>
                      {Array.isArray(forecast?.models_used) && (
                        <div className="dim" style={{ marginTop: 4, fontSize: 11.5 }}>
                          Models: {(forecast!.models_used as string[]).join(", ")}
                        </div>
                      )}
                    </>
                  )}
                </div>
                <div className="card card-pad">
                  <div className="rail-label" style={{ marginTop: 0 }}>Risk · {input}</div>
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

              <div className="card card-pad">
                <div className="rail-label" style={{ marginTop: 0 }}>News · {input}</div>
                {news.length === 0 && <div className="dim">No headlines — add a News API key in Settings for more coverage.</div>}
                {news.slice(0, 6).map((n, i) => (
                  <div key={i} style={{ padding: "10px 0", borderTop: i ? "1px solid var(--border)" : "none" }}>
                    <div style={{ fontWeight: 600 }}>{String(n.title ?? n.headline ?? "Untitled")}</div>
                    <div className="dim" style={{ fontSize: 12, marginTop: 4 }}>
                      {String(n.source ?? n.publisher ?? "")}
                      {n.url ? <> · <a href={String(n.url)} target="_blank" rel="noreferrer" style={{ color: "var(--accent)" }}>open</a></> : null}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}

          {tool === "monte" && (
            <div className="card card-pad">
              <div className="rail-label" style={{ marginTop: 0 }}>Monte Carlo · {input}</div>
              {toolBusy && <div className="dim">Simulating…</div>}
              {!toolBusy && mc?.success === false && (
                <div className="dim">{String(mc.error ?? "Unavailable")}</div>
              )}
              {!toolBusy && mc?.success !== false && mc?.final_p50 != null && (
                <>
                  <div className="kpis" style={{ marginBottom: 12 }}>
                    <div className="card kpi"><div className="label">5th %ile</div>
                      <div className="value num" style={{ fontSize: 18 }}>${Number(mc.final_p5).toLocaleString()}</div></div>
                    <div className="card kpi"><div className="label">Median</div>
                      <div className="value num" style={{ fontSize: 18 }}>${Number(mc.final_p50).toLocaleString()}</div></div>
                    <div className="card kpi"><div className="label">95th %ile</div>
                      <div className="value num" style={{ fontSize: 18 }}>${Number(mc.final_p95).toLocaleString()}</div></div>
                  </div>
                  {mcPath.length > 1 && <Sparkline values={mcPath.map(Number)} width={520} height={56} baseline="start" />}
                </>
              )}
            </div>
          )}

          {tool === "options" && (
            <div className="card card-pad">
              <div className="rail-label" style={{ marginTop: 0 }}>Options sentiment · {input}</div>
              {toolBusy && <div className="dim">Loading…</div>}
              {!toolBusy && opts && (
                opts.success === false
                  ? <div className="dim">{String(opts.error ?? "Unavailable")}</div>
                  : (
                    <pre style={{ whiteSpace: "pre-wrap", fontSize: 12.5, color: "var(--text-2)", margin: 0 }}>
                      {JSON.stringify(opts, null, 2).slice(0, 1200)}
                    </pre>
                  )
              )}
            </div>
          )}

          {tool === "filings" && (
            <div className="card card-pad" style={{ marginBottom: 16 }}>
              <div className="rail-label" style={{ marginTop: 0 }}>SEC filings · {res.symbol}</div>
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
                <button className={lab === "ic" ? "active" : ""} onClick={() => openLab("ic")}>Signal IC</button>
                <button className={lab === "causal" ? "active" : ""} onClick={() => openLab("causal")}>Causal</button>
                <button className={lab === "patterns" ? "active" : ""} onClick={() => openLab("patterns")}>Patterns</button>
                <button className={lab === "gnn" ? "active" : ""} onClick={() => openLab("gnn")}>GNN</button>
              </div>
              {toolBusy && <div className="dim">Computing…</div>}

              {!toolBusy && lab === "ic" && ic && (
                ic.success === false
                  ? <div className="dim">{String(ic.error ?? "IC unavailable")}</div>
                  : (
                    <>
                      <div className="dim" style={{ fontSize: 12.5, marginBottom: 10 }}>
                        Does the score historically predict forward returns? IC above ~0.05 is meaningful.
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

              {!toolBusy && lab === "causal" && causal && (
                causal.success === false
                  ? <div className="dim">{String(causal.error ?? "Unavailable")}</div>
                  : (
                    <>
                      <div className="dim" style={{ fontSize: 12.5, marginBottom: 12 }}>
                        What the return series looks like statistically — informs which models fit.
                        {causal.complexity != null && (
                          <> Complexity: <b style={{ color: "var(--text)" }}>{String(causal.complexity)}</b>.</>
                        )}
                        {causal.n_observations != null && (
                          <> · {Number(causal.n_observations)} observations</>
                        )}
                      </div>
                      {causalFlags.length > 0 && (
                        <>
                          <div className="rail-label" style={{ marginTop: 0 }}>Findings</div>
                          <ul style={{ margin: "0 0 14px", paddingLeft: 18, fontSize: 13.5, lineHeight: 1.55 }}>
                            {causalFlags.map((f, i) => (
                              <li key={i}>{stripEmoji(String(f))}</li>
                            ))}
                          </ul>
                        </>
                      )}
                      {causalRecs.length > 0 && (
                        <>
                          <div className="rail-label">Recommendations</div>
                          <ul style={{ margin: "0 0 14px", paddingLeft: 18, fontSize: 13.5, lineHeight: 1.55 }}>
                            {causalRecs.map((f, i) => (
                              <li key={i}>{stripEmoji(String(f))}</li>
                            ))}
                          </ul>
                        </>
                      )}
                      {causalFlags.length === 0 && causalRecs.length === 0 && (
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

      {!loading && res && res.score == null && (
        <div className="card card-pad empty">
          Couldn't score {res.symbol}{res.error ? ` — ${res.error}` : ""}. Live data required.
        </div>
      )}
    </div>
  );
}
