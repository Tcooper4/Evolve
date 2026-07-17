import { useEffect, useRef, useState } from "react";
import { getScore, runPairs, runScan, type ScoreResult } from "./api";
import UniverseBiasNote from "./UniverseBiasNote";
import PageTour from "./PageTour";

const FILTERS = ["momentum", "oversold", "breakout", "high_short", "insider_buying", "quick_technical"];

const UNIVERSES: { id: string; label: string }[] = [
  { id: "default", label: "Liquid majors (~60, fastest)" },
  { id: "sp100", label: "S&P 100 (~100, fast)" },
  { id: "sp500", label: "S&P 500 (~500)" },
  { id: "nasdaq100", label: "Nasdaq 100 (~100)" },
  { id: "sp500_nasdaq100", label: "S&P 500 + Nasdaq 100 (~500)" },
  { id: "russell1000", label: "Russell 1000 (~1000, slow)" },
  { id: "russell3000", label: "Russell 3000 (~3000, very slow)" },
];

type Tab = "scan" | "pairs";

export default function Scanner({
  onAnalyze,
}: {
  onAnalyze?: (symbol: string) => void;
}) {
  const [tab, setTab] = useState<Tab>("scan");
  const [active, setActive] = useState<string[]>(["momentum"]);
  const [universe, setUniverse] = useState("sp100");
  const [minScore, setMinScore] = useState(6);
  const [maxResults, setMaxResults] = useState(15);
  const [custom, setCustom] = useState("");
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [pairRows, setPairRows] = useState<Record<string, unknown>[]>([]);
  const [meta, setMeta] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [live, setLive] = useState(false);
  const liveRef = useRef(false);
  const [drillSym, setDrillSym] = useState<string | null>(null);
  const [drill, setDrill] = useState<ScoreResult | null>(null);
  const [drillBusy, setDrillBusy] = useState(false);
  const [drillErr, setDrillErr] = useState("");

  async function run() {
    setLoading(true); setError(""); setMeta("");
    setDrillSym(null); setDrill(null); setDrillErr("");
    try {
      const custom_tickers = custom.split(/[,\s]+/).map((s) => s.trim().toUpperCase()).filter(Boolean);
      const r = await runScan({
        filters: active,
        max_results: maxResults,
        universe,
        min_quick_score: minScore,
        custom_tickers,
      });
      if (r.success !== false && !r.error && r.results) {
        setRows(r.results);
        const scanned = r.scanned ?? r.universe_size;
        const passed = r.passed ?? r.results.length;
        if (scanned != null) setMeta(`Scanned ${scanned} · ${passed} passed · showing ${r.results.length}`);
      } else {
        setRows([]);
        setError(r.error ?? "Scan needs live market data.");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "scan failed");
    } finally { setLoading(false); }
  }

  async function runPairScreen() {
    if (loading) return;
    setLoading(true); setError(""); setMeta(""); setPairRows([]);
    try {
      const fromCustom = custom.split(/[,\s]+/).map((s) => s.trim().toUpperCase()).filter(Boolean);
      const r = await runPairs(fromCustom.length >= 2 ? fromCustom : [], {
        universe: fromCustom.length >= 2 ? "" : universe,
        max_symbols: 200,
        max_pairs: 15,
      });
      if (r.success && r.pairs) {
        setPairRows(r.pairs);
        setMeta(
          r.note
            || `Tested ${r.tested ?? 0} symbols · ${r.pairs.length} cointegrated pairs`,
        );
      } else {
        setPairRows([]);
        setError(r.error ?? "Pairs screen failed.");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "pairs failed");
    } finally { setLoading(false); }
  }

  async function openDrill(sym: string) {
    const clean = (sym || "").trim().toUpperCase();
    if (!clean) return;
    setDrillSym(clean);
    setDrill(null);
    setDrillErr("");
    setDrillBusy(true);
    try {
      const s = await getScore(clean, "long");
      setDrill(s);
      if (s.error) setDrillErr(String(s.error));
    } catch (e) {
      setDrillErr(e instanceof Error ? e.message : "score failed");
    } finally {
      setDrillBusy(false);
    }
  }

  useEffect(() => {
    liveRef.current = live;
  }, [live]);

  useEffect(() => {
    if (!live || tab !== "scan" || rows.length === 0) return;
    const id = window.setInterval(() => {
      if (liveRef.current) void run();
    }, 90_000);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [live, tab, rows.length]);

  const SCORE_COLS = [
    "symbol", "quick_score", "price", "change_20d", "rsi", "vs_sma20", "volume_ratio",
  ];
  // Hide bulk AI / short-screen fields from the results table — those belong in drill-in.
  const HIDDEN_COLS = new Set([
    "ai_score", "short_quick_score", "ai_grade", "signals", "pct_from_52w_high",
  ]);
  const COL_LABEL: Record<string, string> = {
    symbol: "Symbol",
    quick_score: "Screen score",
    price: "Price",
    change_20d: "20d chg",
    rsi: "RSI",
    vs_sma20: "vs SMA20",
    volume_ratio: "Vol ratio",
  };
  const cols = rows.length
    ? [
        ...SCORE_COLS.filter((c) => c in rows[0]),
        ...Object.keys(rows[0])
          .filter((c) => !SCORE_COLS.includes(c) && !HIDDEN_COLS.has(c))
          .slice(0, 3),
      ]
    : [];
  const pcols = pairRows.length ? Object.keys(pairRows[0]) : [];

  const drillQuick = (() => {
    if (!drillSym) return null;
    const row = rows.find((r) => String(r.symbol || "").toUpperCase() === drillSym);
    const q = row?.quick_score;
    return typeof q === "number" ? q : q != null ? Number(q) : null;
  })();

  const signals = (() => {
    const list = Array.isArray(drill?.signal_list) ? drill!.signal_list! : [];
    if (list.length) {
      return list
        .filter((s) => s?.name)
        .slice(0, 12)
        .map((s) => ({
          name: String(s.name),
          value: s.value,
          impact: s.impact,
          description: s.description ? String(s.description) : "",
        }));
    }
    return Object.entries(drill?.signals ?? {})
      .slice(0, 8)
      .map(([k, v]) => ({
        name: k,
        value: typeof v === "object" && v && "value" in (v as object)
          ? (v as { value?: unknown }).value
          : v,
        impact: undefined as string | undefined,
        description: "",
      }));
  })();

  return (
    <div className="fade-in">
      <PageTour pageId="scanner" />
      <div className="greeting">
        Scanner{" "}
        <small>
          {tab === "scan"
            ? "fast screen first — click a row for the full AI Score"
            : "Find two stocks that usually move together, then trade when they temporarily split apart."}
        </small>
      </div>

      <div className="seg" style={{ marginBottom: 14 }}>
        <button className={tab === "scan" ? "active" : ""} onClick={() => setTab("scan")}>Scanner</button>
        <button className={tab === "pairs" ? "active" : ""} onClick={() => setTab("pairs")}>Pairs trading</button>
      </div>

      <div className="card card-pad" style={{ marginBottom: 16 }}>
        <div className="form-grid">
          <div className="field">
            <label>Universe</label>
            <select value={universe} onChange={(e) => setUniverse(e.target.value)} disabled={!!custom.trim()}>
              {UNIVERSES.map((u) => <option key={u.id} value={u.id}>{u.label}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Custom tickers (optional)</label>
            <input value={custom} placeholder="AAPL, MSFT, NVDA"
              onChange={(e) => setCustom(e.target.value.toUpperCase())} />
          </div>
          {tab === "scan" && (
            <>
              <div className="field">
                <label>Min screen score</label>
                <input type="number" min={0} max={10} step={0.5} value={minScore}
                  onChange={(e) => setMinScore(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>Max results</label>
                <input type="number" min={5} max={50} step={1} value={maxResults}
                  onChange={(e) => setMaxResults(Number(e.target.value))} />
              </div>
            </>
          )}
        </div>
        {tab === "scan" ? (
          <div className="field" style={{ marginTop: 14 }} data-tour="scanner-filters">
            <label>Filters</label>
            <div className="seg" style={{ display: "inline-flex", flexWrap: "wrap", width: "fit-content", maxWidth: "100%" }}>
              {FILTERS.map((f) => (
                <button key={f} type="button" className={active.includes(f) ? "active" : ""}
                  onClick={() => setActive((a) => a.includes(f) ? a.filter((x) => x !== f) : [...a, f])}>
                  {f.replace(/_/g, " ")}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="dim" data-tour="scanner-filters" style={{ marginTop: 14, fontSize: 12.5 }}>
            Pair screen uses cointegration + correlation ≥ 0.7 (filters apply on the Scanner tab).
          </div>
        )}
        <div
          data-tour="scanner-run"
          style={{ marginTop: 14, display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}
        >
          {tab === "scan"
            ? <button className="primary" onClick={run} disabled={loading}>{loading ? "Scanning…" : "Scan"}</button>
            : <button className="primary" onClick={runPairScreen} disabled={loading}>
                {loading ? "Screening…" : "Find pairs"}
              </button>}
          {tab === "scan" && (
            <label className="dim" style={{ fontSize: 12.5, display: "inline-flex", gap: 8, alignItems: "center", cursor: "pointer" }}>
              <input type="checkbox" checked={live} onChange={(e) => setLive(e.target.checked)} />
              Live refresh (~90s after a scan)
            </label>
          )}
        </div>
        {tab === "scan" && (
          <div className="dim" style={{ marginTop: 10, fontSize: 12.5, lineHeight: 1.45 }}>
            The list uses a fast screen score so scanning stays quick. Click any row for
            that stock&apos;s full AI Score.
          </div>
        )}
        {tab === "scan" && !custom.trim() && (
          <UniverseBiasNote universeId={universe} context="scan" />
        )}
        {tab === "pairs" && (
          <div className="dim" style={{ marginTop: 10, fontSize: 12.5, lineHeight: 1.45 }}>
            Tests every pair in the universe for cointegration (they usually move together)
            and correlation ≥ 0.7. Only statistically linked pairs are shown — zero results
            just means none cleared the bar, not that the screen failed.
          </div>
        )}
        {tab === "pairs" && !custom.trim() && (
          <UniverseBiasNote universeId={universe} context="scan" />
        )}
      </div>

      {meta && !loading && <div className="dim" style={{ marginBottom: 10, fontSize: 12.5 }}>{meta}</div>}
      <div className="card" data-tour="scanner-results">
        {loading && <div className="skeleton" style={{ height: 260, margin: 16 }} />}
        {!loading && tab === "scan" && rows.length > 0 && (
          <table className="tbl">
            <thead><tr>{cols.map((c) => <th key={c}>{COL_LABEL[c] ?? c.replace(/_/g, " ")}</th>)}</tr></thead>
            <tbody>
              {rows.map((r, i) => {
                const sym = String(r.symbol || "");
                const selected = drillSym === sym;
                return (
                  <tr key={i}
                    style={{
                      cursor: "pointer",
                      background: selected ? "rgba(255,255,255,0.04)" : undefined,
                    }}
                    onClick={() => openDrill(sym)}>
                    {cols.map((c) => {
                      const v = r[c];
                      const n = typeof v === "number";
                      const isSym = c === "symbol";
                      return (
                        <td key={c} className={n ? "num" : ""}
                          style={isSym ? { color: "var(--accent)", fontWeight: 650 } : undefined}>
                          {n ? (v as number).toFixed(2) : String(v ?? "—")}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        {!loading && tab === "pairs" && pairRows.length > 0 && (
          <table className="tbl">
            <thead><tr>{pcols.map((c) => <th key={c}>{c.replace(/_/g, " ")}</th>)}</tr></thead>
            <tbody>
              {pairRows.map((r, i) => (
                <tr key={i}>{pcols.map((c) => {
                  const v = r[c];
                  const n = typeof v === "number";
                  return <td key={c} className={n ? "num" : ""}>{n ? (v as number).toFixed(4) : String(v ?? "—")}</td>;
                })}</tr>
              ))}
            </tbody>
          </table>
        )}
        {!loading && ((tab === "scan" && rows.length === 0) || (tab === "pairs" && pairRows.length === 0)) && (
          <div className="empty">{error || (tab === "scan"
            ? "Pick a universe and filters, then Scan."
            : "Pick a universe (or custom tickers), then Find pairs.")}</div>
        )}
      </div>

      {tab === "scan" && drillSym && (
        <div className="card card-pad" style={{ marginTop: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
            <div>
              <div className="rail-label" style={{ marginTop: 0 }}>Full research · {drillSym}</div>
              <div className="dim" style={{ fontSize: 12.5, maxWidth: 520, lineHeight: 1.4 }}>
                Full AI Score for this stock only. It can differ from the screen score —
                that is expected.
              </div>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              {onAnalyze && (
                <button className="primary" onClick={() => onAnalyze(drillSym)}>Open in Analyze</button>
              )}
              <button className="ghost" onClick={() => { setDrillSym(null); setDrill(null); setDrillErr(""); }}>
                Close
              </button>
            </div>
          </div>
          {drillBusy && <div className="skeleton" style={{ height: 100, marginTop: 12 }} />}
          {!drillBusy && drillErr && <div className="dim" style={{ marginTop: 12 }}>⚠ {drillErr}</div>}
          {!drillBusy && drill && (
            <>
              <div className="kpis" style={{ marginTop: 12 }}>
                <div className="card kpi" style={{ padding: "10px 12px" }}>
                  <div className="label">Screen score</div>
                  <div className="value num" style={{ fontSize: 18 }}>
                    {drillQuick != null && Number.isFinite(drillQuick) ? drillQuick.toFixed(1) : "—"}
                  </div>
                  <div className="sub">fast technical filter</div>
                </div>
                <div className="card kpi" style={{ padding: "10px 12px" }}>
                  <div className="label">Full AI Score</div>
                  <div className="value num" style={{ fontSize: 18 }}>
                    {drill.score != null ? Number(drill.score).toFixed(1) : "—"}
                    {drill.grade ? <span className="dim" style={{ fontSize: 12, marginLeft: 6 }}>{drill.grade}</span> : null}
                  </div>
                  <div className="sub">tech + momentum + sentiment + fundamentals</div>
                </div>
                {drillQuick != null && drill.score != null && Number.isFinite(drillQuick) && (
                  <div className="card kpi" style={{ padding: "10px 12px" }}>
                    <div className="label">Gap</div>
                    <div className="value num" style={{
                      fontSize: 18,
                      color: Math.abs(Number(drill.score) - drillQuick) >= 2 ? "var(--down)" : "var(--text-2)",
                    }}>
                      {`${Number(drill.score) - drillQuick >= 0 ? "+" : ""}${(Number(drill.score) - drillQuick).toFixed(1)}`}
                    </div>
                    <div className="sub">full minus screen</div>
                  </div>
                )}
                {([
                  ["Technical", drill.technical_score],
                  ["Momentum", drill.momentum_score],
                  ["Sentiment", drill.sentiment_score],
                  ["Fundamental", drill.fundamental_score],
                ] as const).map(([label, v]) => (
                  <div className="card kpi" key={label} style={{ padding: "10px 12px" }}>
                    <div className="label">{label}</div>
                    <div className="value num" style={{ fontSize: 16 }}>
                      {v != null ? Number(v).toFixed(1) : "—"}
                    </div>
                  </div>
                ))}
              </div>
              {drill.summary && (
                <div style={{ marginTop: 12, fontSize: 13.5, lineHeight: 1.45, color: "var(--text-2)" }}>
                  {String(drill.summary)}
                </div>
              )}
              {signals.length > 0 ? (
                <table className="tbl" style={{ marginTop: 12 }}>
                  <thead>
                    <tr><th>Signal</th><th>Value</th><th>Impact</th><th>Note</th></tr>
                  </thead>
                  <tbody>
                    {signals.map((s) => (
                      <tr key={s.name}>
                        <td style={{ fontWeight: 600 }}>{s.name.replace(/_/g, " ")}</td>
                        <td className="num">
                          {typeof s.value === "number" ? (s.value as number).toFixed(2) : String(s.value ?? "—")}
                        </td>
                        <td style={{
                          color: s.impact === "positive" ? "var(--up)"
                            : s.impact === "negative" ? "var(--down)" : "var(--text-2)",
                        }}>
                          {s.impact ?? "—"}
                        </td>
                        <td className="dim" style={{ fontSize: 12 }}>{s.description.slice(0, 120)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="dim" style={{ marginTop: 12 }}>No per-signal breakdown available for this symbol.</div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
