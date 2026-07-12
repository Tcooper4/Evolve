import { useEffect, useRef, useState } from "react";
import { runPairs, runScan } from "./api";

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

  async function run() {
    setLoading(true); setError(""); setMeta("");
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
      // Always use selected universe unless custom tickers are provided
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

  useEffect(() => {
    liveRef.current = live;
  }, [live]);

  // Live refresh only re-runs after the user has already scanned once —
  // toggling the checkbox alone must not start a scan.
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
  const cols = rows.length
    ? [
        ...SCORE_COLS.filter((c) => c in rows[0]),
        ...Object.keys(rows[0]).filter((c) => !SCORE_COLS.includes(c)).slice(0, 3),
      ]
    : [];
  const pcols = pairRows.length ? Object.keys(pairRows[0]) : [];

  return (
    <div className="fade-in">
      <div className="greeting">
        Scanner{" "}
        <small>
          {tab === "pairs"
            ? "Find two stocks that usually move together, then trade when they temporarily split apart."
            : "screen the market by technical filters and quick score"}
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
                <label>Min quick score</label>
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
        {tab === "scan" && (
          <div className="field" style={{ marginTop: 14 }}>
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
        )}
        <div style={{ marginTop: 14, display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
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
        {tab === "pairs" && (
          <div className="dim" style={{ marginTop: 10, fontSize: 12.5, lineHeight: 1.45 }}>
            Tests every pair in the universe for cointegration (they usually move together)
            and correlation ≥ 0.7. Only statistically linked pairs are shown — zero results
            just means none cleared the bar, not that the screen failed.
          </div>
        )}
      </div>

      {meta && !loading && <div className="dim" style={{ marginBottom: 10, fontSize: 12.5 }}>{meta}</div>}
      <div className="card">
        {loading && <div className="skeleton" style={{ height: 260, margin: 16 }} />}
        {!loading && tab === "scan" && rows.length > 0 && (
          <table className="tbl">
            <thead><tr>{cols.map((c) => <th key={c}>{c.replace(/_/g, " ")}</th>)}</tr></thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} style={{ cursor: onAnalyze ? "pointer" : undefined }}
                  onClick={() => {
                    const sym = String(r.symbol || "");
                    if (sym && onAnalyze) onAnalyze(sym);
                  }}>
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
              ))}
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
    </div>
  );
}
