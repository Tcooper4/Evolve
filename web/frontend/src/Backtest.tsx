import { useEffect, useState } from "react";
import {
  getStrategies, runBacktest, runModelBacktest, runOptimize, runTuneModels,
  type WalkForwardFold,
} from "./api";
import Sparkline from "./Sparkline";

const PCT_KEYS = new Set([
  "total_return", "annualized_return", "max_drawdown", "buy_hold_return",
  "day_win_rate", "trade_win_rate", "win_rate", "excess_vs_bh",
  "mean_directional_accuracy", "hit_rate_5pct", "hit_rate_10pct",
  "illustrative_return",
]);

const PERIODS = ["6mo", "1y", "2y", "5y"];

/** Least → most aggressive objective (label shown in the dropdown). */
const OPT_OBJECTIVES = [
  { id: "sharpe_ratio", label: "Least risky — Sharpe" },
  { id: "total_return", label: "More aggressive — Total return" },
  { id: "excess_vs_bh", label: "Most aggressive — Beat buy & hold" },
] as const;

const FORECAST_MODELS = [
  { id: "xgboost", label: "XGBoost" },
  { id: "ridge", label: "Ridge" },
  { id: "catboost", label: "CatBoost" },
  { id: "arima", label: "ARIMA" },
  { id: "prophet", label: "Prophet" },
  { id: "garch", label: "GARCH" },
] as const;

const PARAM_LABEL: Record<string, string> = {
  rsi_period: "RSI period",
  oversold_threshold: "Oversold",
  overbought_threshold: "Overbought",
  fast_period: "Fast",
  slow_period: "Slow",
  signal_period: "Signal",
  window: "Window",
  num_std: "Std",
  short_window: "Short",
  long_window: "Long",
  confirmation_periods: "Confirm",
  n_estimators: "Trees",
  max_depth: "Depth",
  depth: "Depth",
  learning_rate: "Learn rate",
  subsample: "Subsample",
  colsample_bytree: "Col sample",
  reg_alpha: "L1",
  reg_lambda: "L2",
  iterations: "Iterations",
  alpha: "Alpha",
  changepoint_prior_scale: "Changepoint",
  p: "p",
  q: "q",
};

function prettyStrategy(id: string) {
  return id.replace(/Strategy$/, "");
}

function formatMetric(key: string, v: number): string {
  if (key === "mean_mape") return `${v.toFixed(2)}%`;
  if (key === "n_windows" || key === "n_trades") {
    return Number.isInteger(v) ? String(v) : v.toFixed(0);
  }
  if (PCT_KEYS.has(key)) {
    const scaled = Math.abs(v) <= 1.5 ? v * 100 : v;
    return `${scaled.toFixed(2)}%`;
  }
  if (key === "consistency_score") {
    return Number.isInteger(v) ? String(v) : v.toFixed(2);
  }
  return v.toFixed(3);
}

function formatParamValue(key: string, v: unknown): string {
  if (typeof v === "number") {
    if (key.includes("rate") || key.includes("sample") || key.startsWith("reg_")) {
      return v < 0.01 ? v.toExponential(2) : v.toFixed(v < 1 ? 3 : 2);
    }
    return Number.isInteger(v) ? String(v) : v.toFixed(3);
  }
  return String(v ?? "—");
}

function flattenParams(params: unknown): [string, unknown][] {
  if (!params || typeof params !== "object") return [];
  const out: [string, unknown][] = [];
  for (const [k, v] of Object.entries(params as Record<string, unknown>)) {
    if (v && typeof v === "object" && !Array.isArray(v) && k.includes("params")) {
      for (const [ik, iv] of Object.entries(v as Record<string, unknown>)) out.push([ik, iv]);
    } else out.push([k, v]);
  }
  return out;
}

function scoreLabel(model: string, score: unknown): string {
  if (score == null || !Number.isFinite(Number(score))) return "—";
  const n = Number(score);
  return model === "garch" ? `AIC ${n.toFixed(1)}` : `RMSE ${n.toFixed(4)}`;
}

type Tab = "backtest" | "optimize" | "models";
type Engine = "strategy" | "model";
type ChartView = "equity" | "compare";

export default function Backtest() {
  const [tab, setTab] = useState<Tab>("backtest");
  const [engine, setEngine] = useState<Engine>("strategy");
  const [chartView, setChartView] = useState<ChartView>("equity");
  const [symbol, setSymbol] = useState("SPY");
  const [strategies, setStrategies] = useState([
    "RSIStrategy", "MACDStrategy", "BollingerStrategy", "SMAStrategy",
  ]);
  const [strategy, setStrategy] = useState("RSIStrategy");
  const [model, setModel] = useState("xgboost");
  const [period, setPeriod] = useState("1y");
  const [optMetric, setOptMetric] = useState<string>("sharpe_ratio");
  const [res, setRes] = useState<Record<string, unknown> | null>(null);
  const [opt, setOpt] = useState<Record<string, unknown> | null>(null);
  const [tune, setTune] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem("evolve_backtest");
      if (raw) {
        const pref = JSON.parse(raw) as { symbol?: string; strategy?: string };
        if (pref.symbol) setSymbol(String(pref.symbol).toUpperCase());
        if (pref.strategy) {
          setStrategy(String(pref.strategy));
          setEngine("strategy");
        }
        sessionStorage.removeItem("evolve_backtest");
      }
    } catch { /* ignore */ }
    getStrategies().then((r) => {
      if (r.strategies?.length) {
        setStrategies(r.strategies);
        setStrategy((s) => (r.strategies.includes(s) ? s : r.strategies[0]));
      }
    }).catch(() => {});
  }, []);

  async function run(params: Record<string, unknown> = {}) {
    setLoading(true);
    try {
      if (engine === "model") {
        setRes(await runModelBacktest(symbol, model, period));
      } else {
        setRes(await runBacktest(symbol, strategy, params, period));
      }
    } catch (e) {
      setRes({ success: false, error: String(e) });
    } finally {
      setLoading(false);
    }
  }

  async function optimize() {
    setLoading(true);
    try { setOpt(await runOptimize(strategy, symbol, 24, period, optMetric)); }
    catch (e) { setOpt({ success: false, error: String(e) }); }
    finally { setLoading(false); }
  }

  async function tuneModels() {
    setLoading(true);
    try { setTune(await runTuneModels(symbol, 12)); }
    catch (e) { setTune({ success: false, error: String(e) }); }
    finally { setLoading(false); }
  }

  const isModel = res?.kind === "model";
  const metrics = res && res.success
    ? Object.entries(res).filter(([k, v]) => {
      if (typeof v !== "number") return false;
      if (isModel) {
        return [
          "mean_mape", "mean_directional_accuracy", "hit_rate_10pct",
          "n_windows", "consistency_score",
        ].includes(k);
      }
      return [
        "total_return", "buy_hold_return", "sharpe_ratio",
        "max_drawdown", "trade_win_rate", "n_trades",
      ].includes(k);
    })
    : [];

  const folds: WalkForwardFold[] = Array.isArray(res?.folds)
    ? (res!.folds as WalkForwardFold[])
    : [];
  const foldDAs = folds.map((f) => f.directional_accuracy).filter((v): v is number => v != null);
  const foldStability = foldDAs.length >= 2
    ? foldDAs.filter((v) => v >= 0.5).length / foldDAs.length
    : null;

  const equityRaw = Array.isArray(res?.equity)
    ? (res!.equity as { time: string; value?: number | null }[])
    : [];
  const equity = equityRaw.map((p) =>
    (typeof p.value === "number" && Number.isFinite(p.value) ? p.value : null));
  const equityTimes = equityRaw.map((p) => p.time);

  const comparison = Array.isArray(res?.comparison)
    ? (res!.comparison as {
      actual?: number | null; predicted?: number | null;
      position?: number | null; time?: string;
    }[])
    : [];
  const cmpA = comparison.map((p) =>
    (typeof p.actual === "number" && Number.isFinite(p.actual) ? p.actual : null));
  const cmpB = comparison.map((p) =>
    (typeof p.predicted === "number" && Number.isFinite(p.predicted) ? p.predicted : null));
  const cmpPos = comparison.map((p) => (typeof p.position === "number" ? p.position : null));
  const cmpTimes = comparison.map((p) => p.time ?? null);

  const showEquity = chartView === "equity" && equity.filter((v) => v != null).length > 1;
  const showCompare = chartView === "compare"
    && cmpA.filter((v) => v != null).length > 1
    && cmpB.filter((v) => v != null).length > 1;

  const bestParams = (opt?.best_params && typeof opt.best_params === "object")
    ? Object.entries(opt.best_params as Record<string, unknown>)
    : [];
  const oos = (opt?.oos_metrics && typeof opt.oos_metrics === "object")
    ? Object.entries(opt.oos_metrics as Record<string, unknown>)
      .filter(([k, v]) => typeof v === "number" && ["total_return", "sharpe_ratio", "max_drawdown", "excess_vs_bh", "buy_hold_return"].includes(k))
    : [];
  const tuneResults = (tune?.results && typeof tune.results === "object")
    ? tune.results as Record<string, Record<string, unknown>>
    : {};

  return (
    <div className="fade-in">
      <div className="greeting">
        Backtest <small>strategies, models, and tuning</small>
      </div>

      <div className="seg" style={{ marginBottom: 14 }}>
        <button className={tab === "backtest" ? "active" : ""} onClick={() => { setTab("backtest"); setOpt(null); setTune(null); }}>Backtest</button>
        <button className={tab === "optimize" ? "active" : ""} onClick={() => { setTab("optimize"); setRes(null); setTune(null); }}>Optimize</button>
        <button className={tab === "models" ? "active" : ""} onClick={() => { setTab("models"); setRes(null); setOpt(null); }}>Model tune</button>
      </div>

      <div className="card card-pad" style={{ marginBottom: 16 }}>
        {tab === "backtest" && (
          <div className="seg" style={{ marginBottom: 14 }}>
            <button className={engine === "strategy" ? "active" : ""}
              onClick={() => { setEngine("strategy"); setRes(null); }}>Strategy</button>
            <button className={engine === "model" ? "active" : ""}
              onClick={() => { setEngine("model"); setRes(null); }}>Model</button>
          </div>
        )}
        <div className="form-grid">
          <div className="field"><label>Symbol</label>
            <input value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} /></div>
          {(tab === "backtest" && engine === "strategy") || tab === "optimize" ? (
            <div className="field"><label>Strategy</label>
              <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
                {strategies.map((s) => <option key={s} value={s}>{prettyStrategy(s)}</option>)}
              </select></div>
          ) : null}
          {tab === "backtest" && engine === "model" && (
            <div className="field"><label>Model</label>
              <select value={model} onChange={(e) => setModel(e.target.value)}>
                {FORECAST_MODELS.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
              </select></div>
          )}
          {(tab === "backtest" || tab === "optimize") && (
            <div className="field"><label>Period</label>
              <select value={period} onChange={(e) => setPeriod(e.target.value)}>
                {PERIODS.map((p) => <option key={p}>{p}</option>)}
              </select></div>
          )}
          {tab === "optimize" && (
            <div className="field"><label>Goal</label>
              <select value={optMetric} onChange={(e) => setOptMetric(e.target.value)}>
                {OPT_OBJECTIVES.map((o) => (
                  <option key={o.id} value={o.id}>{o.label}</option>
                ))}
              </select></div>
          )}
        </div>
        <div style={{ marginTop: 14 }}>
          {tab === "backtest" && (
            <button className="primary" onClick={() => run()} disabled={loading}>
              {loading ? "Running…" : (engine === "model" ? "Run model backtest" : "Run backtest")}
            </button>
          )}
          {tab === "optimize" && (
            <button className="primary" onClick={optimize} disabled={loading}>
              {loading ? "Optimizing…" : "Tune parameters"}
            </button>
          )}
          {tab === "models" && (
            <button className="primary" onClick={tuneModels} disabled={loading}>
              {loading ? "Tuning…" : `Tune models on ${symbol}`}
            </button>
          )}
        </div>
      </div>

      {loading && <div className="skeleton" style={{ height: 120 }} />}

      {tab === "backtest" && !loading && res?.success === true && (
        <>
          {!isModel && Number(res.active_bars) === 0 && (
            <div className="card card-pad" style={{ marginBottom: 12, color: "var(--text-2)" }}>
              No trades in this window.
            </div>
          )}
          {!!res.used_saved_params && (
            <div className="dim" style={{ marginBottom: 10, fontSize: 12.5 }}>
              Using saved parameters
            </div>
          )}
          <div className="kpis fade-in">
            {metrics.slice(0, 6).map(([k, v]) => (
              <div className="card kpi" key={k}>
                <div className="label">{k.replace(/_/g, " ")}</div>
                <div className="value num">{formatMetric(k, v as number)}</div>
              </div>
            ))}
          </div>

          {isModel && (
            <div className="card card-pad fade-in" style={{ marginTop: 16 }}>
              <div className="rail-label" style={{ marginTop: 0 }}>
                Fold by fold — does it hold up across time?
              </div>
              {folds.length > 0 ? (
                <>
                  <div className="dim" style={{ fontSize: 12.5, marginBottom: 12 }}>
                    Each chip is one out-of-sample window. Green = called direction
                    right more often than a coin flip; red = worse than one. A model
                    that's only green in one era isn't a model, it's a memory.
                  </div>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {folds.map((f) => {
                      const da = f.directional_accuracy;
                      const col = da == null ? "var(--text-3)"
                        : da >= 0.55 ? "var(--up)"
                          : da <= 0.45 ? "var(--down)" : "var(--text-2)";
                      return (
                        <div key={f.window} className="card" style={{ padding: "8px 12px", minWidth: 118 }}
                          title={`train ${f.train_start} → ${f.train_end}${f.mape != null ? ` · MAPE ${f.mape.toFixed(1)}%` : ""}`}>
                          <div className="dim" style={{ fontSize: 10.5 }}>
                            {f.test_start} → {f.test_end}
                          </div>
                          <div className="num" style={{ fontSize: 16, fontWeight: 700, color: col }}>
                            {da != null ? `${(da * 100).toFixed(0)}%` : "—"}
                          </div>
                          <div className="dim" style={{ fontSize: 10.5 }}>direction right</div>
                        </div>
                      );
                    })}
                  </div>
                  {foldStability != null && (
                    <div className="dim" style={{ fontSize: 12.5, marginTop: 10 }}>
                      {foldStability >= 0.75
                        ? `Consistent: beat a coin flip in ${Math.round(foldStability * foldDAs.length)} of ${foldDAs.length} windows.`
                        : foldStability >= 0.5
                          ? `Mixed: beat a coin flip in only ${Math.round(foldStability * foldDAs.length)} of ${foldDAs.length} windows — treat the average with suspicion.`
                          : `Unstable: worse than a coin flip in most windows. The headline average is hiding this.`}
                    </div>
                  )}
                </>
              ) : (
                <div className="dim" style={{ fontSize: 12.5 }}>
                  {typeof res.folds_note === "string" && res.folds_note
                    ? String(res.folds_note)
                    : "No per-window fold detail for this run — need enough history for ≥2 walk-forward windows. Headline averages above can hide regime luck."}
                </div>
              )}
            </div>
          )}

          {(showEquity || showCompare || equity.filter((v) => v != null).length > 1 || cmpA.length > 1) && (
            <div className="card card-pad fade-in" style={{ marginTop: 16 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, marginBottom: 12, flexWrap: "wrap" }}>
                <div className="rail-label" style={{ marginTop: 0, marginBottom: 0 }}>
                  {chartView === "compare"
                    ? (isModel ? "Forecast vs actual" : "Buy & hold vs strategy")
                    : "Equity"}
                </div>
                <div className="seg">
                  <button className={chartView === "equity" ? "active" : ""}
                    onClick={() => setChartView("equity")}>Equity</button>
                  <button className={chartView === "compare" ? "active" : ""}
                    onClick={() => setChartView("compare")}>
                    {isModel ? "Forecast vs actual" : "vs buy & hold"}
                  </button>
                </div>
              </div>
              {showEquity && (
                <>
                  <Sparkline values={equity} times={equityTimes} width={640} height={88} baseline="start" />
                  <div className="dim" style={{ marginTop: 8, fontSize: 12.5 }}>
                    {isModel ? "Illustrative path" : "Growth of $1"}
                  </div>
                </>
              )}
              {showCompare && (
                <>
                  <Sparkline
                    values={cmpA}
                    compare={cmpB}
                    positions={!isModel ? cmpPos : undefined}
                    times={cmpTimes}
                    width={640}
                    height={96}
                    baseline="start"
                    color="var(--text)"
                    compareColor="var(--accent)"
                    strokeWidth={2.2}
                    compareStrokeWidth={1.6}
                    compareDash="6 5"
                  />
                  <div className="dim" style={{ marginTop: 8, fontSize: 12.5, display: "flex", gap: 14 }}>
                    <span><span style={{ color: "var(--text)" }}>━</span> {isModel ? "Actual" : "Buy & hold"}</span>
                    <span><span style={{ color: "var(--accent)" }}>┄</span> {isModel ? "Forecast" : "Strategy"}</span>
                  </div>
                </>
              )}
            </div>
          )}
        </>
      )}

      {tab === "backtest" && !loading && res && res.success !== true && (
        <div className="card card-pad empty">{String(res.error ?? "Backtest failed.")}</div>
      )}

      {tab === "optimize" && !loading && opt && (
        <div className="card card-pad fade-in">
          {opt.success === false ? (
            <div className="dim">{String(opt.error ?? "Optimization failed")}</div>
          ) : (
            <>
              {typeof opt.warning === "string" && opt.warning && (
                <div className="dim" style={{ marginBottom: 12, color: "var(--down)" }}>{String(opt.warning)}</div>
              )}
              <div className="rail-label" style={{ marginTop: 0 }}>
                Best parameters{opt.saved === true ? " · saved" : ""}
              </div>
              <div className="kpis" style={{ marginBottom: 14 }}>
                {bestParams.slice(0, 6).map(([k, v]) => (
                  <div className="card kpi" key={k}>
                    <div className="label">{PARAM_LABEL[k] ?? k.replace(/_/g, " ")}</div>
                    <div className="value num" style={{ fontSize: 16 }}>{formatParamValue(k, v)}</div>
                  </div>
                ))}
              </div>
              {oos.length > 0 && (
                <>
                  <div className="rail-label">Out-of-sample</div>
                  <div className="kpis" style={{ marginBottom: 14 }}>
                    {oos.map(([k, v]) => (
                      <div className="card kpi" key={k}>
                        <div className="label">{k.replace(/_/g, " ")}</div>
                        <div className="value num" style={{ fontSize: 16 }}>{formatMetric(k, v as number)}</div>
                      </div>
                    ))}
                  </div>
                </>
              )}
              {bestParams.length > 0 && (
                <button
                  className="primary"
                  onClick={() => {
                    const p = (opt.best_params || {}) as Record<string, unknown>;
                    setEngine("strategy");
                    setTab("backtest");
                    setOpt(null);
                    void run(p);
                  }}
                >
                  Run backtest with these
                </button>
              )}
            </>
          )}
        </div>
      )}

      {tab === "models" && !loading && tune && (
        <div className="fade-in">
          {tune.success === false ? (
            <div className="card card-pad"><div className="dim">{String(tune.error ?? "Tuning failed")}</div></div>
          ) : (
            <div style={{ display: "grid", gap: 12 }}>
              {Object.entries(tuneResults).map(([name, payload]) => {
                const params = flattenParams(payload.best_params);
                return (
                  <div className="card card-pad" key={name}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                      <div style={{ fontWeight: 650, textTransform: "capitalize" }}>{name}</div>
                      <div className="num dim" style={{ fontSize: 13 }}>{scoreLabel(name, payload.best_score)}</div>
                    </div>
                    <div className="kpis">
                      {params.slice(0, 6).map(([k, v]) => (
                        <div className="card kpi" key={k} style={{ padding: "10px 12px" }}>
                          <div className="label">{PARAM_LABEL[k] ?? k.replace(/_/g, " ")}</div>
                          <div className="value num" style={{ fontSize: 15 }}>{formatParamValue(k, v)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
