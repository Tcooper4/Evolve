import { useState } from "react";
import { runBacktest } from "./api";

const STRATS = ["RSIStrategy", "MACDStrategy", "BollingerStrategy", "SMAStrategy"];

export default function Backtest() {
  const [symbol, setSymbol] = useState("SPY");
  const [strategy, setStrategy] = useState(STRATS[0]);
  const [res, setRes] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true);
    try { setRes(await runBacktest(symbol, strategy, {})); }
    catch (e) { setRes({ success: false, error: String(e) }); }
    finally { setLoading(false); }
  }

  const metrics = res && res.success
    ? Object.entries(res).filter(([k, v]) => typeof v === "number" && k !== "success")
    : [];

  return (
    <div className="fade-in">
      <div className="greeting">Backtest <small>run a strategy through the execution-verified evaluation engine</small></div>
      <div className="card card-pad" style={{ marginBottom: 16 }}>
        <div className="form-grid">
          <div className="field"><label>Symbol</label>
            <input value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} /></div>
          <div className="field"><label>Strategy</label>
            <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
              {STRATS.map((s) => <option key={s}>{s}</option>)}
            </select></div>
        </div>
        <div style={{ marginTop: 14 }}>
          <button className="primary" onClick={run}>{loading ? "Running…" : "Run backtest"}</button>
        </div>
      </div>
      {loading && <div className="skeleton" style={{ height: 130 }} />}
      {!loading && res && res.success === true && (
        <div className="kpis fade-in">
          {metrics.slice(0, 4).map(([k, v]) => (
            <div className="card kpi" key={k}>
              <div className="label">{k.replace(/_/g, " ")}</div>
              <div className="value num">{(v as number).toFixed(3)}</div>
            </div>
          ))}
        </div>
      )}
      {!loading && res && res.success !== true && (
        <div className="card card-pad empty">
          {String(res.error ?? "Backtest needs live market data.")}
        </div>
      )}
    </div>
  );
}
