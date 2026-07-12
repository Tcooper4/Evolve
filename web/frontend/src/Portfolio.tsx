import { useCallback, useEffect, useState } from "react";
import {
  adjustCash, cancelLimit, deleteAlert, getAlerts, getCashbook, getPortfolio,
  getPortfolioTrades, getRisk, placeLimit, recordTrade, runAllocate, upsertAlert,
  type PortfolioSummary,
} from "./api";

export default function Portfolio() {
  const [data, setData] = useState<PortfolioSummary | null>(null);
  const [trades, setTrades] = useState<Record<string, unknown>[]>([]);
  const [riskSym, setRiskSym] = useState("");
  const [risk, setRisk] = useState<Record<string, string | number> | null>(null);
  const [symbol, setSymbol] = useState("");
  const [qty, setQty] = useState("");
  const [price, setPrice] = useState("");
  const [msg, setMsg] = useState("");
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<"positions" | "trades" | "risk" | "alerts" | "allocate" | "cash">("positions");
  const [alerts, setAlerts] = useState<Record<string, unknown>[]>([]);
  const [triggered, setTriggered] = useState<Record<string, unknown>[]>([]);
  const [alertSym, setAlertSym] = useState("");
  const [alertThr, setAlertThr] = useState("");
  const [alertCond, setAlertCond] = useState("price_above");
  const [allocInput, setAllocInput] = useState("SPY, QQQ, TLT, GLD");
  const [alloc, setAlloc] = useState<Record<string, unknown> | null>(null);
  const [cash, setCash] = useState<number | null>(null);
  const [orders, setOrders] = useState<Record<string, unknown>[]>([]);
  const [limSym, setLimSym] = useState("");
  const [limQty, setLimQty] = useState("");
  const [limPx, setLimPx] = useState("");
  const [limSide, setLimSide] = useState("buy");

  const refresh = useCallback(async () => {
    try {
      const [p, t] = await Promise.all([getPortfolio(), getPortfolioTrades().catch(() => [])]);
      setData(p);
      setTrades(Array.isArray(t) ? t : []);
    } catch { setData(null); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  async function trade(side: "buy" | "sell") {
    if (!symbol || !qty) { setMsg("Symbol and quantity required."); return; }
    const r = await recordTrade(symbol, side, Number(qty),
                                price ? Number(price) : undefined);
    setMsg(r.success
      ? side === "sell" && r.realized_pnl != null
        ? `Sold — realized ${r.realized_pnl >= 0 ? "+" : ""}$${r.realized_pnl}`
        : "Recorded."
      : `⚠ ${r.error}`);
    if (r.success) { setSymbol(""); setQty(""); setPrice(""); refresh(); }
    setTimeout(() => setMsg(""), 4000);
  }

  async function loadRisk() {
    if (!riskSym) return;
    try {
      const r = await getRisk(riskSym);
      setRisk(r.metrics ?? null);
      if (!r.success) setMsg(r.error ?? "Risk unavailable");
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "risk failed");
    }
  }

  async function loadAlerts() {
    try {
      const r = await getAlerts();
      setAlerts(r.alerts ?? []);
      setTriggered(r.triggered ?? []);
    } catch { setAlerts([]); }
  }

  async function addAlert() {
    if (!alertSym || !alertThr) return;
    await upsertAlert(alertSym, alertCond, Number(alertThr));
    setAlertSym(""); setAlertThr("");
    await loadAlerts();
  }

  async function removeAlert(id: string) {
    await deleteAlert(id);
    await loadAlerts();
  }

  async function allocate() {
    const syms = allocInput.split(/[,\s]+/).map((s) => s.trim().toUpperCase()).filter(Boolean);
    try { setAlloc(await runAllocate(syms)); }
    catch (e) { setAlloc({ success: false, error: String(e) }); }
  }

  async function loadCash() {
    try {
      const r = await getCashbook();
      setCash(r.cash ?? null);
      setOrders(r.limit_orders ?? []);
    } catch { setCash(null); }
  }

  async function deposit(n: number) {
    const r = await adjustCash(n);
    setCash(r.cash);
  }

  async function addLimit() {
    if (!limSym || !limQty || !limPx) return;
    const r = await placeLimit(limSym, limSide, Number(limQty), Number(limPx));
    setOrders(r.limit_orders ?? []);
    setLimSym(""); setLimQty(""); setLimPx("");
  }

  const pnlCls = (v: number | null) =>
    v == null ? "" : v >= 0 ? "up" : "down";

  const weights = alloc && typeof alloc.weights === "object" && alloc.weights
    ? Object.entries(alloc.weights as Record<string, number>)
    : alloc && typeof alloc.allocation === "object" && alloc.allocation
      ? Object.entries(alloc.allocation as Record<string, number>)
      : [];

  return (
    <div className="fade-in">
      <div className="greeting">
        Portfolio <small>paper only — practice with zero risk, tracked like the real thing</small>
      </div>

      {data && (
        <div className="kpis">
          <div className="card kpi"><div className="label">Market value</div>
            <div className="value num">${data.total_market_value.toLocaleString()}</div>
            {!data.all_prices_live && <div className="sub">some prices unavailable — showing cost</div>}
          </div>
          <div className="card kpi"><div className="label">Cost basis</div>
            <div className="value num">${data.total_cost_basis.toLocaleString()}</div></div>
          <div className="card kpi"><div className="label">Unrealized P&L</div>
            <div className={`value num ${pnlCls(data.total_unrealized_pnl)}`}>
              {data.total_unrealized_pnl >= 0 ? "+" : ""}${data.total_unrealized_pnl.toLocaleString()}
            </div><div className="sub">open positions</div></div>
          <div className="card kpi"><div className="label">Realized P&L</div>
            <div className={`value num ${pnlCls(data.realized_pnl)}`}>
              {data.realized_pnl >= 0 ? "+" : ""}${data.realized_pnl.toLocaleString()}
            </div><div className="sub">closed trades, all time</div></div>
        </div>
      )}

      <div className="card card-pad" style={{ marginBottom: 16 }}>
        <div className="row" style={{ flexWrap: "wrap" }}>
          <input placeholder="Symbol" value={symbol} style={{ width: 110 }}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())} />
          <input placeholder="Qty" value={qty} style={{ width: 90 }}
            onChange={(e) => setQty(e.target.value)} />
          <input placeholder="Price (blank = live)" value={price} style={{ width: 150 }}
            onChange={(e) => setPrice(e.target.value)} />
          <button className="primary" onClick={() => trade("buy")}>Buy</button>
          <button onClick={() => trade("sell")}>Sell</button>
          {msg && <span className="dim">{msg}</span>}
        </div>
      </div>

      <div className="seg" style={{ marginBottom: 12, flexWrap: "wrap" }}>
        <button className={tab === "positions" ? "active" : ""} onClick={() => setTab("positions")}>Positions</button>
        <button className={tab === "trades" ? "active" : ""} onClick={() => setTab("trades")}>Trade ledger</button>
        <button className={tab === "cash" ? "active" : ""} onClick={() => { setTab("cash"); void loadCash(); }}>Cash / limits</button>
        <button className={tab === "risk" ? "active" : ""} onClick={() => setTab("risk")}>Symbol risk</button>
        <button className={tab === "alerts" ? "active" : ""} onClick={() => { setTab("alerts"); void loadAlerts(); }}>Alerts</button>
        <button className={tab === "allocate" ? "active" : ""} onClick={() => setTab("allocate")}>Allocate</button>
      </div>

      <div className="card">
        {loading && <div className="skeleton" style={{ height: 160, margin: 16 }} />}
        {!loading && tab === "positions" && data && data.positions.length > 0 && (
          <table className="tbl">
            <thead><tr>
              <th>Symbol</th><th>Qty</th><th>Avg cost</th><th>Last</th>
              <th>Value</th><th>P&L</th><th>P&L %</th>
            </tr></thead>
            <tbody>
              {data.positions.map((p) => (
                <tr key={p.symbol}>
                  <td style={{ fontWeight: 650 }}>{p.symbol}</td>
                  <td className="num">{p.quantity}</td>
                  <td className="num">{p.avg_cost.toFixed(2)}</td>
                  <td className="num">{p.last_price?.toFixed(2) ?? "—"}</td>
                  <td className="num">{p.market_value?.toLocaleString() ?? "—"}</td>
                  <td className={`num ${pnlCls(p.unrealized_pnl)}`}>
                    {p.unrealized_pnl != null
                      ? `${p.unrealized_pnl >= 0 ? "+" : ""}${p.unrealized_pnl.toLocaleString()}` : "—"}
                  </td>
                  <td className={`num ${pnlCls(p.unrealized_pct)}`}>
                    {p.unrealized_pct != null ? `${p.unrealized_pct.toFixed(2)}%` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {!loading && tab === "positions" && (!data || data.positions.length === 0) && (
          <div className="empty">
            No positions yet. Record a paper buy above — or ask Chat
            "buy 10 shares of SPY on paper".
          </div>
        )}
        {!loading && tab === "trades" && (
          trades.length ? (
            <table className="tbl">
              <thead><tr>
                {Object.keys(trades[0]).slice(0, 6).map((c) => <th key={c}>{c.replace(/_/g, " ")}</th>)}
              </tr></thead>
              <tbody>
                {trades.slice().reverse().slice(0, 40).map((t, i) => (
                  <tr key={i}>
                    {Object.keys(trades[0]).slice(0, 6).map((c) => {
                      const v = t[c];
                      const n = typeof v === "number";
                      return <td key={c} className={n ? "num" : ""}>{n ? (v as number).toFixed(2) : String(v ?? "—")}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <div className="empty">No trades recorded yet.</div>
        )}
        {!loading && tab === "cash" && (
          <div className="card-pad">
            <div className="kpis" style={{ marginBottom: 14 }}>
              <div className="card kpi">
                <div className="label">Paper cash</div>
                <div className="value num">${(cash ?? 0).toLocaleString()}</div>
              </div>
            </div>
            <div className="row" style={{ marginBottom: 14, flexWrap: "wrap" }}>
              <button onClick={() => deposit(10000)}>+$10k</button>
              <button onClick={() => deposit(50000)}>+$50k</button>
              <button onClick={() => deposit(-10000)}>−$10k</button>
            </div>
            <div className="rail-label">Limit orders</div>
            <div className="row" style={{ marginBottom: 12, flexWrap: "wrap" }}>
              <input placeholder="Symbol" value={limSym} style={{ width: 90 }}
                onChange={(e) => setLimSym(e.target.value.toUpperCase())} />
              <select value={limSide} onChange={(e) => setLimSide(e.target.value)} style={{ width: 90 }}>
                <option value="buy">Buy</option>
                <option value="sell">Sell</option>
              </select>
              <input placeholder="Qty" value={limQty} style={{ width: 80 }}
                onChange={(e) => setLimQty(e.target.value)} />
              <input placeholder="Limit px" value={limPx} style={{ width: 100 }}
                onChange={(e) => setLimPx(e.target.value)} />
              <button className="primary" onClick={addLimit}>Park order</button>
            </div>
            {orders.length === 0 ? (
              <div className="dim">No open limit orders.</div>
            ) : (
              <table className="tbl">
                <thead><tr><th>Side</th><th>Symbol</th><th>Qty</th><th>Limit</th><th /></tr></thead>
                <tbody>
                  {orders.map((o) => (
                    <tr key={String(o.id)}>
                      <td>{String(o.side)}</td>
                      <td style={{ fontWeight: 650 }}>{String(o.symbol)}</td>
                      <td className="num">{String(o.quantity)}</td>
                      <td className="num">{Number(o.limit_price).toFixed(2)}</td>
                      <td><button className="ghost" onClick={async () => {
                        const r = await cancelLimit(String(o.id));
                        setOrders(r.limit_orders ?? []);
                      }}>Cancel</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}
        {!loading && tab === "risk" && (
          <div className="card-pad">
            <div className="row" style={{ marginBottom: 14 }}>
              <input placeholder="Symbol" value={riskSym} style={{ width: 120 }}
                onChange={(e) => setRiskSym(e.target.value.toUpperCase())} />
              <button className="primary" onClick={loadRisk}>Load risk</button>
            </div>
            {risk ? (
              <div className="kpis">
                {Object.entries(risk).slice(0, 8).map(([k, v]) => (
                  <div className="card kpi" key={k}>
                    <div className="label">{k.replace(/_/g, " ")}</div>
                    <div className="value num" style={{ fontSize: 18 }}>{String(v)}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="dim">Enter a symbol to see Sharpe, drawdown, volatility, and more.</div>
            )}
          </div>
        )}
        {!loading && tab === "alerts" && (
          <div className="card-pad">
            <div className="row" style={{ marginBottom: 14, flexWrap: "wrap" }}>
              <input placeholder="Symbol" value={alertSym} style={{ width: 100 }}
                onChange={(e) => setAlertSym(e.target.value.toUpperCase())} />
              <select value={alertCond} onChange={(e) => setAlertCond(e.target.value)} style={{ width: 140 }}>
                <option value="price_above">Price above</option>
                <option value="price_below">Price below</option>
                <option value="score_above">Score above</option>
                <option value="score_below">Score below</option>
              </select>
              <input placeholder="Threshold" value={alertThr} style={{ width: 110 }}
                onChange={(e) => setAlertThr(e.target.value)} />
              <button className="primary" onClick={addAlert}>Add alert</button>
            </div>
            {triggered.length > 0 && (
              <div className="dim" style={{ marginBottom: 10, fontSize: 12.5 }}>
                Triggered now: {triggered.map((t) => String(t.symbol)).join(", ")}
              </div>
            )}
            {alerts.length === 0 ? (
              <div className="dim">No alerts yet — price/score watches stay in your prefs.</div>
            ) : (
              <table className="tbl">
                <thead><tr><th>Symbol</th><th>Condition</th><th>Threshold</th><th /></tr></thead>
                <tbody>
                  {alerts.map((a) => (
                    <tr key={String(a.id)}>
                      <td style={{ fontWeight: 650 }}>{String(a.symbol)}</td>
                      <td>{String(a.condition).replace(/_/g, " ")}</td>
                      <td className="num">{String(a.threshold)}</td>
                      <td><button className="ghost" onClick={() => removeAlert(String(a.id))}>Remove</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}
        {!loading && tab === "allocate" && (
          <div className="card-pad">
            <div className="dim" style={{ fontSize: 12.5, marginBottom: 10 }}>
              Risk-parity split — each name contributes similar risk.
            </div>
            <div className="row" style={{ marginBottom: 14 }}>
              <input value={allocInput} style={{ flex: 1, minWidth: 200 }}
                onChange={(e) => setAllocInput(e.target.value.toUpperCase())} />
              <button className="primary" onClick={allocate}>Optimize</button>
            </div>
            {alloc?.success === false && <div className="dim">{String(alloc.error)}</div>}
            {weights.length > 0 && (
              <div className="kpis">
                {weights.map(([k, v]) => (
                  <div className="card kpi" key={k}>
                    <div className="label">{k}</div>
                    <div className="value num" style={{ fontSize: 18 }}>
                      {(Number(v) <= 1 ? Number(v) * 100 : Number(v)).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
