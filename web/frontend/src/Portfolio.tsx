import { useCallback, useEffect, useState } from "react";
import {
  adjustCash, cancelLimit, deleteAlert, deleteRec, getAccountRisk, getAlerts,
  getCashbook, getPortfolio, getPortfolioTrades, getRecs, getRisk, placeLimit,
  recordTrade, runAllocate, upsertAlert,
  type AccountRisk, type PortfolioSummary, type TrackedRec,
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
  const [tab, setTab] = useState<"positions" | "trades" | "risk" | "alerts" | "allocate" | "cash" | "tracked">("positions");
  const [acctRisk, setAcctRisk] = useState<AccountRisk | null>(null);
  const [acctRiskLoading, setAcctRiskLoading] = useState(false);
  const [recs, setRecs] = useState<TrackedRec[]>([]);
  const [recsLoading, setRecsLoading] = useState(false);
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
    if (!r.success) {
      setMsg(`⚠ ${r.error}`);
    } else {
      const rec = r.recommendation?.status;
      const base = side === "sell" && r.realized_pnl != null
        ? `Sold — realized ${r.realized_pnl >= 0 ? "+" : ""}$${r.realized_pnl}`
        : "Recorded.";
      setMsg(
        rec === "acted" ? `${base} Tracked idea → Bought.`
          : rec === "closed" ? `${base} Tracked idea → Closed.`
            : base,
      );
      setSymbol(""); setQty(""); setPrice("");
      refresh();
      if (tab === "tracked" || rec) void loadRecs();
    }
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

  async function loadAcctRisk() {
    setAcctRiskLoading(true);
    try { setAcctRisk(await getAccountRisk()); }
    catch { setAcctRisk(null); }
    finally { setAcctRiskLoading(false); }
  }

  async function loadRecs() {
    setRecsLoading(true);
    try { setRecs((await getRecs()).recommendations ?? []); }
    catch { setRecs([]); }
    finally { setRecsLoading(false); }
  }

  async function removeRec(id: string) {
    await deleteRec(id);
    await loadRecs();
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
          <div className="card kpi"><div className="label">Total equity</div>
            <div className="value num">${data.total_equity.toLocaleString()}</div>
            <div className="sub">cash + market value</div>
          </div>
          <div className="card kpi"><div className="label">Cash available</div>
            <div className="value num">${data.cash.toLocaleString()}</div>
            {!data.all_prices_live && <div className="sub">some prices unavailable — showing cost</div>}
          </div>
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
        <button className={tab === "risk" ? "active" : ""} onClick={() => { setTab("risk"); if (!acctRisk) void loadAcctRisk(); }}>Risk</button>
        <button className={tab === "tracked" ? "active" : ""} onClick={() => { setTab("tracked"); void loadRecs(); }}>Tracked ideas</button>
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
            <div className="rail-label" style={{ marginTop: 0 }}>Your account</div>
            {acctRiskLoading && <div className="skeleton" style={{ height: 120, marginBottom: 14 }} />}
            {!acctRiskLoading && acctRisk?.success && (
              <>
                <div className="kpis" style={{ marginBottom: 12 }}>
                  <div className="card kpi">
                    <div className="label">Sizing guide (half Kelly)</div>
                    <div className="value num" style={{ fontSize: 18 }}>
                      {acctRisk.kelly?.half_kelly_dollars != null
                        ? `$${acctRisk.kelly.half_kelly_dollars.toLocaleString()}`
                        : "—"}
                    </div>
                    <div className="sub">
                      {acctRisk.kelly?.half_kelly_dollars != null
                        ? `from your ${acctRisk.trade_stats?.closed_trades ?? 0} closed paper trades`
                        : (acctRisk.trade_stats?.closed_trades ?? 0) < 5
                          ? "close a few more paper trades and this fills in from your real stats"
                          : "needs both wins and losses to size from"}
                    </div>
                  </div>
                  {acctRisk.trade_stats?.win_rate != null && (
                    <div className="card kpi">
                      <div className="label">Your win rate</div>
                      <div className="value num" style={{ fontSize: 18 }}>
                        {(acctRisk.trade_stats.win_rate * 100).toFixed(0)}%
                      </div>
                      <div className="sub">closed trades only</div>
                    </div>
                  )}
                  {acctRisk.stress && Object.entries(acctRisk.stress).map(([k, v]) => (
                    <div className="card kpi" key={k}>
                      <div className="label">{k.replace("stress_", "").replace("sd", "σ")} down day</div>
                      <div className="value num down" style={{ fontSize: 18 }}>
                        {v.dollar_impact != null ? `$${Math.abs(v.dollar_impact).toLocaleString()}` : "—"}
                      </div>
                      <div className="sub">{v.daily_return_pct}% on this mix</div>
                    </div>
                  ))}
                </div>
                <div className="dim" style={{ fontSize: 12, marginBottom: 10 }}>
                  {acctRisk.kelly_note
                    || acctRisk.kelly?.note
                    || "Guide only — paper-trade Kelly is not a live broker size."}
                </div>
                {acctRisk.stress_note && (
                  <div className="dim" style={{ fontSize: 12, marginBottom: 10 }}>{acctRisk.stress_note}</div>
                )}
                {acctRisk.portfolio_metrics && (
                  <div className="kpis" style={{ marginBottom: 14 }}>
                    {Object.entries(acctRisk.portfolio_metrics).slice(0, 8).map(([k, v]) => (
                      <div className="card kpi" key={k}>
                        <div className="label">{k.replace(/_/g, " ")}</div>
                        <div className="value num" style={{ fontSize: 16 }}>{String(v)}</div>
                      </div>
                    ))}
                  </div>
                )}
                {acctRisk.positions === 0 && (
                  <div className="dim" style={{ marginBottom: 14 }}>
                    No positions yet — account-level risk fills in once you hold something.
                  </div>
                )}
              </>
            )}
            <div className="rail-label">Single symbol</div>
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
        {!loading && tab === "tracked" && (
          <div className="card-pad">
            <div className="dim" style={{ fontSize: 12.5, marginBottom: 12 }}>
              Track from Analyze (or chat "track NVDA"). Paper-buying an open
              idea marks it Bought; selling the whole position marks it Closed
              so you can see how the pick actually played out.
            </div>
            {recsLoading && <div className="skeleton" style={{ height: 120 }} />}
            {!recsLoading && recs.length === 0 && (
              <div className="empty" style={{ padding: "30px 0" }}>
                Nothing tracked yet. On Analyze, hit "Track idea" after scoring
                a symbol — it snapshots the price so the scoreboard is honest.
              </div>
            )}
            {!recsLoading && recs.length > 0 && (
              <table className="tbl">
                <thead><tr>
                  <th>Symbol</th><th>Status</th><th>Tracked</th><th>Score then</th>
                  <th>Price then</th><th>Now / exit</th><th>Since track</th><th />
                </tr></thead>
                <tbody>
                  {recs.map((r) => {
                    const st = r.status || "open";
                    const stLabel = st === "acted" ? "Bought" : st === "closed" ? "Closed" : "Open";
                    return (
                    <tr key={r.id} style={{ opacity: st === "closed" ? 0.72 : 1 }}>
                      <td style={{ fontWeight: 650 }}>{r.symbol}</td>
                      <td>
                        <span style={{
                          fontSize: 11.5, fontWeight: 650,
                          color: st === "acted" ? "var(--up)"
                            : st === "closed" ? "var(--text-2)" : "var(--accent)",
                        }}>{stLabel}</span>
                      </td>
                      <td className="dim">{r.created_at?.slice(0, 10)}</td>
                      <td className="num">{r.score != null ? r.score.toFixed(1) : "—"}</td>
                      <td className="num">{r.price_at_rec != null ? r.price_at_rec.toFixed(2) : "—"}</td>
                      <td className="num">{r.last_price != null ? r.last_price.toFixed(2) : "—"}</td>
                      <td className={`num ${r.change_pct != null ? (r.change_pct >= 0 ? "up" : "down") : ""}`}>
                        {r.change_pct != null ? `${r.change_pct >= 0 ? "+" : ""}${r.change_pct.toFixed(2)}%` : "—"}
                        {r.change_since_acted_pct != null && st !== "open" && (
                          <div className="dim" style={{ fontSize: 10.5 }}>
                            since buy {r.change_since_acted_pct >= 0 ? "+" : ""}
                            {r.change_since_acted_pct.toFixed(1)}%
                          </div>
                        )}
                      </td>
                      <td><button className="ghost" onClick={() => removeRec(r.id)}>Remove</button></td>
                    </tr>
                    );
                  })}
                </tbody>
              </table>
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
