import { useCallback, useEffect, useState } from "react";
import { getPortfolio, recordTrade, type PortfolioSummary } from "./api";

export default function Portfolio() {
  const [data, setData] = useState<PortfolioSummary | null>(null);
  const [symbol, setSymbol] = useState("");
  const [qty, setQty] = useState("");
  const [price, setPrice] = useState("");
  const [msg, setMsg] = useState("");
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try { setData(await getPortfolio()); } catch { setData(null); }
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

  const pnlCls = (v: number | null) =>
    v == null ? "" : v >= 0 ? "up" : "down";

  return (
    <div className="fade-in">
      <div className="greeting">
        Portfolio <small>paper only, always — practice with zero risk, tracked like the real thing</small>
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

      <div className="card">
        {loading && <div className="skeleton" style={{ height: 160, margin: 16 }} />}
        {!loading && data && data.positions.length > 0 && (
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
        {!loading && (!data || data.positions.length === 0) && (
          <div className="empty">
            No positions yet. Record a paper buy above — or ask the Chat
            "buy 10 shares of SPY on paper" and it'll do it for you.
          </div>
        )}
      </div>
    </div>
  );
}
