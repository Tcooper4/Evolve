import { useState } from "react";
import { runScan } from "./api";

const FILTERS = ["oversold_rsi", "momentum", "high_volume", "breakout", "unusual_options"];

export default function Scanner() {
  const [active, setActive] = useState<string[]>(["momentum"]);
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function run() {
    setLoading(true); setError("");
    try {
      const r = await runScan(active);
      if (r.success && r.results) setRows(r.results);
      else { setRows([]); setError(r.error ?? "Scan needs live market data."); }
    } catch (e) {
      setError(e instanceof Error ? e.message : "scan failed");
    } finally { setLoading(false); }
  }

  const cols = rows.length ? Object.keys(rows[0]).slice(0, 6) : [];

  return (
    <div className="fade-in">
      <div className="greeting">Scanner <small>screen the market by technical filters and quick score</small></div>
      <div className="topbar" style={{ flexWrap: "wrap" }}>
        <div className="seg">
          {FILTERS.map((f) => (
            <button key={f} className={active.includes(f) ? "active" : ""}
              onClick={() => setActive((a) => a.includes(f) ? a.filter((x) => x !== f) : [...a, f])}>
              {f.replace(/_/g, " ")}
            </button>
          ))}
        </div>
        <button className="primary" onClick={run}>{loading ? "Scanning…" : "Scan"}</button>
      </div>
      <div className="card">
        {loading && <div className="skeleton" style={{ height: 260, margin: 16 }} />}
        {!loading && rows.length > 0 && (
          <table className="tbl">
            <thead><tr>{cols.map((c) => <th key={c}>{c.replace(/_/g, " ")}</th>)}</tr></thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>{cols.map((c) => {
                  const v = r[c];
                  const n = typeof v === "number";
                  return <td key={c} className={n ? "num" : ""}>{n ? (v as number).toFixed(2) : String(v ?? "—")}</td>;
                })}</tr>
              ))}
            </tbody>
          </table>
        )}
        {!loading && rows.length === 0 && (
          <div className="empty">{error || "Pick filters and run a scan."}</div>
        )}
      </div>
    </div>
  );
}
