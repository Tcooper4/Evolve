import { useState } from "react";
import { getScore, type ScoreResult } from "./api";

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

export default function Analyze() {
  const [input, setInput] = useState("SPY");
  const [res, setRes] = useState<ScoreResult | null>(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true);
    try { setRes(await getScore(input)); } catch { setRes(null); }
    finally { setLoading(false); }
  }

  const signals = res?.signals
    ? Object.entries(res.signals)
        .map(([k, v]) => {
          const num = typeof v === "number" ? v
            : typeof v === "object" && v && "score" in (v as object)
              ? Number((v as { score: unknown }).score) : NaN;
          return [k, num] as const;
        })
        .filter(([, v]) => !Number.isNaN(v))
        .sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div className="fade-in">
      <div className="greeting">
        Analyze <small>16-signal AI Score with per-signal breakdown</small>
      </div>
      <div className="topbar">
        <div className="search" style={{ maxWidth: 320 }}>
          <span className="icon">⌕</span>
          <input value={input}
            onChange={(e) => setInput(e.target.value.toUpperCase())}
            onKeyDown={(e) => e.key === "Enter" && run()} />
        </div>
        <button className="primary" onClick={run}>
          {loading ? "Scoring…" : "Score"}
        </button>
      </div>
      {loading && <div className="skeleton" style={{ height: 220 }} />}
      {!loading && res && res.score != null && (
        <div className="card card-pad ring-wrap fade-in">
          <ScoreRing score={res.score} />
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 8 }}>
              {res.symbol} <span className="grad-text">{res.grade ?? ""}</span>
            </div>
            {signals.slice(0, 10).map(([k, v]) => (
              <div className="signal-row" key={k}>
                <span style={{ color: "var(--text-2)" }}>{k.replace(/_/g, " ")}</span>
                <div className="signal-bar">
                  <div className="signal-fill" style={{ width: `${(v / 10) * 100}%` }} />
                </div>
                <span className="num" style={{ textAlign: "right" }}>{v.toFixed(1)}</span>
              </div>
            ))}
            {signals.length === 0 && (
              <div className="dim">Signal breakdown appears when data is available.</div>
            )}
          </div>
        </div>
      )}
      {!loading && res && res.score == null && (
        <div className="card card-pad empty">
          Couldn't score {res.symbol}{res.error ? ` — ${res.error}` : ""}. Live data required.
        </div>
      )}
    </div>
  );
}
