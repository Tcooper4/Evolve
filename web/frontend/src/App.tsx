import { useState } from "react";
import { hasToken, setToken } from "./api";
import Analyze from "./Analyze";
import Backtest from "./Backtest";
import Chat from "./Chat";
import Dashboard from "./Dashboard";
import Login from "./Login";
import MarketTicker from "./MarketTicker";
import Portfolio from "./Portfolio";
import Scanner from "./Scanner";
import Settings from "./Settings";

const PAGES = [
  { id: "dashboard", label: "Dashboard", ic: "◧" },
  { id: "analyze", label: "Analyze", ic: "◎" },
  { id: "scanner", label: "Scanner", ic: "⌕" },
  { id: "portfolio", label: "Portfolio", ic: "◫" },
  { id: "backtest", label: "Backtest", ic: "↺" },
  { id: "chat", label: "Chat", ic: "✦" },
  { id: "settings", label: "Settings", ic: "⚙" },
] as const;
type PageId = (typeof PAGES)[number]["id"];

export default function App() {
  const [name, setName] = useState<string | null>(
    hasToken() ? sessionStorage.getItem("evolve_name") : null,
  );
  const [page, setPage] = useState<PageId>("dashboard");
  const [analyzeSymbol, setAnalyzeSymbol] = useState("SPY");

  if (!name) {
    return (
      <Login onLogin={(n) => {
        sessionStorage.setItem("evolve_name", n);
        setName(n);
      }} />
    );
  }

  const logout = () => {
    setToken(null);
    sessionStorage.removeItem("evolve_name");
    setName(null);
  };

  const goAnalyze = (sym: string) => {
    const clean = (sym || "SPY").trim().toUpperCase();
    setAnalyzeSymbol(clean);
    setPage("analyze");
  };

  return (
    <div className="app">
      <aside className="rail">
        <div className="brand"><span className="dot" /> EVOLVE <small>terminal</small></div>
        <div className="user-chip">
          <span>{name}</span>
          <button className="ghost" onClick={logout}>Log out</button>
        </div>
        <nav className="nav">
          {PAGES.map((p) => (
            <button key={p.id} className={page === p.id ? "active" : ""}
              onClick={() => setPage(p.id)}>
              <span className="ic">{p.ic}</span> {p.label}
            </button>
          ))}
        </nav>
      </aside>
      <main className="main">
        <MarketTicker onSelect={goAnalyze} />
        {page === "dashboard" && (
          <Dashboard displayName={name} onAnalyze={goAnalyze} />
        )}
        {page === "analyze" && (
          <Analyze
            key={analyzeSymbol}
            initialSymbol={analyzeSymbol}
            onOpenBacktest={(sym, strategy) => {
              sessionStorage.setItem(
                "evolve_backtest",
                JSON.stringify({ symbol: sym, strategy }),
              );
              setPage("backtest");
            }}
          />
        )}
        {page === "scanner" && <Scanner onAnalyze={goAnalyze} />}
        {page === "portfolio" && <Portfolio />}
        {page === "backtest" && <Backtest />}
        {page === "chat" && <Chat />}
        {page === "settings" && <Settings />}
      </main>
    </div>
  );
}
