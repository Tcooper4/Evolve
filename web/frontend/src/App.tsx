import { useEffect, useState } from "react";
import { hasToken, notificationsSocket, setToken } from "./api";
import Analyze from "./Analyze";
import Backtest from "./Backtest";
import Chat from "./Chat";
import Dashboard from "./Dashboard";
import Login from "./Login";
import Signup from "./Signup";
import MarketTicker from "./MarketTicker";
import Portfolio from "./Portfolio";
import Scanner from "./Scanner";
import ReloadPrompt from "./ReloadPrompt";
import InsecureContextNotice from "./InsecureContextNotice";
import Settings from "./Settings";

const PAGES = [
  { id: "dashboard", label: "Dashboard", ic: "◧" },
  { id: "analyze", label: "Analyze", ic: "◎" },
  { id: "scanner", label: "Scanner", ic: "⌕" },
  { id: "portfolio", label: "Portfolio", ic: "◫" },
  { id: "backtest", label: "Simulate", ic: "↺" },
  { id: "chat", label: "Chat", ic: "✦" },
  { id: "settings", label: "Settings", ic: "⚙" },
] as const;
type PageId = (typeof PAGES)[number]["id"];

export default function App() {
  const [name, setName] = useState<string | null>(
    hasToken() ? sessionStorage.getItem("evolve_name") : null,
  );
  const [authMode, setAuthMode] = useState<"login" | "signup">("login");
  const [page, setPage] = useState<PageId>("dashboard");
  const [analyzeSymbol, setAnalyzeSymbol] = useState("SPY");
  const [toast, setToast] = useState<string | null>(null);

  const finishAuth = (n: string) => {
    sessionStorage.setItem("evolve_name", n);
    setName(n);
    setAuthMode("login");
  };

  useEffect(() => {
    if (!name) return;
    const ws = notificationsSocket((n) => {
      const msg = String(n.message || n.type || "").trim();
      if (!msg) return;
      setToast(msg);
      window.setTimeout(() => setToast(null), 8000);
    });
    return () => { try { ws.close(); } catch { /* skip */ } };
  }, [name]);

  if (!name) {
    return (
      <>
        {authMode === "login" ? (
          <Login
            onLogin={finishAuth}
            onCreateAccount={() => setAuthMode("signup")}
          />
        ) : (
          <Signup
            onLogin={finishAuth}
            onBack={() => setAuthMode("login")}
          />
        )}
        <ReloadPrompt />
        <InsecureContextNotice />
      </>
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
        <div className="brand"><span className="dot" /> EVOLVE <small>research hub</small></div>
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
        {toast && (
          <div
            role="status"
            style={{
              position: "sticky",
              top: 0,
              zIndex: 40,
              margin: "0 0 8px",
              padding: "10px 14px",
              background: "rgba(18, 28, 42, 0.95)",
              borderBottom: "1px solid #2a3a4f",
              color: "#e8eef7",
              fontSize: 13,
            }}
          >
            {toast}
          </div>
        )}
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
        {page === "settings" && (
          <Settings onToast={(msg) => {
            setToast(msg);
            window.setTimeout(() => setToast(null), 8000);
          }} />
        )}
      </main>
      <ReloadPrompt />
      <InsecureContextNotice />
    </div>
  );
}
