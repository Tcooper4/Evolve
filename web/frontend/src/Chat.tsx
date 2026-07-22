import { useEffect, useRef, useState } from "react";
import { sendChat } from "./api";
import PageTour from "./PageTour";

interface Msg { role: "user" | "bot"; text: string; tools?: string[]; }

const STARTERS = [
  "What stocks should I consider buying today?",
  "Is now a good time to buy?",
  "Explain what's happening in the market like I'm new",
  "Check the risk on my watchlist",
  "Summarize today's top stock ideas",
];

export default function Chat() {
  const [msgs, setMsgs] = useState<Msg[]>([
    { role: "bot", text: "Hey — I'm Evolve's assistant. Ask me anything about the market in plain language: I'll do the research (scans, scores, news, risk) and explain it clearly." },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setMsgs((m) => [...m, { role: "user", text }]);
    setBusy(true);
    try {
      const r = await sendChat(text);
      setMsgs((m) => [...m, {
        role: "bot",
        text: r.success ? (r.reply ?? "") : `⚠ ${r.error}`,
        tools: r.tool_captions,
      }]);
    } catch (e) {
      setMsgs((m) => [...m, { role: "bot", text: `⚠ ${e instanceof Error ? e.message : "failed"}` }]);
    } finally {
      setBusy(false);
      setTimeout(() => box.current?.scrollTo({ top: 1e9, behavior: "smooth" }), 50);
    }
  }

  // Keep the composer in view when the mobile keyboard opens (visualViewport).
  useEffect(() => {
    const vv = window.visualViewport;
    if (!vv) return;
    const onResize = () => {
      const el = inputRef.current;
      if (!el || document.activeElement !== el) return;
      el.scrollIntoView({ block: "end", behavior: "smooth" });
    };
    vv.addEventListener("resize", onResize);
    return () => vv.removeEventListener("resize", onResize);
  }, []);

  return (
    <div className="fade-in">
      <PageTour pageId="chat" />
      <div className="greeting">Chat <small>ask in plain English — uses your saved settings and optional AI key</small></div>

      <div className="card chat-shell">
        <div className="chat-box" data-tour="chat-box" ref={box}>
          {msgs.map((m, i) => (
            <div key={i} className={`msg ${m.role} fade-in`}>
              {m.tools && m.tools.length > 0 && (
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 8 }}>
                  {m.tools.map((t, j) => <span key={j} className="pill mid">{t}</span>)}
                </div>
              )}
              {m.text}
            </div>
          ))}
          {busy && <div className="msg bot dim">thinking…</div>}
        </div>
        <div className="chat-starters" data-tour="chat-starters">
          {STARTERS.map((q) => (
            <button
              key={q}
              type="button"
              className="ghost chat-starter"
              onClick={() => { setInput(q); inputRef.current?.focus(); }}
            >
              {q}
            </button>
          ))}
        </div>
        <div className="chat-input" data-tour="chat-input">
          <input
            ref={inputRef}
            value={input}
            placeholder="Ask about a stock symbol, risk, news…"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            onFocus={() => {
              // Delay so the keyboard animation can finish before scrolling.
              window.setTimeout(() => {
                inputRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
              }, 300);
            }}
          />
          <button className="primary" onClick={send} disabled={busy}>Send</button>
        </div>
      </div>
    </div>
  );
}
