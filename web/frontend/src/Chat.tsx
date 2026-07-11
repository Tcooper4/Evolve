import { useRef, useState } from "react";
import { sendChat } from "./api";

interface Msg { role: "user" | "bot"; text: string; tools?: string[]; }

export default function Chat() {
  const [msgs, setMsgs] = useState<Msg[]>([
    { role: "bot", text: "Hey — I'm Evolve's assistant. Ask me anything about the market in plain language: I'll do the research (scans, scores, news, risk) and explain it clearly. No finance background needed." },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const box = useRef<HTMLDivElement>(null);

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

  return (
    <div className="fade-in">
      <div className="greeting">Chat <small>your assistant, your memory, your API key</small></div>
      <div className="card">
        <div className="chat-box" ref={box}>
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
        {msgs.length <= 1 && (
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", padding: "0 18px 12px" }}>
            {[
              "What stocks should I consider buying today?",
              "Is now a good time to buy?",
              "Explain what's happening in the market like I'm new",
              "Check the risk on my watchlist",
            ].map((q) => (
              <button key={q} className="ghost" style={{ fontSize: 12.5, border: "1px solid var(--border)" }}
                onClick={() => { setInput(q); }}>
                {q}
              </button>
            ))}
          </div>
        )}
        <div className="chat-input">
          <input value={input} placeholder="Ask anything…"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()} />
          <button className="primary" onClick={send}>Send</button>
        </div>
      </div>
    </div>
  );
}
