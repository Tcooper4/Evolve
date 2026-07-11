import { useRef, useState } from "react";
import { sendChat } from "./api";

interface Msg { role: "user" | "bot"; text: string; }

export default function Chat() {
  const [msgs, setMsgs] = useState<Msg[]>([
    { role: "bot", text: "Hey — I'm Evolve's assistant. Ask about your watchlist, a symbol, or the market. (Tool-calling chat lives in the Streamlit app for now; this is the fast lane.)" },
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
      setMsgs((m) => [...m, { role: "bot", text: r.success ? (r.reply ?? "") : `⚠ ${r.error}` }]);
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
          {msgs.map((m, i) => <div key={i} className={`msg ${m.role} fade-in`}>{m.text}</div>)}
          {busy && <div className="msg bot dim">thinking…</div>}
        </div>
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
