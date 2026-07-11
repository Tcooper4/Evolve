import { useEffect, useState } from "react";
import { getKeys, saveKeys } from "./api";

export default function Settings() {
  const [saved, setSaved] = useState({ anthropic: false, openai: false, news: false });
  const [anthropic, setAnthropic] = useState("");
  const [openai, setOpenai] = useState("");
  const [news, setNews] = useState("");
  const [msg, setMsg] = useState("");

  useEffect(() => { getKeys().then(setSaved).catch(() => {}); }, []);

  async function save() {
    const payload: Record<string, string> = {};
    if (anthropic) payload.anthropic = anthropic;
    if (openai) payload.openai = openai;
    if (news) payload.news = news;
    await saveKeys(payload);
    setAnthropic(""); setOpenai(""); setNews("");
    setSaved(await getKeys());
    setMsg("Saved — encrypted, and only your requests use them.");
    setTimeout(() => setMsg(""), 3500);
  }

  const Field = ({ label, val, set, has }: {
    label: string; val: string; set: (v: string) => void; has: boolean;
  }) => (
    <div className="field" style={{ marginBottom: 14 }}>
      <label>{label}{has && <span className="saved-dot" title="saved" />}</label>
      <input type="password" value={val} placeholder={has ? "•••••• (saved — enter to replace)" : "paste key"}
        onChange={(e) => set(e.target.value)} />
    </div>
  );

  return (
    <div className="fade-in">
      <div className="greeting">Settings <small>your keys, encrypted at rest, used only for your requests</small></div>
      <div className="card card-pad" style={{ maxWidth: 520 }}>
        <Field label="Anthropic API key" val={anthropic} set={setAnthropic} has={saved.anthropic} />
        <Field label="OpenAI API key" val={openai} set={setOpenai} has={saved.openai} />
        <Field label="News API key" val={news} set={setNews} has={saved.news} />
        <button className="primary" onClick={save}>Save keys</button>
        {msg && <div style={{ color: "var(--up)", marginTop: 10, fontSize: 13 }}>{msg}</div>}
      </div>
    </div>
  );
}
