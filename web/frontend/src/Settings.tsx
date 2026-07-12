import { useEffect, useState } from "react";
import { getKeys, getPrefs, saveKeys, savePrefs } from "./api";

const SCORE_STYLES = [
  "Balanced (default)",
  "Momentum-heavy",
  "Technical-heavy",
  "Fundamental-heavy",
];

const DIRECTIONS = [
  "Bullish only (BUY signals)",
  "Bearish only (SHORT signals)",
  "Both",
];

export default function Settings() {
  const [saved, setSaved] = useState({
    anthropic: false, openai: false, news: false, reddit: false, twitter: false,
  });
  const [anthropic, setAnthropic] = useState("");
  const [openai, setOpenai] = useState("");
  const [news, setNews] = useState("");
  const [redditId, setRedditId] = useState("");
  const [redditSecret, setRedditSecret] = useState("");
  const [twitterBearer, setTwitterBearer] = useState("");
  const [scoringStyle, setScoringStyle] = useState(SCORE_STYLES[0]);
  const [briefUniverse, setBriefUniverse] = useState("sp100");
  const [minAi, setMinAi] = useState(6.0);
  const [direction, setDirection] = useState(DIRECTIONS[0]);
  const [msg, setMsg] = useState("");

  useEffect(() => {
    getKeys().then((k) => setSaved({
      anthropic: k.anthropic, openai: k.openai, news: k.news,
      reddit: !!k.reddit, twitter: !!k.twitter,
    })).catch(() => {});
    getPrefs().then((r) => {
      const p = r.prefs || {};
      if (typeof p.scoring_style === "string") setScoringStyle(p.scoring_style);
      if (typeof p.briefing_universe === "string") setBriefUniverse(p.briefing_universe);
      if (typeof p.min_ai_score === "number") setMinAi(p.min_ai_score);
      if (typeof p.opportunity_direction === "string") setDirection(p.opportunity_direction);
    }).catch(() => {});
  }, []);

  async function save() {
    const payload: Record<string, string> = {};
    if (anthropic) payload.anthropic = anthropic;
    if (openai) payload.openai = openai;
    if (news) payload.news = news;
    if (redditId) payload.reddit_client_id = redditId;
    if (redditSecret) payload.reddit_client_secret = redditSecret;
    if (twitterBearer) payload.twitter_bearer = twitterBearer;
    if (Object.keys(payload).length) await saveKeys(payload);
    await savePrefs({
      scoring_style: scoringStyle,
      briefing_universe: briefUniverse,
      min_ai_score: minAi,
      opportunity_direction: direction,
    });
    setAnthropic(""); setOpenai(""); setNews(""); setRedditId(""); setRedditSecret("");
    setTwitterBearer("");
    setSaved(await getKeys().then((k) => ({
      anthropic: k.anthropic, openai: k.openai, news: k.news,
      reddit: !!k.reddit, twitter: !!k.twitter,
    })));
    setMsg("Saved — encrypted keys + research prefs for your account only.");
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
      <div className="greeting">Settings <small>keys, scoring style, briefing defaults</small></div>

      <div className="card card-pad" style={{ maxWidth: 560, marginBottom: 16 }}>
        <div className="rail-label" style={{ marginTop: 0 }}>API keys</div>
        <Field label="Anthropic API key" val={anthropic} set={setAnthropic} has={saved.anthropic} />
        <Field label="OpenAI API key" val={openai} set={setOpenai} has={saved.openai} />
        <Field label="News API key" val={news} set={setNews} has={saved.news} />
        <Field label="Twitter/X bearer token" val={twitterBearer} set={setTwitterBearer} has={saved.twitter} />
        <p style={{ fontSize: 12, color: "var(--muted)", margin: "0 0 12px" }}>
          Bearer token powers breaking headlines and volume-chart news overlays. Without it,
          Evolve falls back to wire RSS (and a Walter Bloomberg RSS mirror when available).
        </p>
        <Field label="Reddit client ID" val={redditId} set={setRedditId} has={saved.reddit} />
        <Field label="Reddit client secret" val={redditSecret} set={setRedditSecret} has={saved.reddit} />
        <p style={{ fontSize: 12, color: "var(--muted)", margin: "0 0 4px" }}>
          Optional. AI Score sentiment is news-first; Reddit is a 30% blend when configured.
        </p>
      </div>

      <div className="card card-pad" style={{ maxWidth: 560, marginBottom: 16 }}>
        <div className="rail-label" style={{ marginTop: 0 }}>Research preferences</div>
        <div className="form-grid">
          <div className="field">
            <label>Scoring style</label>
            <select value={scoringStyle} onChange={(e) => setScoringStyle(e.target.value)}>
              {SCORE_STYLES.map((s) => <option key={s}>{s}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Briefing universe</label>
            <select value={briefUniverse} onChange={(e) => setBriefUniverse(e.target.value)}>
              {["default", "sp100", "sp500", "nasdaq100"].map((u) => <option key={u}>{u}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Min AI score</label>
            <input type="number" min={0} max={10} step={0.5} value={minAi}
              onChange={(e) => setMinAi(Number(e.target.value))} />
          </div>
          <div className="field">
            <label>Opportunity direction</label>
            <select value={direction} onChange={(e) => setDirection(e.target.value)}>
              {DIRECTIONS.map((d) => <option key={d}>{d}</option>)}
            </select>
          </div>
        </div>
      </div>

      <button className="primary" onClick={save}>Save settings</button>
      {msg && <div style={{ color: "var(--up)", marginTop: 10, fontSize: 13 }}>{msg}</div>}
    </div>
  );
}
