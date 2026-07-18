import { useEffect, useState } from "react";
import {
  getKeys, getMarketSignals, getPrefs, loadGpr, loadRevisionBreadth,
  resetTours, saveKeys, savePrefs, type GprSignal, type RevisionBreadth,
} from "./api";
import { CHART_TIMEZONES, cacheChartTimezone } from "./chartTime";
import PageTour from "./PageTour";
import AdminInvites from "./AdminInvites";
import AdminSharedKeys from "./AdminSharedKeys";

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

const RISK_TOLERANCES = [
  { id: "conservative", label: "Conservative" },
  { id: "moderate", label: "Moderate (default)" },
  { id: "aggressive", label: "Aggressive" },
] as const;

export default function Settings({ onToast }: { onToast?: (msg: string) => void }) {
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
  const [chartTimezone, setChartTimezone] = useState("America/New_York");
  const [riskTolerance, setRiskTolerance] = useState<string>("moderate");
  const [allowUndefinedRisk, setAllowUndefinedRisk] = useState(false);
  const [preferredDteMin, setPreferredDteMin] = useState(7);
  const [preferredDteMax, setPreferredDteMax] = useState(45);
  const [msg, setMsg] = useState("");
  const [gpr, setGpr] = useState<GprSignal | null>(null);
  const [rb, setRb] = useState<RevisionBreadth | null>(null);
  const [gprBusy, setGprBusy] = useState(false);
  const [rbBusy, setRbBusy] = useState(false);
  const [sigMsg, setSigMsg] = useState("");
  const [tourBusy, setTourBusy] = useState(false);

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
      if (typeof p.chart_timezone === "string") {
        setChartTimezone(p.chart_timezone);
        cacheChartTimezone(p.chart_timezone);
      }
      if (typeof p.risk_tolerance === "string") setRiskTolerance(p.risk_tolerance);
      if (typeof p.allow_undefined_risk === "boolean") setAllowUndefinedRisk(p.allow_undefined_risk);
      if (typeof p.preferred_dte_min === "number") setPreferredDteMin(p.preferred_dte_min);
      if (typeof p.preferred_dte_max === "number") setPreferredDteMax(p.preferred_dte_max);
    }).catch(() => {});
    getMarketSignals().then((s) => {
      setGpr(s.gpr);
      setRb(s.revision_breadth);
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
      chart_timezone: chartTimezone,
      risk_tolerance: riskTolerance,
      allow_undefined_risk: allowUndefinedRisk,
      preferred_dte_min: preferredDteMin,
      preferred_dte_max: preferredDteMax,
    });
    cacheChartTimezone(chartTimezone);
    setAnthropic(""); setOpenai(""); setNews(""); setRedditId(""); setRedditSecret("");
    setTwitterBearer("");
    setSaved(await getKeys().then((k) => ({
      anthropic: k.anthropic, openai: k.openai, news: k.news,
      reddit: !!k.reddit, twitter: !!k.twitter,
    })));
    setMsg("Saved — encrypted keys + stated preferences for your account only.");
    setTimeout(() => setMsg(""), 3500);
  }

  async function restartTour() {
    setTourBusy(true);
    try {
      const r = await resetTours();
      if (r.success) {
        onToast?.("Tour reset — it'll replay as you visit each page.");
      } else {
        onToast?.(r.error ?? "Could not reset tour.");
      }
    } catch (e) {
      onToast?.(e instanceof Error ? e.message : "Could not reset tour.");
    } finally {
      setTourBusy(false);
    }
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
      <PageTour pageId="settings" />
      <div className="greeting">Settings <small>keys, scoring style, briefing defaults</small></div>

      <AdminSharedKeys onToast={onToast} />
      <AdminInvites onToast={onToast} />

      <div className="card card-pad" data-tour="settings-keys" style={{ maxWidth: 560, marginBottom: 16 }}>
        <div className="rail-label" style={{ marginTop: 0 }}>API keys</div>
        <Field label="Anthropic API key" val={anthropic} set={setAnthropic} has={saved.anthropic} />
        <Field label="OpenAI API key" val={openai} set={setOpenai} has={saved.openai} />
        <Field label="News API key" val={news} set={setNews} has={saved.news} />
        <Field label="Twitter/X bearer token" val={twitterBearer} set={setTwitterBearer} has={saved.twitter} />
        <p style={{ fontSize: 12, color: "var(--text-2)", margin: "0 0 12px" }}>
          Bearer token powers breaking headlines and volume-chart news overlays. Without it,
          Evolve falls back to wire RSS (and a Walter Bloomberg RSS mirror when available).
        </p>
        <Field label="Reddit client ID" val={redditId} set={setRedditId} has={saved.reddit} />
        <Field label="Reddit client secret" val={redditSecret} set={setRedditSecret} has={saved.reddit} />
        <p style={{ fontSize: 12, color: "var(--text-2)", margin: "0 0 4px" }}>
          Optional. AI Score sentiment is news-first; Reddit is a 30% blend when configured.
        </p>
      </div>

      <div className="card card-pad" data-tour="settings-risk" style={{ maxWidth: 560, marginBottom: 16 }}>
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
          <div className="field">
            <label>Chart timezone</label>
            <select value={chartTimezone} onChange={(e) => setChartTimezone(e.target.value)}>
              {CHART_TIMEZONES.map((z) => (
                <option key={z.id} value={z.id}>{z.label}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Risk tolerance (stated)</label>
            <select value={riskTolerance} onChange={(e) => setRiskTolerance(e.target.value)}>
              {RISK_TOLERANCES.map((r) => (
                <option key={r.id} value={r.id}>{r.label}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Comfortable with undefined-risk / naked positions?</label>
            <select
              value={allowUndefinedRisk ? "yes" : "no"}
              onChange={(e) => setAllowUndefinedRisk(e.target.value === "yes")}
            >
              <option value="no">No — flag &amp; deprioritize</option>
              <option value="yes">Yes — still list, less flagging</option>
            </select>
          </div>
          <div className="field">
            <label>Preferred DTE min</label>
            <input type="number" min={0} max={365} step={1} value={preferredDteMin}
              onChange={(e) => setPreferredDteMin(Number(e.target.value))} />
          </div>
          <div className="field">
            <label>Preferred DTE max</label>
            <input type="number" min={0} max={365} step={1} value={preferredDteMax}
              onChange={(e) => setPreferredDteMax(Number(e.target.value))} />
          </div>
        </div>
        <p style={{ fontSize: 12, color: "var(--text-2)", margin: "10px 0 0" }}>
          Risk tolerance is what you set here — Evolve never infers it from clicks or
          watchlist activity. Default is moderate (keeps prior Kelly guidance).
          Conservative forces quarter-Kelly as a profile reason and deprioritizes
          undefined-risk options ideas without hiding them.
        </p>
        <p style={{ fontSize: 12, color: "var(--text-2)", margin: "10px 0 0" }}>
          Intraday chart labels and hover times use this zone. Candle data is stored in UTC.
          Default is US Eastern (NYSE session).
        </p>        <button className="primary" onClick={save} style={{ marginTop: 14 }}>Save settings</button>
        {msg && <div style={{ color: "var(--up)", marginTop: 10, fontSize: 13 }}>{msg}</div>}
      </div>

      <div className="card card-pad" data-tour="settings-restart" style={{ maxWidth: 560, marginBottom: 16 }}>
        <div className="rail-label" style={{ marginTop: 0 }}>Spotlight tour</div>
        <p style={{ fontSize: 12.5, color: "var(--text-2)", margin: "0 0 12px", lineHeight: 1.45 }}>
          First-visit tips on each page. Restart clears seen state so spotlights replay
          when you visit Dashboard, Analyze, and the rest again.
        </p>
        <button className="primary" disabled={tourBusy} onClick={() => void restartTour()}>
          {tourBusy ? "Resetting…" : "Restart tour"}
        </button>
      </div>

      <div className="card card-pad" data-tour="settings-market" style={{ maxWidth: 560, marginBottom: 16 }}>
        <div className="rail-label" style={{ marginTop: 0 }}>Market signals</div>
        <p style={{ fontSize: 12.5, color: "var(--text-2)", margin: "0 0 12px", lineHeight: 1.45 }}>
          Manual load only — results save to your profile and show on the Dashboard pulse.
          GPR is the Caldara &amp; Iacoviello academic index (disk-cached ~30 days).
          EPS breadth samples S&amp;P names and can take a few minutes.
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div>
            <div className="dim" style={{ fontSize: 12.5, marginBottom: 6 }}>
              {gpr
                ? `GPR: ${gpr.current} (${gpr.level ?? "—"}) · ${gpr.trend ?? ""}`
                : "GPR: not loaded"}
            </div>
            <button className="primary" disabled={gprBusy} onClick={async () => {
              setGprBusy(true); setSigMsg("");
              try {
                const r = await loadGpr();
                if (r.success && r.gpr) { setGpr(r.gpr); setSigMsg("GPR saved to your profile."); }
                else setSigMsg(r.error ?? "GPR load failed.");
              } catch (e) {
                setSigMsg(e instanceof Error ? e.message : "GPR failed");
              } finally { setGprBusy(false); setTimeout(() => setSigMsg(""), 4000); }
            }}>
              {gprBusy ? "Loading GPR…" : "Load Geopolitical Risk Index"}
            </button>
          </div>
          <div>
            <div className="dim" style={{ fontSize: 12.5, marginBottom: 6 }}>
              {rb
                ? `EPS breadth: ${rb.pct_up.toFixed(0)}% ↑ / ${rb.pct_down.toFixed(0)}% ↓ (${rb.sample_size ?? "?"} stocks) · ${rb.signal ?? ""}`
                : "EPS revision breadth: not loaded"}
            </div>
            <button disabled={rbBusy} onClick={async () => {
              setRbBusy(true); setSigMsg("");
              try {
                const r = await loadRevisionBreadth(150);
                if (r.success && r.revision_breadth) {
                  setRb(r.revision_breadth);
                  setSigMsg("EPS breadth saved to your profile.");
                } else setSigMsg(r.error ?? "Breadth compute failed.");
              } catch (e) {
                setSigMsg(e instanceof Error ? e.message : "Breadth failed");
              } finally { setRbBusy(false); setTimeout(() => setSigMsg(""), 5000); }
            }}>
              {rbBusy ? "Computing breadth (may take minutes)…" : "Compute EPS revision breadth"}
            </button>
          </div>
        </div>
        {sigMsg && <div style={{ marginTop: 10, fontSize: 13, color: "var(--text-2)" }}>{sigMsg}</div>}
      </div>
    </div>
  );
}
