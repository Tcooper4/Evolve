/** Spotlight step copy — concise, 4–6 steps per App.tsx page id. */

import type { Step } from "react-joyride";
import type { TourPageId } from "./api";

export const TOUR_STEPS: Record<TourPageId, Step[]> = {
  dashboard: [
    {
      target: '[data-tour="dashboard-watchlist"]',
      title: "Watchlist",
      content:
        "Pin symbols you follow. Scores fill in after sparks so the board paints quickly.",
      disableBeacon: true,
    },
    {
      target: '[data-tour="dashboard-chart"]',
      title: "Price chart",
      content:
        "Main candle chart for the loaded ticker. News marks (N / n / E) appear when volume spikes line up with headlines.",
    },
    {
      target: '[data-tour="dashboard-headlines"]',
      title: "Headlines",
      content:
        "Recent stories for this symbol. LLM blurbs (when keyed) are context only — not a buy/sell call.",
    },
  ],
  analyze: [
    {
      target: '[data-tour="analyze-score"]',
      title: "AI Score",
      content:
        "The score ring summarizes technical, momentum, sentiment, and fundamental dimensions on a 0–10 scale.",
      disableBeacon: true,
    },
    {
      target: '[data-tour="analyze-chart"]',
      title: "Chart + markers",
      content:
        "Candles with volume/news marks and optional pattern markers once you open Patterns.",
    },
    {
      target: '[data-tour="analyze-tools"]',
      title: "Tool tabs",
      content:
        "Monte Carlo, Options, Filings, and Labs dig deeper. GEX and skew under Options are advanced/optional — fine to skip when you’re learning.",
    },
  ],
  scanner: [
    {
      target: '[data-tour="scanner-run"]',
      title: "Scan",
      content: "Run a universe scan (or pairs screen) with the current filters.",
      disableBeacon: true,
    },
    {
      target: '[data-tour="scanner-filters"]',
      title: "Filters",
      content: "Toggle which screens must pass before a name shows up in results.",
    },
    {
      target: '[data-tour="scanner-results"]',
      title: "Results",
      content: "Click a row to drill into Analyze for that symbol.",
    },
  ],
  portfolio: [
    {
      target: '[data-tour="portfolio-tabs"]',
      title: "Tabs",
      content:
        "Positions, trades, cash/limits, risk, tracked ideas, alerts, and allocate — switch here.",
      disableBeacon: true,
    },
    {
      target: '[data-tour="portfolio-trade"]',
      title: "Buy / sell",
      content: "Paper trades go through this form. Cash updates with each fill.",
    },
    {
      target: '[data-tour="portfolio-risk"]',
      title: "Kelly & stress",
      content:
        "On the Risk tab: sizing guide from your closed paper stats, plus stress cards for rough down-day impact.",
    },
  ],
  backtest: [
    {
      target: '[data-tour="backtest-strategy"]',
      title: "Strategy / model",
      content: "Pick a strategy (or forecast model) and symbol, then run.",
      disableBeacon: true,
    },
    {
      target: '[data-tour="backtest-folds"]',
      title: "Fold strip",
      content:
        "After a run, each chip is one out-of-sample window — does the edge hold across time, or just one era?",
    },
    {
      target: '[data-tour="backtest-honesty"]',
      title: "OOS & DSR",
      content:
        "OOS means held-out time. DSR adjusts Sharpe for how many trials you ran — still research-only; not auto-wired into live defaults.",
    },
  ],
  chat: [
    {
      target: '[data-tour="chat-input"]',
      title: "Ask anything",
      content: "Type in plain language — tickers, risk, news, briefings.",
      disableBeacon: true,
    },
    {
      target: '[data-tour="chat-starters"]',
      title: "Starter chips",
      content: "Tap a prompt to fill the box, then Send.",
    },
    {
      target: '[data-tour="chat-box"]',
      title: "Real research",
      content:
        "This isn’t just chat — it can run multi-step tools (scan, score, news, risk) and show what it used.",
    },
  ],
  settings: [
    {
      target: '[data-tour="settings-risk"]',
      title: "Risk profile",
      content:
        "Stated preference only — never inferred from clicks. Drives Kelly framing and chat tone.",
      disableBeacon: true,
    },
    {
      target: '[data-tour="settings-keys"]',
      title: "API keys",
      content: "Optional Anthropic / OpenAI / news / Twitter keys for LLM and headline features.",
    },
    {
      target: '[data-tour="settings-restart"]',
      title: "Restart tour",
      content:
        "Use Restart tour here anytime for a refresher — it clears all pages so each spotlight plays again on visit.",
    },
  ],
};
