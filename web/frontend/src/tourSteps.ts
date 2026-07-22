/** Spotlight step copy — top-to-bottom per App.tsx page id. */

import type { Step } from "react-joyride";
import type { TourPageId } from "./api";

export const TOUR_STEPS: Record<TourPageId, Step[]> = {
  dashboard: [
    {
      target: '[data-tour="dashboard-controls"]',
      title: "Load and navigate",
      content:
        "Search or load a symbol, add it to your watchlist, jump to Analyze, or generate a morning briefing.",
      disableBeacon: true,
    },
    {
      target: '[data-tour="dashboard-pulse"]',
      title: "Market pulse",
      content:
        "The top cards summarize price, range, market mood, world-news stress, and how many companies are raising profit forecasts.",
    },
    {
      target: '[data-tour="dashboard-chart"]',
      title: "Price chart",
      content:
        "Candle chart for this ticker. Dots flag busy volume days. "
        + "Optional overlays: past-strategy buy/sell markers, and an options levels guide "
        + "(orange key price line + a letter on the day's final bar — usually the close).",
    },
    {
      target: '[data-tour="dashboard-headlines"]',
      title: "Headlines",
      content:
        "Recent stories for this symbol. AI summaries (when keyed) are background only — not buy/sell advice.",
    },
    {
      target: '[data-tour="dashboard-briefing"]',
      title: "Morning briefing",
      content:
        "Generate a briefing to surface top buy and sell ideas and click any symbol to load it on the chart.",
    },
    {
      target: '[data-tour="dashboard-watchlist"]',
      title: "Watchlist",
      content:
        "Pinned symbols live here. Scores fill in after sparks so the board paints quickly.",
    },
  ],
  analyze: [
    {
      target: '[data-tour="analyze-controls"]',
      title: "Choose the question",
      content:
        "Enter a ticker, choose buy or bet-it-falls framing, then run the full Analyze workflow.",
      disableBeacon: true,
    },
    {
      target: '[data-tour="analyze-score"]',
      title: "Overall rating",
      content:
        "The big number is our overall read (0–10). The sentence under it says what we expect the stock to do and why — in plain English.",
    },
    {
      target: '[data-tour="analyze-chart"]',
      title: "Chart + markers",
      content:
        "Price candles for this symbol. Dots flag busy volume days; pattern marks show up after you open Patterns.",
    },
    {
      target: '[data-tour="analyze-tools"]',
      title: "Tool tabs",
      content:
        "Dig deeper when you want: what-if price ranges, Options, Filings, Labs. Fine to skip Options while learning.",
    },
    {
      target: '[data-tour="analyze-forecast-risk"]',
      title: "Forecast + risk",
      content:
        "Forecast = where models think price could go. Risk = how bumpy the ride can get. Neither is a trade order.",
    },
    {
      target: '[data-tour="analyze-news"]',
      title: "News context",
      content:
        "Recent headlines that may help explain the move. Context only — not buy/sell advice.",
    },
  ],
  scanner: [
    {
      target: '[data-tour="scanner-mode"]',
      title: "Scanner modes",
      content: "Switch between the quick single-stock scanner and pairs that move together.",
      disableBeacon: true,
    },
    {
      target: '[data-tour="scanner-setup"]',
      title: "Stock list setup",
      content:
        "Pick a stock list or paste your own symbols, then tune score thresholds and result count.",
    },
    {
      target: '[data-tour="scanner-filters"]',
      title: "Filters",
      content: "Toggle which filters must pass before a name shows up in results.",
    },
    {
      target: '[data-tour="scanner-run"]',
      title: "Run",
      content: "Run a scan or pairs screen with the current settings.",
    },
    {
      target: '[data-tour="scanner-results"]',
      title: "Results",
      content: "Click a row to open full details in Analyze for that symbol.",
    },
  ],
  portfolio: [
    {
      target: '[data-tour="portfolio-summary"]',
      title: "Account summary",
      content:
        "Top cards summarize total account value, cash, open profit/loss, and closed profit/loss.",
      disableBeacon: true,
    },
    {
      target: '[data-tour="portfolio-trade"]',
      title: "Buy / sell",
      content: "Practice trades go through this form. Cash updates with each fill.",
    },
    {
      target: '[data-tour="portfolio-tabs"]',
      title: "Portfolio sections",
      content:
        "Switch between positions, trades, cash/limits, risk, tracked ideas, alerts, and allocation.",
    },
    {
      target: '[data-tour="portfolio-content"]',
      title: "Active section",
      content:
        "The selected tab renders here: positions by default, plus cash tools, risk, alerts, tracked ideas, and allocation.",
    },
  ],
  backtest: [
    {
      target: '[data-tour="backtest-tabs"]',
      title: "Research modes",
      content:
        "Switch between simulating on past data, fine-tuning settings, forecast models, and options structure tests.",
      disableBeacon: true,
    },
    {
      target: '[data-tour="backtest-strategy"]',
      title: "Strategy / model",
      content: "Pick a strategy (or forecast model) and symbol, then run.",
    },
    {
      target: '[data-tour="backtest-folds"]',
      title: "Fold strip",
      content:
        "After a run, each chip is one unseen time window — does the advantage repeat across eras, or just one period?",
    },
    {
      target: '[data-tour="backtest-honesty"]',
      title: "Honesty check",
      content:
        "Test on dates the model never saw and adjust for how many ideas you tried — still research-only; not auto-wired into live defaults.",
    },
  ],
  chat: [
    {
      target: '[data-tour="chat-box"]',
      title: "Research thread",
      content:
        "Responses appear here with tool captions when Chat runs scans, scores, news, or risk checks.",
      disableBeacon: true,
    },
    {
      target: '[data-tour="chat-starters"]',
      title: "Starter chips",
      content: "Tap a prompt to fill the box, then Send.",
    },
    {
      target: '[data-tour="chat-input"]',
      title: "Ask anything",
      content: "Type in plain language — stock symbols, risk, news, briefings.",
    },
  ],
  settings: [
    {
      target: '[data-tour="settings-keys"]',
      title: "API keys",
      content: "Optional Anthropic / OpenAI / news / Twitter keys unlock AI and headline features.",
      disableBeacon: true,
    },
    {
      target: '[data-tour="settings-risk"]',
      title: "Risk profile",
      content:
        "Research preferences and how aggressive you say you are drive suggested position sizes, briefing defaults, and chat tone.",
    },
    {
      target: '[data-tour="settings-restart"]',
      title: "Restart tour",
      content:
        "Use Restart tour here anytime for a refresher — it clears all pages so each spotlight plays again on visit.",
    },
    {
      target: '[data-tour="settings-market"]',
      title: "Market signals",
      content:
        "Load slower macro-style signals manually. Results save to your profile and show on Dashboard.",
    },
  ],
};
