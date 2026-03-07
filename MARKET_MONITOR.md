# Live Market Events Monitor

The **Live Market Events Monitor** runs on the **Home** page (`pages/0_Home.py`) and shows significant volume/price spikes from a configurable watchlist, with related news and an optional AI explanation.

## Overview

- **Background polling**: Every 60 seconds the app scans the watchlist for significant 5m candles (volume ≥ threshold × 20-period average, price move ≥ min %).
- **Featured event**: The single highest-impact spike is shown as the "featured event" in the main (left) column, with a candlestick + volume chart and related news.
- **Event feed**: The right column lists recent events (newest first); clicking an item selects it and shows its chart and news in the main area.
- **Impact score**: `volume_ratio × price_move_pct × (1 / market_cap_rank)` so larger names and bigger moves rank higher.

## Files

| File | Purpose |
|------|--------|
| `trading/analysis/market_monitor.py` | `scan_watchlist()` — yfinance 5m download, rolling volume, spike detection, impact scoring. |
| `trading/analysis/event_news_fetcher.py` | `fetch_news_around_event()` — NewsAPI fetch for symbol/company around event time. |
| `trading/analysis/news_ranker.py` | `rank_news_by_relevance()` — scores/ranks articles by symbol/company/direction. |
| `trading/analysis/chart_builder.py` | `build_event_chart()` — Plotly 2-row (candlestick + volume), yellow spike line, plotly_dark. |
| `pages/0_Home.py` | Session state, sidebar settings, scan loop, two-column UI, auto-rerun. |

## Session State (Home page)

- `last_scan_time` — timestamp of last scan (for 60s countdown and next scan).
- `event_feed` — list of spike dicts (newest first), max 20.
- `featured_event` — current highest-impact spike (chart + news).
- `selected_event` — event chosen from the feed (overrides featured in main column).

## Sidebar Settings

- **Volume spike threshold** (1.5–5.0, default 3.0) — min volume multiple over 20-period average.
- **Min price move %** (0.5–5.0, default 2.0) — min absolute % move (open → close).
- **Watchlist** — comma-separated symbols; used as the scan universe.

## Spike Dict

Each event in the feed/featured has:

- `symbol`, `timestamp`, `direction` ("up"/"down"), `price_move_pct`, `volume_ratio`, `close`, `impact_score`
- `df` — full OHLCV DataFrame for the symbol (for chart).
- After news fetch: `news` (list of ranked articles), `top_headline`, `top_url`.

## News and Explain

- News is fetched (NewsAPI) only for the **top** spike from each scan to limit API usage.
- Articles are ranked by relevance (symbol/company in title, direction-related words); each gets `relevance_score` and `relevance_reason`.
- **"Explain this move"** uses `agents.llm.active_llm_calls.call_active_llm_simple()` to summarize which headline likely caused the move.

## Auto-Refresh

At the bottom of the Home page:

- If `should_scan` (≥ 60s since last scan): brief `time.sleep(0.1)` then `st.rerun()` to run the next scan.
- Else: `time.sleep(min(seconds_left, 30))` then `st.rerun()` to update the countdown and eventually trigger the next scan.

## Compile Check

```bash
py -3.10 -m py_compile trading/analysis/market_monitor.py trading/analysis/chart_builder.py pages/0_Home.py
```

## Dependencies

- `yfinance`, `pandas`, `numpy` — market data and spike logic.
- `plotly` — chart builder.
- `requests` — NewsAPI in event_news_fetcher.
- `NEWS_API_KEY` or `NEWSAPI_KEY` — optional; news is skipped if unset.
