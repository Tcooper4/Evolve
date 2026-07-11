# Next Sessions — Roadmap and the Live-Data Mandate

Written 2026-07-11 at the close of the Fable sessions. Branch:
`codebase-audit-consolidated`. Everything below assumes that branch.

## SESSION A (next, fresh chat): LIVE-DATA SHAKEDOWN

Thomas has "All domains" network egress enabled; a fresh conversation
should have it (this one was frozen on the old permission snapshot —
verify first: `curl -sI https://query1.finance.yahoo.com/v8/finance/chart/AAPL`
must NOT return `x-deny-reason: host_not_allowed`).

Everything so far was verified on synthetic data. This session proves the
platform on real markets, in priority order:

1. **Data layer live**: `get_history/get_quote/get_news/batch_quotes` on
   real symbols (SPY, AAPL, an index ^GSPC, a future ES=F, crypto
   BTC-USD — exercises ticker_resolver's whole alias/probe table).
   Verify tz-naive indexes, column shapes, news url extraction against
   real yfinance payloads.
2. **The fetch-heavy files deferred from the depth pass** — these were
   deliberately NOT deep-verified offline because parsing synthetic
   fixtures proves little: `sec_edgar.py` (678L) against real EDGAR
   responses; `news_aggregator.py` parse paths against real feeds;
   `earnings_calendar.py`; `providers/yfinance_provider.py` +
   `alpha_vantage_provider.py` normalization; `congressional_trading.py`
   and `dark_pool.py` (real FINRA API shapes); `insider_flow.py` and
   `analyst_signals.py` against real yfinance fields (field names drift!).
3. **AI Score end-to-end** on 5–10 real symbols: all 16 signals populate,
   no dimension silently neutral (the sentiment-tokenization fix should
   now show non-zero news sentiment).
4. **Optimizer on real SPX/SPY history** with out-of-sample validation —
   the first genuinely meaningful optimization run.
5. **Options flow on a real chain** (SPY 0DTE if market hours): max pain,
   unusual volume, put/call sentiment against numbers Thomas can eyeball.
6. **Scanner on the real S&P 100**, then 500 (watch rate limits).
7. **All seven pages** via AppTest with live data flowing.
8. **MCP server tools** one by one with real data.
9. Fix whatever breaks — real payload shapes always surprise.

## SESSION B: remaining depth + wire-or-archive decisions

- `agents/llm/agent.py` (2,901L) line-by-line — the chat brain; its tool
  path is execution-verified but the body has only had targeted reads.
- `execution_risk_agent`, `data_quality_agent`, `performance_critic_agent`
  bodies; `agent_manager` loop mechanics.
- The kept-but-flagged five (TECHNICAL_DEBT.md): wire or archive
  streaming_pipeline, prompt_response_validator, risk_analyzer,
  prompt_processor, rsi_optimizer.
- Stale legacy test suites: fix or archive the pre-existing failures
  (test_strategy_optimizer, test_rsi_strategy signature drift;
  test_backtest_optimizer/test_hyperparameter_tuner import nonexistent
  modules).

## SESSIONS C+: React + FastAPI + websockets rewrite (4–6 sessions)

Sequenced last deliberately: the auth/identity layer it needs now exists.
Sketch: FastAPI API layer over the existing trading/* backend (reuse
per-user resolver + accounts db for JWT auth), React front end with a
real charting library (TradingView lightweight-charts or ECharts),
websocket price stream (the kept `data/streaming_pipeline.py` becomes
relevant), Streamlit retained during migration. Do NOT start this in a
session that can't finish a coherent vertical slice.

## Standing rules (unchanged)

- Execution-verified or it didn't happen; honest incompleteness over
  fake completeness; archive (reversible) never delete; personal mode
  stays byte-identical; every push through a Thomas-provided token that
  he revokes after.
