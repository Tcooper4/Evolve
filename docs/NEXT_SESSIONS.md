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

- `agents/llm/agent.py` — COMPLETE (Session B, 2026-07). Full-depth
  pass: routing fallback fixed (substring intent matching where 'test'
  fired inside 'latest'; generic intents shadowing specific ones;
  domain words RSI/BANDS/LAST/MODEL/PRICE returned as tickers); few-shot
  cluster fixed (numpy-truthiness crash that fired the moment a real
  encoder existed, dot-product mislabeled as cosine, single malformed
  stored example silently disabling ALL retrieval/saving); optimization
  handler truth-fixed (hardcoded fake baseline metrics now replaced by a
  real measured baseline with honest disclosure offline); all seven
  handlers smoke-verified to fail honestly offline. Token accounting
  verified correct.
- CRITIC AGENTS — COMPLETE (Session B). DataQualityAgent was
  unbuildable for THREE independent reasons (abstract-method drift,
  hard AlphaVantage key requirement, registry referencing detectors
  that never existed) and, once built, detected NOTHING: every detector
  read lowercase column names against the platform's Title-case data.
  Now detects planted anomalies and scores dirty<clean, verified.
  ExecutionRiskAgent: abstract drift + NotImplementedError _setup ->
  implemented with the exact state keys the checks read; scenario
  battery verified (oversize rejection, drawdown halt, cooling period).
  PerformanceCriticAgent: column-case fix, information-ratio math
  corrected (tracking-error denominator, annualized), config.get crash
  fixed; metrics hand-verified.
- AGENT MANAGER — COMPLETE (Session B). Was unconstructible (hard redis
  import via ExecutionAgent chain + module-level agent imports defeating
  its own per-agent isolation); worse, success bookkeeping read
  result.sharpe_ratio (nonexistent on AgentResult), converting EVERY
  successful agent run into a failure and burning all retries - the
  retry loop had never completed a run. Fixed; verified fail-fail-recover
  end-to-end with metrics. Note: trading/agents/execution/ imports a
  missing execution_providers module - ExecutionAgent is independently
  broken and now degrades gracefully; revive or archive in a future
  session.
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
