---
name: optimizer-results-review
description: How to judge strategy-optimization output before trusting it
triggers: optimize, optimizer, optimization, best params, parameters, overfit, walk-forward, out-of-sample, in-sample, tuned
---
# Judging optimizer output

Apply whenever presenting or discussing results from Evolve's strategy
optimizer (grid/genetic/PSO/Bayesian) or the self-tuning path.

## The order of trust
1. **Out-of-sample (test window) metrics** — the realistic expectation.
2. In-sample optimized metrics — an upper bound, systematically inflated
   because the search selects for whatever fit the sample, signal or noise.
3. Convergence behavior — supporting evidence, never the headline.

Never present in-sample numbers without the out-of-sample comparison when
validation was run. If validation was NOT run, say the results are
in-sample only and recommend re-running with the holdout enabled.

## Overfit signatures — call these out explicitly
- Large train→test degradation (e.g. Sharpe 2.0 → 0.3): partial overfit at
  best. The test number is the estimate; the train number is marketing.
- Optimized parameters at the edge of their allowed range: the search hit
  the wall — the "optimum" is an artifact of the bounds, not the market.
- Best parameters that change drastically between nearby windows or
  symbols: the strategy is fitting noise, not structure.
- Tiny trade counts (few signal events): metrics on a handful of trades
  are statistically meaningless — say so.

## Plausibility checks before accepting parameters
- Do the parameters make trading sense? (An RSI period of 5 with bands at
  49/51 is a coin-flipper, whatever its backtest says.)
- Are results net of transaction costs? Evolve's objective charges costs —
  confirm the bps setting matches how the user actually trades.
- Would the defaults have been fine? "Defaults win" is a legitimate,
  publishable result — say it without apology.

## When parameters pass review
- Recommend paper-trading or a small sizing tier first; parameters earn
  size with live evidence, not backtests.
