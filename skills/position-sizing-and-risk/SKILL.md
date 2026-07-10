---
name: position-sizing-and-risk
description: Sizing and risk discipline for suggestions surfaced to the user
triggers: position size, sizing, how much, kelly, risk, stop, drawdown, allocate, allocation, leverage, 0dte
---
# Position sizing & risk discipline

Apply whenever discussing how large a position could be or how to manage
risk on an idea. Evolve informs decisions; the user decides.

## Core rules
- Lead with the loss, not the gain: state the max plausible loss in
  dollars and as % of account before discussing upside.
- Full Kelly is a ceiling, not a target — practitioners run half-Kelly or
  less because edge estimates are noisy and drawdown pain is asymmetric.
  When Evolve's Kelly tool returns f, present f/2 as the reference point.
- Volatility scales size inversely: in a high-vol regime (check the regime
  tool / VIX context), the same conviction warrants a smaller position.
- Correlated positions are one position. If the user already holds names
  that move together, treat new additions as adding to that cluster.
- A stop level must be structural (beyond support/resistance, or an
  ATR-multiple), never a round number chosen for comfort. If a proper stop
  implies risk beyond the per-trade budget, the position is too big — cut
  size, don't widen the stop.

## Defined-risk options (incl. SPX 0DTE)
- For spreads, max loss is the width minus credit (or debit paid) — state
  it explicitly per contract and in total.
- 0DTE: gamma risk accelerates into the close and liquidity can gap around
  scheduled events (CPI, FOMC, NFP). Flag any same-day macro events before
  discussing entries, and treat "letting it ride into the last hour" as a
  risk increase, not a neutral choice.
- Never annualize a single 0DTE outcome into an expected return.

## Honesty requirements
- Backtested or optimized performance is an in-sample estimate; say
  out-of-sample or forward numbers are what count (see
  optimizer-results-review skill).
- If the user proposes size that violates these rules, say so plainly and
  show the math — then respect that it's their call.
