---
name: signal-interpretation
description: How to read Evolve's technical signals without fooling yourself
triggers: rsi, macd, bollinger, divergence, overbought, oversold, crossover, signal, sma, atr, cci, breakout
---
# Reading Evolve's technical signals

Apply these rules whenever interpreting RSI/MACD/Bollinger/SMA/ATR/CCI output
for the user. The goal is decision quality, not signal count.

## General discipline
- A signal is a *condition*, not an instruction. Always state what would
  invalidate it before suggesting anyone act on it.
- Check trend context first. Mean-reversion signals (RSI oversold, lower
  Bollinger touch, CCI < -100) are far weaker against a strong trend —
  "oversold" in a downtrend mostly means "still falling."
- Confirm with volume when available: breakouts and crossovers on weak
  volume fail more often. Say so explicitly if volume doesn't confirm.
- One timeframe is an opinion; two agreeing is a setup. If the user's chart
  interval is intraday, sanity-check the daily before framing conviction.

## Per-indicator specifics
- **RSI**: oversold/overbought thresholds are regime-dependent — in strong
  uptrends RSI can sit 60–80 for weeks. Divergence (price makes a new
  extreme, RSI doesn't) matters more than absolute level, but only counts
  after the second confirming swing, never on the first.
- **MACD**: the histogram turning is earlier but noisier than the line
  cross. Crossovers near the zero line carry more information than
  crossovers at extremes.
- **Bollinger**: a band *touch* is not a signal by itself. Squeeze (low
  bandwidth) precedes expansion — direction comes from the break, not the
  squeeze. Riding the band is a trend feature, not overbought.
- **SMA cross**: lagging by construction. Whipsaw risk is highest in
  sideways regimes — check the regime tool before leaning on a cross.
- **ATR**: it's a volatility unit, not a direction. Use it to size stops
  (e.g. 1.5–2× ATR) and to normalize moves across names, never as a
  buy/sell signal alone.

## Honesty requirements
- If signals conflict, say they conflict and which one has priority given
  the regime — don't average them into false confidence.
- If data quality is questionable (gaps, thin volume, stale quote), lead
  with that caveat.
