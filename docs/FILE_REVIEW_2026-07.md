# File Review — 2026-07-10 (untouched-file pass)

Scope: every live `.py` file NOT modified by the Fable session (542 of 574
total; 404 live non-test files), reviewed for "is this dead weight, does it
need upgrading, or is it fine as-is" — with the explicit standard that
nothing is removed without mechanical proof plus an individual read, and
removal means reversible archival, never deletion.

## Method (why this can be trusted)

1. **AST import graph** from every live entry point (app.py, all seven
   pages, the MCP server, every script) — catches top-level, lazy,
   in-function, and relative imports. 339 of 404 live files proved
   reachable this way.
2. **Dynamic-import exoneration.** The graph alone LIES for this codebase,
   and the review proved it immediately: `atr_strategy.py`/`cci_strategy.py`
   showed as "unreached" despite being executed by the previous day's
   tests. Cause: the strategy registry does **filesystem discovery** —
   it globs `trading/strategies/*.py` and imports every file it finds.
   Three more such mechanisms were found and mapped (forecast router globs
   `*_model.py`; provider manager globs providers; the agents barrel
   exposes lazy `_lazy_import` getters). Everything reachable through any
   of these was exonerated and kept.
3. **Quoted-string reference scan** across the whole tree for each
   remaining candidate (catches registry strings and lazy imports the
   graph resolver missed): 45 of 65 graph-candidates were exonerated as
   live-referenced; 7 were test-only; 13 had zero references of any kind.
4. **Individual file reads** of every true candidate (purpose, size, git
   age, duplication against the live equivalent), then **execution
   checks**: broken packages proven unimportable; every touched barrel
   re-imported clean after archival; full test battery identical
   pass/fail with the change stashed vs applied (zero regressions); all
   seven pages smoke-run clean.

## Outcome

- **15 files/packages archived** to `_archive/orphans-2026-07/` with
  per-file evidence in its `MANIFEST.md`. Highlights: two packages that
  were *unimportable* (imports of submodules that don't exist), a third
  uncoordinated copy of the Sharpe/drawdown formulas, a parallel abandoned
  memory system (961 lines, zero importers, vs the canonical MemoryStore's
  19), a duplicate PositionSizingEngine, a Redis microservice trio
  superseded by the live agents, a discovery mechanism nothing invokes,
  and an unmounted dashboard.
- **1 reversal during the pass** (the conservatism working as intended):
  `rsi_optimizer.py` was archived, then restored when a live module's test
  file turned out to hard-import it — kept and flagged as superseded
  instead.
- **5 working-but-unwired files kept and flagged** (streaming_pipeline,
  prompt_response_validator, risk_analyzer, prompt_processor,
  rsi_optimizer) — each is importable, substantial, and either
  test-covered or a plausible future feature; the keep-or-wire call is
  Thomas's, documented in TECHNICAL_DEBT.md.

## Status of the 339 reachable-but-untouched files

These are live, load-bearing, and NOT dead weight — "untouched for months"
turned out to mean "working" in every spot-checked case (`trading/nlp` is
imported by app.py and the sentiment pipeline; the agents barrel serves
everything through lazy getters; etc.). Their review status is honest and
tiered:

- **Depth-verified (execution-level), this session or the prior audit:**
  all strategies, all 22 sizing methods, all optimization, risk metrics,
  the four torch models, all seven pages at orchestration level, the chat
  pipeline end-to-end, earnings reaction, the services layer touched this
  session.
- **Import/signature-verified (prior audit), not re-deepened:** the
  remainder — notably `trading/data` providers and caches,
  `trading/report`, `trading/memory` internals beyond the store,
  `trading/agents` bodies beyond the ones fixed, `trading/nlp` internals.
  These are the right targets for the next depth sessions, in that order
  of business value: data layer (feeds everything) → nlp/sentiment
  (feeds AI Score) → report → memory internals.

Nothing in this pass found evidence that any reachable file is broken —
but "reachable and signature-checked" is a weaker claim than
"execution-verified," and this document does not pretend otherwise.
