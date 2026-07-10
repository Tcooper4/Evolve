# Dead code archived 2026-07-10 (Fable session, reachability sweep)

Every file here passed FOUR independent gates before moving, each verified
mechanically (scripts/reachability_analysis.py + name-level scan):

1. **Unreachable**: not in the transitive import graph from any entry point
   (app.py, all pages, MCP server), including importlib string-literal
   edges and package __init__ execution.
2. **No dynamic/string references**: the module name appears nowhere in
   live .py/.md/.yaml/.json/.ini outside prior dead-code-analysis
   artifacts (which are self-referential) and historical docs.
3. **No exported-name usage**: none of the file's public classes/functions
   are referenced anywhere in reachable code (guards against
   barrel/`from package import Class` invisibility — this gate is what
   kept ATRStrategy/CCIStrategy alive and correctly so).
4. **Not discovery-eligible**: not in a directory scanned by a dynamic
   discovery mechanism (trading/strategies and agents dirs are
   filesystem-discovered and were excluded from archiving on principle),
   and not referenced by any test.

Deliberately NOT archived despite being unreachable:
- trading/agents/leaderboard_dashboard.py, trading/agents/rl_trainer.py —
  live in a discovery-scanned directory (agents/registry.py scans
  trading.agents); left in place, flagged in the review tracker.
- trading/risk/risk_analyzer.py — imported by legacy tests
  (tests/test_imports.py, tests/test_system_status.py).
- Everything with any exported-name reference in live code, even where
  the reference is likely a same-name collision — collisions land on the
  keep side by design.

Post-archive verification: all seven pages AppTest-clean, full session
test battery green, and every parent package of an archived module
imports successfully.
