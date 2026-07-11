# Stale test suites archived 2026-07

All four tested previous generations of the optimization APIs and could
not pass against the current code for structural (not behavioral)
reasons; current optimizer behavior is covered by the passing suites
under tests/test_optimization/test_strategy_backtest_objective.py and
the session batteries.

- test_backtest_optimizer.py — imports trading.optimization.backtest_optimizer (module no longer exists)
- test_hyperparameter_tuner.py — imports trading.optimization.optuna_optimizer (module no longer exists)
- test_strategy_optimizer.py — written against a prior StrategyOptimizer API (.optimizer/._create_optimizer/.optimizer_type attributes gone); 16 constructions also missed the now-required config `name`
- test_rsi_strategy.py — subject class uninstantiable (abstract-method drift) plus three more stale layers; archived with its subject (see ../orphans-2026-07-b/MANIFEST.md)

Restore with git mv; each would need a rewrite against current APIs.
