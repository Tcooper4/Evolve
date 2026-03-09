# Technical Debt

Generated from codebase grep for TODO, FIXME, HACK, XXX, TEMP, temporary, placeholder.

| File | Line | Comment | Priority |
|------|------|---------|----------|
| pages/7_Performance.py | 278 | sample_returns = None  # TODO: load from strategy logger or trade history | Medium |
| pages/7_Performance.py | 1447 | sample_returns = None  # TODO: load from strategy logger or trade history | Medium |
| pages/7_Performance.py | 1451 | benchmark_returns = None  # TODO: load benchmark from data provider | Medium |
| pages/7_Performance.py | 2000 | sample_returns = None  # TODO: load from strategy logger or trade history | Medium |
| pages/9_Reports.py | 228 | has_report_data = False  # TODO: set True when real data from backtest/logs exists | Medium |
| pages/6_Risk_Management.py | 266 | # TODO: get actual returns from portfolio manager / FallbackDataProvider / strategy logs | Medium |
| trading/strategies/hybrid_engine.py | 324 | # This is a placeholder for more sophisticated correlation analysis | Low |
| trading/backtesting/edge_case_handler.py | 219 | # Create performance metrics placeholder | Low |
| trading/models/tcn_model.py | 516 | "confidence": 0.8,  # Placeholder confidence | Low |
| trading/models/lstm_model.py | 1615 | "confidence": np.full(horizon, 0.8),  # Placeholder confidence | Low |
| trading/config/__init__.py | 112 | except Exception as _unused_var:  # Placeholder, flake8 ignore | Low |
| trading/backtesting/enhanced_backtester.py | 570 | _unused_var = trade  # Placeholder, flake8 ignore | Low |

Priority: High = blocks correctness or security; Medium = missing feature or data source; Low = cosmetic or placeholder.
