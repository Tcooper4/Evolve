"""
Output validation for all major platform components. Tests that each returns
non-trivial, plausible real data.

Run after adding any new feature:
  .\\evolve_venv\\Scripts\\python.exe scripts\\verify_outputs.py
"""

import sys
import time
from typing import Any, Callable, Optional

PASSED = []
FAILED = []
WARNINGS = []


def check(
    name: str,
    fn: Callable[[], Any],
    assert_fn: Optional[Callable[[Any], bool]] = None,
    warn_fn: Optional[Callable[[Any], Optional[str]]] = None,
) -> None:
    try:
        start = time.time()
        result = fn()
        elapsed = time.time() - start

        if assert_fn is not None and not assert_fn(result):
            _r = repr(result)
            if len(_r) > 120:
                _r = _r[:117] + "..."
            FAILED.append(
                f"{name}: assertion failed — result={_r}"
            )
            return

        if warn_fn is not None:
            msg = warn_fn(result)
            if msg:
                WARNINGS.append(f"{name}: {msg}")

        PASSED.append(f"{name} ({elapsed:.1f}s)")
    except Exception as e:
        FAILED.append(f"{name}: {e}")


from trading.analysis.ai_score import compute_ai_score
from trading.analysis.factor_model import FactorModel
from trading.analysis.signal_ic import get_ic_analyzer
from trading.data.options_flow import get_options_flow
from trading.data.price_cache import get_history, get_quote
from trading.data.sec_edgar import get_cik, get_latest_filing
from trading.optimization.portfolio_optimizer import get_portfolio_optimizer

import pandas as pd

check(
    "get_quote(AAPL)",
    lambda: get_quote("AAPL"),
    assert_fn=lambda r: (
        r is not None
        and (r.get("price") or 0) > 0
    ),
)

check(
    "compute_ai_score(AAPL)",
    lambda: compute_ai_score("AAPL"),
    assert_fn=lambda r: (
        isinstance(r, dict)
        and 4.0 <= r.get("overall_score", 0) <= 10.0
    ),
    warn_fn=lambda r: (
        f"score={r.get('overall_score'):.1f} suspiciously low"
        if r.get("overall_score", 5) < 2.0
        else None
    ),
)

check(
    "get_options_flow(AAPL)",
    lambda: get_options_flow("AAPL"),
    assert_fn=lambda r: (
        isinstance(r, dict)
        and r.get("success") is True
        and r.get("source") == "yfinance"
    ),
    warn_fn=lambda r: (
        "using simulated data"
        if r.get("source") != "yfinance"
        else None
    ),
)


def _test_factor() -> dict:
    fm = FactorModel()
    hist = get_history("AAPL", period="126d")
    col = [c for c in hist.columns if c.lower() == "close"][0]
    returns = hist[col].pct_change().dropna()
    return fm.compute_exposures("AAPL", returns, ohlcv=hist)


check(
    "factor_model(AAPL)",
    _test_factor,
    assert_fn=lambda r: isinstance(r, dict) and len(r) > 0,
    warn_fn=lambda r: (
        "all exposures zero — column mapping issue?"
        if all(float(v) == 0.0 for v in r.values())
        else None
    ),
)

check(
    "get_cik(AAPL)",
    lambda: get_cik("AAPL"),
    assert_fn=lambda r: r == "0000320193",
)

check(
    "get_latest_filing(AAPL, 10-Q)",
    lambda: get_latest_filing("AAPL", "10-Q"),
    assert_fn=lambda r: (
        r is not None and r.get("form") == "10-Q"
    ),
)


def _test_ic() -> Any:
    a = get_ic_analyzer()
    return a.compute_ic_for_symbol("AAPL", lookback_days=126)


check(
    "ic_analysis(AAPL, 126d)",
    _test_ic,
    assert_fn=lambda r: r is not None and r.n_signals >= 5,
    warn_fn=lambda r: (
        f"IC={r.ic_7d:.3f} outside expected range [-0.3, 0.3]"
        if r is not None and abs(r.ic_7d) > 0.3
        else None
    ),
)


def _test_portfolio() -> dict:
    opt = get_portfolio_optimizer()
    syms = ["AAPL", "MSFT"]
    returns = {}
    for s in syms:
        h = get_history(s, period="63d")
        col = [c for c in h.columns if c.lower() == "close"][0]
        returns[s] = h[col].pct_change().dropna()
    df = pd.DataFrame(returns).dropna()
    return opt.mean_variance_optimization(df, target_return=None)


check(
    "portfolio_optimizer(AAPL+MSFT)",
    _test_portfolio,
    assert_fn=lambda r: (
        r is not None
        and isinstance(r, dict)
        and "weights" in r
    ),
)


def main() -> None:
    print()
    print("OUTPUT VALIDATION RESULTS")
    print("=" * 50)
    print(f"PASSED ({len(PASSED)}):")
    for p in PASSED:
        print(f"   {p}")
    if WARNINGS:
        print(f"WARNINGS ({len(WARNINGS)}):")
        for w in WARNINGS:
            print(f"   {w}")
    if FAILED:
        print(f"FAILED ({len(FAILED)}):")
        for fmsg in FAILED:
            print(f"   {fmsg}")

    total = len(PASSED) + len(FAILED)
    print()
    print(f"Result: {len(PASSED)}/{total} passed")
    sys.exit(0 if not FAILED else 1)


if __name__ == "__main__":
    main()
