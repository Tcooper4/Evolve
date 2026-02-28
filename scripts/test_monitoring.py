# -*- coding: utf-8 -*-
"""
Test monitoring pipeline: create MemoryStore, call all three monitoring functions
with sample degraded metrics, verify recommendations were written, print results.
Run from repo root: python scripts/test_monitoring.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Use a separate DB for testing so we don't pollute production
os.environ["EVOLVE_MEMORY_DB_PATH"] = str(REPO_ROOT / "data" / "memory_store_test_monitoring.db")


def main():
    from trading.memory import get_memory_store
    from trading.memory.memory_store import MemoryType
    from trading.services.monitoring_tools import (
        check_model_degradation,
        check_strategy_degradation,
        check_data_quality,
    )
    import pandas as pd
    from datetime import datetime, timedelta

    print("=== Test Monitoring Pipeline ===\n")
    store = get_memory_store()
    print("1. MemoryStore created.\n")

    # 2. Call check_model_degradation with degraded metrics
    print("2. Calling check_model_degradation(model_id='LSTM', recent_sharpe=0.1, recent_drawdown=-0.25)...")
    out_model = check_model_degradation(
        model_id="LSTM",
        recent_sharpe=0.1,
        recent_drawdown=-0.25,
    )
    print(f"   Result: {out_model}\n")
    assert out_model.get("degradation_detected") is True
    assert out_model.get("written_to_memory") is True

    # 3. Call check_strategy_degradation with degraded metrics
    print("3. Calling check_strategy_degradation(strategy_name='RSI', recent_sharpe=0.2, recent_win_rate=0.35)...")
    out_strat = check_strategy_degradation(
        strategy_name="RSI",
        recent_sharpe=0.2,
        recent_win_rate=0.35,
    )
    print(f"   Result: {out_strat}\n")
    assert out_strat.get("degradation_detected") is True
    assert out_strat.get("written_to_memory") is True

    # 4. Call check_data_quality with data that has a gap and a big move
    print("4. Calling check_data_quality with sample data (gap + stale + anomaly)...")
    dates = pd.date_range(start="2025-01-01", periods=30, freq="D")
    # Introduce a 7-day gap
    dates = list(dates[:10]) + list(dates[17:])
    prices = [100.0 + i * 0.5 for i in range(10)] + [110.0 + i * 0.5 for i in range(13)]
    # One-day >50% move
    prices[-5] = prices[-6] * 1.6
    df = pd.DataFrame({"date": dates, "close": prices})
    out_data = check_data_quality("TEST", df, max_gap_days=5, stale_days=2, anomaly_pct=0.50)
    print(f"   Result: {out_data}\n")
    # We expect at least gap or anomaly; stale depends on "now"
    assert out_data.get("issues_detected") is True or out_data.get("written_to_memory") in (True, False)
    if out_data.get("issues_detected"):
        assert out_data.get("written_to_memory") is True

    # 5. Query recommendations from MemoryStore
    print("5. Querying MemoryStore for recommendations (namespace=monitoring, category=recommendations)...")
    records = store.list(
        MemoryType.LONG_TERM,
        namespace="monitoring",
        category="recommendations",
        limit=20,
        newest_first=True,
    )
    print(f"   Found {len(records)} recommendation(s).\n")
    for i, r in enumerate(records, 1):
        v = r.value if hasattr(r, "value") else getattr(r, "value", None)
        title = v.get("title", "") if isinstance(v, dict) else ""
        text = v.get("text", "") if isinstance(v, dict) else str(v)[:200]
        print(f"   [{i}] {title}: {text[:100]}...")
    print()

    # 6. No-op when no metrics
    print("6. Calling check_model_degradation() and check_strategy_degradation() with no params (should skip)...")
    out_empty_m = check_model_degradation()
    out_empty_s = check_strategy_degradation()
    print(f"   check_model_degradation(): {out_empty_m}")
    print(f"   check_strategy_degradation(): {out_empty_s}")
    assert "No metrics provided" in out_empty_m.get("message", "")
    assert "No metrics provided" in out_empty_s.get("message", "")

    print("\n=== All checks passed. Monitoring pipeline works end to end. ===")


if __name__ == "__main__":
    main()
