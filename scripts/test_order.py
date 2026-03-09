"""
Minimal order-flow test: verifies ExecutionEngine.execute_order works end-to-end.
Usage: .\evolve_venv\Scripts\python.exe scripts\test_order.py
"""

import os
import sys

sys.path.insert(0, os.getcwd())


def main() -> int:
    from trading.execution.execution_engine import ExecutionEngine

    engine = ExecutionEngine()
    result = engine.execute_order(
        {"symbol": "AAPL", "side": "buy", "quantity": 1, "order_type": "market"}
    )
    print(f"Order result: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

