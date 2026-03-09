import os
import sys
import traceback


def main() -> None:
    sys.path.insert(0, os.getcwd())

    try:
        from trading.execution.execution_engine import ExecutionEngine

        engine = ExecutionEngine()
        result = engine.execute_order(
            {
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 1,
                "order_type": "market",
                "price": None,
            }
        )
        print(f"[OK] Order result: {result}")

        assert result.get("success"), f"Order failed: {result}"
        avg_price = float(result.get("avg_price") or 0.0)
        assert avg_price > 0, f"Price looks wrong: {avg_price}"
        print(f"[OK] Filled at ${avg_price:.2f}")
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

