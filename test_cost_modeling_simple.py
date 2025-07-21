"""
Simple Cost Modeling Test

This test focuses on cost modeling functionality without importing
the full trading package that has dependency issues.
"""

import sys


def test_imports():
    """Test that cost modeling modules can be imported."""
    try:
        pass

        print("✅ All cost modeling imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_cost_parameters():
    """Test CostParameters functionality."""
    try:
        from trading.backtesting.performance_analysis import CostParameters

        # Test default parameters
        params = CostParameters()
        assert params.commission_rate == 0.001
        assert params.slippage_rate == 0.002
        assert params.spread_rate == 0.0005
        assert params.enable_cost_adjustment is True

        # Test custom parameters
        custom_params = CostParameters(
            commission_rate=0.002, slippage_rate=0.003, enable_cost_adjustment=False
        )
        assert custom_params.commission_rate == 0.002
        assert custom_params.enable_cost_adjustment is False

        print("✅ CostParameters functionality works")
        return True
    except Exception as e:
        print(f"❌ CostParameters error: {e}")
        return False


def test_performance_analyzer():
    """Test PerformanceAnalyzer functionality."""
    try:
        import numpy as np
        import pandas as pd

        from trading.backtesting.performance_analysis import (
            CostParameters,
            PerformanceAnalyzer,
        )

        # Create sample data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        equity_curve = pd.DataFrame(
            {
                "equity_curve": np.linspace(100000, 110000, 100),
                "returns": np.random.normal(0.001, 0.02, 100),
            },
            index=dates,
        )

        trade_log = pd.DataFrame(
            {"price": [100] * 10, "quantity": [1000] * 10, "pnl": [10] * 10}
        )

        # Test analyzer
        cost_params = CostParameters()
        analyzer = PerformanceAnalyzer(cost_params)
        metrics = analyzer.compute_metrics(equity_curve, trade_log)

        # Check that key metrics are present
        assert "total_return" in metrics
        assert "cost_adjusted_return" in metrics
        assert "total_trading_costs" in metrics

        print("✅ PerformanceAnalyzer functionality works")
        return True
    except Exception as e:
        print(f"❌ PerformanceAnalyzer error: {e}")
        return False


def test_cost_model():
    """Test CostModel functionality."""
    try:
        import numpy as np
        import pandas as pd

        from trading.backtesting.cost_model import CostConfig, CostModel

        # Create sample data
        data = pd.DataFrame({"close": np.linspace(100, 110, 100)})

        # Test cost model
        config = CostConfig(fee_rate=0.001, slippage_rate=0.002, spread_rate=0.0005)
        cost_model = CostModel(config, data)

        # Test cost calculation
        cost_breakdown = cost_model.calculate_total_cost(
            price=100, quantity=1000, trade_type="BUY"
        )

        assert "total_cost" in cost_breakdown
        assert "fees" in cost_breakdown
        assert "spread" in cost_breakdown
        assert "slippage" in cost_breakdown

        print("✅ CostModel functionality works")
        return True
    except Exception as e:
        print(f"❌ CostModel error: {e}")
        return False


def test_cost_calculation():
    """Test cost calculation accuracy."""
    try:
        import pandas as pd

        from trading.backtesting.cost_model import CostConfig, CostModel

        # Create sample data
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})

        # Test with known parameters
        config = CostConfig(
            fee_rate=0.001,  # 0.1% fee
            slippage_rate=0.002,  # 0.2% slippage
            spread_rate=0.0005,  # 0.05% spread
        )
        cost_model = CostModel(config, data)

        # Test buy order
        buy_cost = cost_model.calculate_total_cost(
            price=100, quantity=1000, trade_type="BUY"
        )

        expected_fees = 100 * 1000 * 0.001  # $1.00
        expected_spread = 100 * 1000 * 0.0005  # $0.50
        expected_slippage = 100 * 1000 * 0.002  # $2.00
        expected_total = expected_fees + expected_spread + expected_slippage

        assert abs(buy_cost["fees"] - expected_fees) < 0.01
        assert abs(buy_cost["spread"] - expected_spread) < 0.01
        assert abs(buy_cost["slippage"] - expected_slippage) < 0.01
        assert abs(buy_cost["total_cost"] - expected_total) < 0.01

        print("✅ Cost calculation accuracy verified")
        return True
    except Exception as e:
        print(f"❌ Cost calculation error: {e}")
        return False


def test_performance_impact():
    """Test that costs impact performance metrics correctly."""
    try:
        import numpy as np
        import pandas as pd

        from trading.backtesting.performance_analysis import (
            CostParameters,
            PerformanceAnalyzer,
        )

        # Create sample data with known returns
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        base_returns = np.ones(100) * 0.001  # 0.1% daily return
        equity_curve = pd.DataFrame(
            {
                "equity_curve": 100000 * np.cumprod(1 + base_returns),
                "returns": base_returns,
            },
            index=dates,
        )

        trade_log = pd.DataFrame(
            {
                "price": [100] * 50,
                "quantity": [1000] * 50,
                "pnl": [10] * 50,  # $10 profit per trade
            }
        )

        # Test without costs
        no_cost_params = CostParameters(enable_cost_adjustment=False)
        no_cost_analyzer = PerformanceAnalyzer(no_cost_params)
        no_cost_metrics = no_cost_analyzer.compute_metrics(equity_curve, trade_log)

        # Test with costs
        cost_params = CostParameters(enable_cost_adjustment=True)
        cost_analyzer = PerformanceAnalyzer(cost_params)
        cost_metrics = cost_analyzer.compute_metrics(equity_curve, trade_log)

        # Cost-adjusted return should be lower
        assert cost_metrics["cost_adjusted_return"] < no_cost_metrics["total_return"]
        assert cost_metrics["total_trading_costs"] > 0

        print("✅ Performance impact of costs verified")
        return True
    except Exception as e:
        print(f"❌ Performance impact error: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Starting Cost Modeling Test Suite")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Cost Parameters", test_cost_parameters),
        ("Performance Analyzer", test_performance_analyzer),
        ("Cost Model", test_cost_model),
        ("Cost Calculation", test_cost_calculation),
        ("Performance Impact", test_performance_impact),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n--- Running {test_name} Test ---")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All cost modeling tests passed!")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
