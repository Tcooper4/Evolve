#!/usr/bin/env python3
"""
Simple test script for enhanced cost modeling functionality.
"""

def test_imports():
    """Test that all imports work correctly."""
    try:
        from trading.backtesting.performance_analysis import PerformanceAnalyzer, CostParameters
        from trading.backtesting.cost_model import CostModel, CostConfig
        from trading.ui.cost_config import CostConfigUI
        print("✅ All imports successful")
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
            commission_rate=0.002,
            slippage_rate=0.003,
            enable_cost_adjustment=False
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
        import pandas as pd
        import numpy as np
        from trading.backtesting.performance_analysis import PerformanceAnalyzer, CostParameters

        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        equity_curve = pd.DataFrame({
            'equity_curve': np.linspace(100000, 110000, 100),
            'returns': np.random.normal(0.001, 0.02, 100)
        }, index=dates)

        trade_log = pd.DataFrame({
            'price': [100] * 10,
            'quantity': [1000] * 10,
            'pnl': [10] * 10
        })

        # Test analyzer
        cost_params = CostParameters()
        analyzer = PerformanceAnalyzer(cost_params)
        metrics = analyzer.compute_metrics(equity_curve, trade_log)

        # Check that key metrics are present
        assert 'total_return' in metrics
        assert 'cost_adjusted_return' in metrics
        assert 'total_trading_costs' in metrics

        print("✅ PerformanceAnalyzer functionality works")
        return True
    except Exception as e:
        print(f"❌ PerformanceAnalyzer error: {e}")
        return False

def test_cost_model():
    """Test CostModel functionality."""
    try:
        import pandas as pd
        import numpy as np
        from trading.backtesting.cost_model import CostModel, CostConfig

        # Create sample data
        data = pd.DataFrame({
            'close': np.linspace(100, 110, 100)
        })

        # Test cost model
        config = CostConfig(fee_rate=0.001, slippage_rate=0.002, spread_rate=0.0005)
        cost_model = CostModel(config, data)

        # Test cost calculation
        cost_breakdown = cost_model.calculate_total_cost(
            price=100,
            quantity=1000,
            trade_type='BUY'
        )

        assert 'total_cost' in cost_breakdown
        assert 'fees' in cost_breakdown
        assert 'spread' in cost_breakdown
        assert 'slippage' in cost_breakdown

        print("✅ CostModel functionality works")
        return True
    except Exception as e:
        print(f"❌ CostModel error: {e}")
        return False

def test_cost_config_ui():
    """Test CostConfigUI functionality."""
    try:
        from trading.ui.cost_config import CostConfigUI, _get_preset_params

        # Test UI config
        config = CostConfigUI()
        assert config.show_advanced is False
        assert config.show_presets is True

        # Test preset parameters
        retail_params = _get_preset_params("Retail Trading")
        assert retail_params.commission_rate == 0.001
        assert retail_params.slippage_rate == 0.002

        print("✅ CostConfigUI functionality works")
        return True
    except Exception as e:
        print(f"❌ CostConfigUI error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Enhanced Cost Modeling Functionality")
    print("=" * 50)

    tests = [
        test_imports,
        test_cost_parameters,
        test_performance_analyzer,
        test_cost_model,
        test_cost_config_ui
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Enhanced cost modeling is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")

    return passed == total

if __name__ == "__main__":
    main()

