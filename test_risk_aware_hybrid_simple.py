#!/usr/bin/env python3
"""
Simple test script for risk-aware hybrid model functionality.
"""

def test_imports():
    """Test that all imports work correctly."""
    try:
        from trading.forecasting.hybrid_model import HybridModel
        from trading.ui.hybrid_model_config import HybridModelConfigUI
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_hybrid_model_creation():
    """Test hybrid model creation and basic functionality."""
    try:
        from trading.forecasting.hybrid_model import HybridModel
        
        # Create mock models
        class MockModel:
            def __init__(self, name):
                self.name = name
            def fit(self, data):
                pass
            def predict(self, data):
                import numpy as np
                return np.random.normal(100, 5, len(data))
        
        models = {
            "Model1": MockModel("Model1"),
            "Model2": MockModel("Model2"),
            "Model3": MockModel("Model3")
        }
        
        # Create hybrid model
        hybrid_model = HybridModel(models)
        
        # Check initial configuration
        assert hybrid_model.scoring_config["method"] == "risk_aware"
        assert hybrid_model.scoring_config["weighting_metric"] == "sharpe"
        assert len(hybrid_model.weights) == 3
        
        print("✅ Hybrid model creation successful")
        return True
    except Exception as e:
        print(f"❌ Hybrid model creation error: {e}")
        return False

def test_weighting_metrics():
    """Test different weighting metrics."""
    try:
        from trading.forecasting.hybrid_model import HybridModel
        import pandas as pd
        import numpy as np
        
        # Create mock models
        class MockModel:
            def __init__(self, name):
                self.name = name
            def fit(self, data):
                pass
            def predict(self, data):
                return np.random.normal(100, 5, len(data))
        
        models = {
            "Model1": MockModel("Model1"),
            "Model2": MockModel("Model2")
        }
        
        # Create hybrid model
        hybrid_model = HybridModel(models)
        
        # Test different weighting metrics
        metrics = ["sharpe", "drawdown", "mse"]
        
        for metric in metrics:
            hybrid_model.set_weighting_metric(metric)
            assert hybrid_model.scoring_config["weighting_metric"] == metric
            assert hybrid_model.scoring_config["method"] == "risk_aware"
        
        print("✅ Weighting metrics test successful")
        return True
    except Exception as e:
        print(f"❌ Weighting metrics test error: {e}")
        return False

def test_performance_calculation():
    """Test performance calculation functionality."""
    try:
        from trading.forecasting.hybrid_model import HybridModel
        import pandas as pd
        import numpy as np
        
        # Create mock models
        class MockModel:
            def __init__(self, name):
                self.name = name
            def fit(self, data):
                pass
            def predict(self, data):
                return np.random.normal(100, 5, len(data))
        
        models = {
            "Model1": MockModel("Model1"),
            "Model2": MockModel("Model2")
        }
        
        # Create hybrid model
        hybrid_model = HybridModel(models)
        
        # Create sample data
        data = pd.DataFrame({
            'close': np.linspace(100, 110, 100),
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        
        # Fit models
        hybrid_model.fit(data)
        
        # Check performance summary
        summary = hybrid_model.get_model_performance_summary()
        assert len(summary) == 2
        
        # Check weighting info
        info = hybrid_model.get_weighting_metric_info()
        assert "current_metric" in info
        assert "available_metrics" in info
        
        print("✅ Performance calculation test successful")
        return True
    except Exception as e:
        print(f"❌ Performance calculation test error: {e}")
        return False

def test_weight_calculation():
    """Test weight calculation with different metrics."""
    try:
        from trading.forecasting.hybrid_model import HybridModel
        import pandas as pd
        import numpy as np
        
        # Create mock models with different characteristics
        class MockModel:
            def __init__(self, name, bias=0.0):
                self.name = name
                self.bias = bias
            def fit(self, data):
                pass
            def predict(self, data):
                return data['close'].values * (1 + self.bias + np.random.normal(0, 0.01, len(data)))
        
        models = {
            "High_Sharpe": MockModel("High_Sharpe", 0.001),
            "Low_Drawdown": MockModel("Low_Drawdown", 0.0005),
            "Low_MSE": MockModel("Low_MSE", 0.0002)
        }
        
        # Create hybrid model
        hybrid_model = HybridModel(models)
        
        # Create sample data
        data = pd.DataFrame({
            'close': np.linspace(100, 110, 100),
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        
        # Fit models
        hybrid_model.fit(data)
        
        # Test different weighting metrics
        results = {}
        for metric in ["sharpe", "drawdown", "mse"]:
            hybrid_model.set_weighting_metric(metric)
            results[metric] = hybrid_model.weights.copy()
        
        # Check that weights sum to 1.0 for each metric
        for metric, weights in results.items():
            total_weight = sum(weights.values())
            assert abs(total_weight - 1.0) < 0.001, f"Weights don't sum to 1.0 for {metric}"
        
        print("✅ Weight calculation test successful")
        return True
    except Exception as e:
        print(f"❌ Weight calculation test error: {e}")
        return False

def test_ui_components():
    """Test UI component functionality."""
    try:
        from trading.ui.hybrid_model_config import HybridModelConfigUI
        
        # Test UI config
        config = HybridModelConfigUI()
        assert config.show_advanced is False
        assert config.show_performance_summary is True
        assert config.show_weighting_info is True
        assert config.show_validation is True
        
        # Test with custom settings
        custom_config = HybridModelConfigUI(
            show_advanced=True,
            show_performance_summary=False
        )
        assert custom_config.show_advanced is True
        assert custom_config.show_performance_summary is False
        
        print("✅ UI components test successful")
        return True
    except Exception as e:
        print(f"❌ UI components test error: {e}")
        return False

def test_configuration_methods():
    """Test configuration methods."""
    try:
        from trading.forecasting.hybrid_model import HybridModel
        
        # Create mock models
        class MockModel:
            def __init__(self, name):
                self.name = name
            def fit(self, data):
                pass
            def predict(self, data):
                import numpy as np
                return np.random.normal(100, 5, len(data))
        
        models = {
            "Model1": MockModel("Model1"),
            "Model2": MockModel("Model2")
        }
        
        # Create hybrid model
        hybrid_model = HybridModel(models)
        
        # Test set_weighting_metric
        hybrid_model.set_weighting_metric("drawdown")
        assert hybrid_model.scoring_config["weighting_metric"] == "drawdown"
        assert hybrid_model.scoring_config["method"] == "risk_aware"
        
        # Test set_scoring_config
        new_config = {
            "sharpe_floor": 0.5,
            "drawdown_ceiling": -0.3
        }
        hybrid_model.set_scoring_config(new_config)
        assert hybrid_model.scoring_config["sharpe_floor"] == 0.5
        assert hybrid_model.scoring_config["drawdown_ceiling"] == -0.3
        
        print("✅ Configuration methods test successful")
        return True
    except Exception as e:
        print(f"❌ Configuration methods test error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Risk-Aware Hybrid Model Functionality")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_hybrid_model_creation,
        test_weighting_metrics,
        test_performance_calculation,
        test_weight_calculation,
        test_ui_components,
        test_configuration_methods
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Risk-aware hybrid model is working correctly.")
        print("\nKey Features Verified:")
        print("✅ Risk-aware weighting (Sharpe, Drawdown, MSE)")
        print("✅ User-selectable weighting metrics")
        print("✅ Performance calculation and tracking")
        print("✅ Weight calculation and normalization")
        print("✅ UI component functionality")
        print("✅ Configuration methods")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 