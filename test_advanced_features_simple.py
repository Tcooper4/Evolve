#!/usr/bin/env python3
"""Simple test script for advanced features."""

import sys
import traceback
from typing import Dict, List

def test_imports():
    """Test all advanced feature imports."""
    results = {}
    
    # Test basic imports
    try:
        import pandas as pd
        import numpy as np
        import streamlit as st
        results['basic_imports'] = "✅ PASS"
    except ImportError as e:
        results['basic_imports'] = f"❌ FAIL: {e}"
    
    # Test configuration loader
    try:
        from utils.config_loader import ConfigLoader
        config = ConfigLoader()
        results['config_loader'] = "✅ PASS"
    except ImportError as e:
        results['config_loader'] = f"❌ FAIL: {e}"
    
    # Test RL module (with fallback)
    try:
        from rl.strategy_trainer import create_rl_strategy_trainer
        trainer = create_rl_strategy_trainer()
        results['rl_module'] = "✅ PASS" if trainer['available'] else "⚠️ PARTIAL (dependencies missing)"
    except ImportError as e:
        results['rl_module'] = f"❌ FAIL: {e}"
    
    # Test causal inference
    try:
        from causal.causal_model import create_causal_model
        model = create_causal_model()
        results['causal_inference'] = "✅ PASS"
    except ImportError as e:
        results['causal_inference'] = f"❌ FAIL: {e}"
    
    # Test TFT model
    try:
        from models.tft_model import create_tft_model
        results['tft_model'] = "✅ PASS"
    except ImportError as e:
        results['tft_model'] = f"❌ FAIL: {e}"
    
    # Test strategy gatekeeper
    try:
        from strategies.gatekeeper import create_strategy_gatekeeper
        results['strategy_gatekeeper'] = "✅ PASS"
    except ImportError as e:
        results['strategy_gatekeeper'] = f"❌ FAIL: {e}"
    
    # Test streaming pipeline
    try:
        from data.streaming_pipeline import create_streaming_pipeline
        results['streaming_pipeline'] = "✅ PASS"
    except ImportError as e:
        results['streaming_pipeline'] = f"❌ FAIL: {e}"
    
    # Test live trading interface
    try:
        from execution.live_trading_interface import create_live_trading_interface
        results['live_trading_interface'] = "✅ PASS"
    except ImportError as e:
        results['live_trading_interface'] = f"❌ FAIL: {e}"
    
    # Test voice interface
    try:
        from ui.voice_interface import create_voice_interface
        results['voice_interface'] = "✅ PASS"
    except ImportError as e:
        results['voice_interface'] = f"❌ FAIL: {e}"
    
    # Test risk engine
    try:
        from risk.tail_risk import TailRiskEngine
        results['risk_engine'] = "✅ PASS"
    except ImportError as e:
        results['risk_engine'] = f"❌ FAIL: {e}"
    
    # Test model generator agent
    try:
        from agents.model_generator_agent import create_model_generator_agent
        results['model_generator_agent'] = "✅ PASS"
    except ImportError as e:
        results['model_generator_agent'] = f"❌ FAIL: {e}"
    
    return results

def test_functionality():
    """Test basic functionality of advanced features."""
    results = {}
    
    # Test configuration loading
    try:
        from utils.config_loader import ConfigLoader
        config = ConfigLoader()
        settings = config.get_optimization_settings()
        results['config_functionality'] = "✅ PASS" if settings else "❌ FAIL"
    except Exception as e:
        results['config_functionality'] = f"❌ FAIL: {e}"
    
    # Test causal model functionality
    try:
        import pandas as pd
        import numpy as np
        from causal.causal_model import create_causal_model
        
        # Create sample data
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'target': np.random.normal(0, 1, 100)
        })
        
        model = create_causal_model()
        analysis = model.analyze_market_relationships(data, 'target')
        results['causal_functionality'] = "✅ PASS" if analysis else "⚠️ PARTIAL"
    except Exception as e:
        results['causal_functionality'] = f"❌ FAIL: {e}"
    
    # Test RL environment creation
    try:
        import pandas as pd
        import numpy as np
        from rl.strategy_trainer import create_rl_environment
        
        # Create sample data
        data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 50),
            'High': np.random.uniform(105, 115, 50),
            'Low': np.random.uniform(95, 105, 50),
            'Close': np.random.uniform(100, 110, 50),
            'Volume': np.random.uniform(1000000, 5000000, 50)
        })
        
        env = create_rl_environment(data)
        obs, info = env.reset()
        results['rl_functionality'] = "✅ PASS" if obs is not None else "❌ FAIL"
    except Exception as e:
        results['rl_functionality'] = f"❌ FAIL: {e}"
    
    return results

def main():
    """Run all tests."""
    print("🧪 Testing Advanced Features")
    print("=" * 50)
    
    print("\n📦 Testing Imports:")
    import_results = test_imports()
    for feature, result in import_results.items():
        print(f"  {feature:25} {result}")
    
    print("\n⚙️  Testing Functionality:")
    func_results = test_functionality()
    for feature, result in func_results.items():
        print(f"  {feature:25} {result}")
    
    # Summary
    print("\n📊 Summary:")
    total_tests = len(import_results) + len(func_results)
    passed_tests = sum(1 for r in import_results.values() if "PASS" in r)
    passed_tests += sum(1 for r in func_results.values() if "PASS" in r)
    
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Advanced features are ready.")
    else:
        print("\n⚠️  Some tests failed. Check the results above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 