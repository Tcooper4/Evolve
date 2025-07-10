#!/usr/bin/env python3
"""
Test script to verify app startup without errors.
"""

import sys
import traceback
import os

# Load environment variables from .env file BEFORE any other imports
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment variables")
except Exception as e:
    print(f"⚠️ Error loading .env file: {e}")

# Debug: Check if API keys are loaded
alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
finnhub_key = os.getenv('FINNHUB_API_KEY')
polygon_key = os.getenv('POLYGON_API_KEY')

if alpha_vantage_key:
    print(f"✅ ALPHA_VANTAGE_API_KEY loaded: {alpha_vantage_key[:10]}...")
else:
    print("❌ ALPHA_VANTAGE_API_KEY not found")

if finnhub_key:
    print(f"✅ FINNHUB_API_KEY loaded: {finnhub_key[:10]}...")
else:
    print("❌ FINNHUB_API_KEY not found")

if polygon_key:
    print(f"✅ POLYGON_API_KEY loaded: {polygon_key[:10]}...")
else:
    print("❌ POLYGON_API_KEY not found")

print()

def test_imports():
    """Test all critical imports."""
    print("🔍 Testing critical imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import plotly
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    return True

def test_core_modules():
    """Test core trading modules."""
    print("\n🔍 Testing core trading modules...")
    
    try:
        from trading.llm.agent import PromptAgent
        print("✅ PromptAgent imported successfully")
    except ImportError as e:
        print(f"❌ PromptAgent import failed: {e}")
        return False
    
    # Test strategy engine import
    try:
        from trading.strategies.enhanced_strategy_engine import StrategyEngine
        print("✅ StrategyEngine imported successfully")
    except ImportError as e:
        print(f"❌ StrategyEngine import failed: {e}")
        return False
    
    try:
        from models.forecast_router import ForecastRouter
        print("✅ ForecastRouter imported successfully")
    except ImportError as e:
        print(f"❌ ForecastRouter import failed: {e}")
        return False
    
    return True

def test_agent_imports():
    """Test agent imports."""
    print("\n🔍 Testing agent imports...")
    
    try:
        from trading.agents.model_improver_agent import ModelImprovementRequest
        print("✅ ModelImprovementRequest imported successfully")
    except ImportError as e:
        print(f"❌ ModelImprovementRequest import failed: {e}")
        return False
    
    try:
        from trading.agents.execution_agent import ExecutionAgent, ExecutionRequest, ExecutionResult
        print("✅ ExecutionAgent imports successful")
    except ImportError as e:
        print(f"❌ ExecutionAgent imports failed: {e}")
        return False
    
    try:
        from trading.agents.execution_risk_agent import ExecutionRiskAgent, RiskAssessmentRequest, RiskAssessmentResult
        print("✅ ExecutionRiskAgent imports successful")
    except ImportError as e:
        print(f"❌ ExecutionRiskAgent imports failed: {e}")
        return False
    
    try:
        from trading.agents.execution_risk_control_agent import ExecutionRiskControlAgent, RiskControlRequest, RiskControlResult
        print("✅ ExecutionRiskControlAgent imports successful")
    except ImportError as e:
        print(f"❌ ExecutionRiskControlAgent imports failed: {e}")
        return False
    
    return True

def test_evaluation_metrics():
    """Test evaluation metrics imports."""
    print("\n🔍 Testing evaluation metrics...")
    
    try:
        from trading.evaluation.metrics import calculate_sharpe_ratio, calculate_max_drawdown, calculate_win_rate
        print("✅ Evaluation metrics imported successfully")
    except ImportError as e:
        print(f"❌ Evaluation metrics import failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 Starting Evolve AI Trading Platform Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_core_modules,
        test_agent_imports,
        test_evaluation_metrics
    ]
    
    all_passed = True
    
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! The app should start successfully.")
        print("💡 You can now run: streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 