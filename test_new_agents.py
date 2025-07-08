#!/usr/bin/env python3
"""
Test Script for New Agents (Prompt Router and Regime Detection)

This script tests the newly added Prompt Router and Regime Detection agents
to ensure they work correctly and integrate with the existing system.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_agent_imports():
    """Test that the new agents can be imported."""
    print("🔧 Testing Agent Imports...")
    
    try:
        from trading.agents import PromptRouterAgent, RegimeDetectionAgent, create_prompt_router, create_regime_detection_agent
        print("✅ Prompt Router and Regime Detection agents imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_prompt_router_agent():
    """Test the Prompt Router agent functionality."""
    print("\n🔧 Testing Prompt Router Agent...")
    
    try:
        from trading.agents import create_prompt_router
        
        # Create router agent
        router = create_prompt_router()
        print("✅ Prompt Router agent created")
        
        # Test request routing
        test_requests = [
            "What will AAPL stock price be next week?",
            "Generate a trading strategy for TSLA",
            "Analyze the market performance of GOOGL",
            "Optimize the parameters for my RSI strategy",
            "Check my portfolio allocation",
            "What's the system status?",
            "Hello, how are you?"
        ]
        
        for request in test_requests:
            decision = router.route_request(request)
            print(f"✅ Routed '{request[:30]}...' to {decision.primary_agent} (confidence: {decision.confidence:.2f})")
        
        # Test routing statistics
        stats = router.get_routing_statistics()
        print(f"✅ Routing statistics: {stats['total_requests']} requests processed")
        
        return True
        
    except Exception as e:
        print(f"❌ Prompt Router agent error: {e}")
        return False

def test_regime_detection_agent():
    """Test the Regime Detection agent functionality."""
    print("\n🔧 Testing Regime Detection Agent...")
    
    try:
        from trading.agents import create_regime_detection_agent
        
        # Create regime detection agent
        regime_agent = create_regime_detection_agent()
        print("✅ Regime Detection agent created")
        
        # Create sample market data
        np.random.seed(42)
        n_samples = 200
        
        # Generate different market scenarios
        scenarios = {
            'bull_market': {
                'trend': 0.02,  # Upward trend
                'volatility': 0.15,
                'description': 'Bull market with moderate volatility'
            },
            'bear_market': {
                'trend': -0.02,  # Downward trend
                'volatility': 0.20,
                'description': 'Bear market with higher volatility'
            },
            'sideways_market': {
                'trend': 0.0,  # No trend
                'volatility': 0.10,
                'description': 'Sideways market with low volatility'
            },
            'volatile_market': {
                'trend': 0.0,  # No trend
                'volatility': 0.40,
                'description': 'Volatile market with high volatility'
            }
        }
        
        for scenario_name, params in scenarios.items():
            # Generate price data
            returns = np.random.normal(params['trend'], params['volatility'], n_samples)
            prices = 100 * np.exp(np.cumsum(returns))
            
            # Create DataFrame
            data = pd.DataFrame({
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, n_samples)
            })
            
            # Detect regime
            result = regime_agent.detect_regime(data, f"TEST_{scenario_name.upper()}")
            
            print(f"✅ {scenario_name}: Detected {result.regime.value} regime "
                  f"(confidence: {result.confidence:.2f}, strategies: {result.recommended_strategies})")
        
        # Test regime statistics
        stats = regime_agent.get_regime_statistics()
        print(f"✅ Regime statistics: {stats['total_detections']} detections, "
              f"recent regime: {stats['recent_regime']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Regime Detection agent error: {e}")
        return False

def test_agent_integration():
    """Test integration between the new agents."""
    print("\n🔧 Testing Agent Integration...")
    
    try:
        from trading.agents import create_prompt_router, create_regime_detection_agent
        
        # Create both agents
        router = create_prompt_router()
        regime_agent = create_regime_detection_agent()
        
        # Test routing a regime-related request
        regime_request = "What's the current market regime for AAPL?"
        routing_decision = router.route_request(regime_request)
        
        print(f"✅ Routed regime request to: {routing_decision.primary_agent}")
        
        # Test regime detection with sample data
        np.random.seed(42)
        data = pd.DataFrame({
            'close': 100 * np.exp(np.cumsum(np.random.normal(0.01, 0.15, 100))),
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        
        regime_result = regime_agent.detect_regime(data, "AAPL")
        
        print(f"✅ Detected regime: {regime_result.regime.value} "
              f"with recommended strategies: {regime_result.recommended_strategies}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent integration error: {e}")
        return False

def test_agent_interface():
    """Test that agents follow the BaseAgent interface."""
    print("\n🔧 Testing Agent Interface...")
    
    try:
        from trading.agents import PromptRouterAgent, RegimeDetectionAgent
        from trading.agents.base_agent_interface import BaseAgent
        
        # Test inheritance
        assert issubclass(PromptRouterAgent, BaseAgent)
        assert issubclass(RegimeDetectionAgent, BaseAgent)
        print("✅ Agents inherit from BaseAgent")
        
        # Test required methods exist
        router = PromptRouterAgent()
        regime_agent = RegimeDetectionAgent()
        
        # Check for specific methods
        assert hasattr(router, 'route_request'), "PromptRouterAgent missing route_request"
        assert hasattr(regime_agent, 'detect_regime'), "RegimeDetectionAgent missing detect_regime"
        
        print("✅ Agents have required methods")
        
        return True
        
    except Exception as e:
        print(f"❌ Interface test error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting New Agents Tests")
    print("=" * 50)
    
    tests = [
        ("Agent Imports", test_agent_imports),
        ("Prompt Router Agent", test_prompt_router_agent),
        ("Regime Detection Agent", test_regime_detection_agent),
        ("Agent Integration", test_agent_integration),
        ("Agent Interface", test_agent_interface),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! New agents are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 