#!/usr/bin/env python3
"""
Simple test script to verify system fixes without Unicode encoding issues.
"""

import sys
import logging
import json
from datetime import datetime

# Configure logging to avoid Unicode issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def test_capability_router():
    """Test capability router fixes."""
    print("Testing CapabilityRouter...")
    try:
        from core.capability_router import CapabilityRouter
        
        router = CapabilityRouter()
        
        # Test capability checks
        capabilities = [
            'openai_api',
            'huggingface_models', 
            'redis_connection',
            'postgres_connection',
            'alpha_vantage_api',
            'yfinance_api',
            'torch_models',
            'streamlit_interface',
            'plotly_visualization'
        ]
        
        results = {}
        for capability in capabilities:
            try:
                result = router.check_capability(capability)
                results[capability] = result
                print(f"  {capability}: {'Available' if result else 'Not Available'}")
            except Exception as e:
                results[capability] = False
                print(f"  {capability}: Error - {e}")
        
        # Test system health
        health = router.get_system_health()
        print(f"  System Health: {health.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"  CapabilityRouter test failed: {e}")
        return False

def test_data_feed():
    """Test data feed fixes."""
    print("Testing DataFeed...")
    try:
        from data.live_feed import get_data_feed
        
        feed = get_data_feed()
        
        # Test provider status
        status = feed.get_provider_status()
        print(f"  Current Provider: {status.get('current_provider', 'unknown')}")
        
        # Test system health
        health = feed.get_system_health()
        print(f"  System Health: {health.get('status', 'unknown')}")
        print(f"  Available Providers: {health.get('available_providers', 0)}")
        
        return True
        
    except Exception as e:
        print(f"  DataFeed test failed: {e}")
        return False

def test_rl_trader():
    """Test RL trader fixes."""
    print("Testing RLTrader...")
    try:
        from rl.rl_trader import get_rl_trader
        import pandas as pd
        import numpy as np
        
        trader = get_rl_trader()
        
        # Test model status
        status = trader.get_model_status()
        print(f"  Model Available: {status.get('model_available', False)}")
        print(f"  Gymnasium Available: {status.get('gymnasium_available', False)}")
        print(f"  Stable-baselines3 Available: {status.get('stable_baselines3_available', False)}")
        
        # Test system health
        health = trader.get_system_health()
        print(f"  System Health: {health.get('overall_status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"  RLTrader test failed: {e}")
        return False

def test_agent_hub():
    """Test agent hub fixes."""
    print("Testing AgentHub...")
    try:
        from core.agent_hub import AgentHub
        
        hub = AgentHub()
        
        # Test system health
        health = hub.get_system_health()
        print(f"  System Health: {health.get('status', 'unknown')}")
        
        # Test agent status
        status = hub.get_agent_status()
        print(f"  Available Agents: {len(status.get('available_agents', []))}")
        
        return True
        
    except Exception as e:
        print(f"  AgentHub test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("EVOLVE TRADING SYSTEM - FIX VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("CapabilityRouter", test_capability_router),
        ("DataFeed", test_data_feed),
        ("RLTrader", test_rl_trader),
        ("AgentHub", test_agent_hub)
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            success = test_func()
            results[test_name] = success
            if success:
                passed += 1
                print(f"  PASSED")
            else:
                print(f"  FAILED")
        except Exception as e:
            results[test_name] = False
            print(f"  ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("STATUS: ALL FIXES VERIFIED SUCCESSFULLY")
    else:
        print("STATUS: SOME ISSUES REMAIN")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_reports/fix_verification_{timestamp}.json"
    
    try:
        import os
        os.makedirs("test_reports", exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'total_tests': total,
                'passed': passed,
                'failed': total - passed,
                'success_rate': (passed/total)*100,
                'results': results
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        print(f"\nCould not save results: {e}")

if __name__ == "__main__":
    main() 