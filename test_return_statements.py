#!/usr/bin/env python3
"""
Test Return Statements

Comprehensive test to verify all functions return structured outputs.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_voice_prompt_agent():
    """Test voice prompt agent return statements."""
    print("🔊 Testing Voice Prompt Agent...")
    
    try:
        from voice_prompt_agent import VoicePromptAgent
        
        agent = VoicePromptAgent()
        
        # Test parameter extraction
        result = agent._extract_additional_parameters("Buy 100 shares of AAPL now", {})
        assert isinstance(result, dict) and "status" in result
        print("  ✅ _extract_additional_parameters returns structured output")
        
        # Test voice history update
        result = agent._update_voice_history({}, {})
        assert isinstance(result, dict) and "status" in result
        print("  ✅ _update_voice_history returns structured output")
        
        # Test clear voice history
        result = agent.clear_voice_history()
        assert isinstance(result, dict) and "status" in result
        print("  ✅ clear_voice_history returns structured output")
        
        return {"status": "voice_prompt_agent_tests_passed"}
        
    except Exception as e:
        return {"status": "voice_prompt_agent_tests_failed", "error": str(e)}

def test_system_status():
    """Test system status return statements."""
    print("📊 Testing System Status...")
    
    try:
        from utils.system_status import SystemStatus
        
        status = SystemStatus()
        
        # Test save status report
        result = status.save_status_report("test_status.json")
        assert isinstance(result, dict) and "status" in result
        print("  ✅ save_status_report returns structured output")
        
        # Test print status
        result = status.print_status()
        assert isinstance(result, dict) and "status" in result
        print("  ✅ print_status returns structured output")
        
        return {"status": "system_status_tests_passed"}
        
    except Exception as e:
        return {"status": "system_status_tests_failed", "error": str(e)}

def test_unified_interface():
    """Test unified interface return statements."""
    print("🔗 Testing Unified Interface...")
    
    try:
        from unified_interface import EnhancedUnifiedInterface
        
        interface = EnhancedUnifiedInterface()
        
        # Test component initialization
        result = interface._initialize_components()
        assert isinstance(result, dict) and "status" in result
        print("  ✅ _initialize_components returns structured output")
        
        # Test fallback initialization
        result = interface._initialize_fallback_components()
        assert isinstance(result, dict) and "status" in result
        print("  ✅ _initialize_fallback_components returns structured output")
        
        # Test logging setup
        result = interface._setup_logging()
        assert isinstance(result, dict) and "status" in result
        print("  ✅ _setup_logging returns structured output")
        
        return {"status": "unified_interface_tests_passed"}
        
    except Exception as e:
        return {"status": "unified_interface_tests_failed", "error": str(e)}

def test_rl_trader():
    """Test RL trader return statements."""
    print("🤖 Testing RL Trader...")
    
    try:
        from rl.rl_trader import TradingEnvironment
        import pandas as pd
        import numpy as np
        
        # Create test data
        data = pd.DataFrame({
            'Open': np.random.normal(100, 10, 100),
            'High': np.random.normal(105, 10, 100),
            'Low': np.random.normal(95, 10, 100),
            'Close': np.random.normal(100, 10, 100),
            'Volume': np.random.normal(1000000, 200000, 100)
        })
        
        env = TradingEnvironment(data)
        
        # Test portfolio value update
        result = env._update_portfolio_value()
        assert isinstance(result, dict) and "status" in result
        print("  ✅ _update_portfolio_value returns structured output")
        
        return {"status": "rl_trader_tests_passed"}
        
    except Exception as e:
        return {"status": "rl_trader_tests_failed", "error": str(e)}

def test_runner():
    """Test runner return statements."""
    print("🏃 Testing Runner...")
    
    try:
        from utils.runner import display_system_status
        
        # Test display system status
        result = display_system_status({})
        assert isinstance(result, dict) and "status" in result
        print("  ✅ display_system_status returns structured output")
        
        return {"status": "runner_tests_passed"}
        
    except Exception as e:
        return {"status": "runner_tests_failed", "error": str(e)}

def test_config_loader():
    """Test config loader return statements."""
    print("⚙️ Testing Config Loader...")
    
    try:
        from utils.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        
        # Check that __init__ sets status
        assert hasattr(loader, 'status') and isinstance(loader.status, dict)
        print("  ✅ __init__ sets status attribute")
        
        return {"status": "config_loader_tests_passed"}
        
    except Exception as e:
        return {"status": "config_loader_tests_failed", "error": str(e)}

def test_demo_interface():
    """Test demo interface return statements."""
    print("🎮 Testing Demo Interface...")
    
    try:
        from demo_unified_interface import demo_unified_interface, show_usage_examples, main
        
        # Test demo function
        result = demo_unified_interface()
        assert isinstance(result, dict) and "status" in result
        print("  ✅ demo_unified_interface returns structured output")
        
        # Test examples function
        result = show_usage_examples()
        assert isinstance(result, dict) and "status" in result
        print("  ✅ show_usage_examples returns structured output")
        
        # Test main function
        result = main()
        assert isinstance(result, dict) and "status" in result
        print("  ✅ main returns structured output")
        
        return {"status": "demo_interface_tests_passed"}
        
    except Exception as e:
        return {"status": "demo_interface_tests_failed", "error": str(e)}

def main():
    """Run all return statement tests."""
    print("🧪 COMPREHENSIVE RETURN STATEMENT TEST")
    print("=" * 60)
    
    tests = [
        test_voice_prompt_agent,
        test_system_status,
        test_unified_interface,
        test_rl_trader,
        test_runner,
        test_config_loader,
        test_demo_interface
    ]
    
    results = []
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = test()
            results.append(result)
            
            if result.get('status', '').endswith('_passed'):
                passed += 1
            else:
                failed += 1
                
        except Exception as e:
            failed += 1
            results.append({"status": f"{test.__name__}_failed", "error": str(e)})
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Return statement compliance: 100%")
        return {"status": "all_tests_passed", "passed": passed, "failed": failed, "success_rate": 100.0}
    else:
        print(f"\n⚠️ {failed} tests failed. Check individual results below.")
        for result in results:
            if not result.get('status', '').endswith('_passed'):
                print(f"  ❌ {result['status']}: {result.get('error', 'Unknown error')}")
        
        return {"status": "some_tests_failed", "passed": passed, "failed": failed, "success_rate": (passed/(passed+failed)*100)}

if __name__ == "__main__":
    result = main()
    print(f"\nFinal result: {result['status']}") 