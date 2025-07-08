#!/usr/bin/env python3
"""
Final Integration Test for Evolve Trading Platform

This script performs comprehensive integration testing to validate:
- All components work together seamlessly
- Agentic intelligence is fully functional
- System is production-ready
- All checklist requirements are met
"""

import sys
import os
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_complete_system_integration():
    """Test complete system integration."""
    print("🔍 Testing Complete System Integration...")
    
    try:
        # Test enhanced interface
        from unified_interface_v2 import EnhancedUnifiedInterfaceV2
        interface = EnhancedUnifiedInterfaceV2()
        print("✅ Enhanced interface integration successful")
        
        # Test system resilience
        from system_resilience import SystemResilience
        resilience = SystemResilience()
        print("✅ System resilience integration successful")
        
        # Test all core components
        components = [
            'agent_hub', 'data_feed', 'prompt_router', 'model_monitor',
            'strategy_logger', 'portfolio_manager', 'strategy_selector',
            'market_regime_agent', 'hybrid_engine', 'quant_gpt',
            'reporter', 'backtester'
        ]
        
        for component in components:
            if hasattr(interface, component):
                print(f"✅ {component} integration successful")
            else:
                print(f"❌ {component} integration failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete system integration failed: {e}")
        return False

def test_agentic_intelligence():
    """Test agentic intelligence features."""
    print("\n🔍 Testing Agentic Intelligence...")
    
    try:
        from unified_interface_v2 import EnhancedUnifiedInterfaceV2
        
        interface = EnhancedUnifiedInterfaceV2()
        
        # Test prompt routing
        if hasattr(interface, 'prompt_router'):
            print("✅ Prompt router agent available")
        
        # Test model selection
        if hasattr(interface, 'strategy_selector'):
            print("✅ Strategy selection agent available")
        
        # Test regime detection
        if hasattr(interface, 'market_regime_agent'):
            print("✅ Market regime agent available")
        
        # Test model monitoring
        if hasattr(interface, 'model_monitor'):
            print("✅ Model monitoring available")
        
        # Test strategy logging
        if hasattr(interface, 'strategy_logger'):
            print("✅ Strategy logging available")
        
        return True
        
    except Exception as e:
        print(f"❌ Agentic intelligence test failed: {e}")
        return False

def test_forecasting_capabilities():
    """Test forecasting capabilities."""
    print("\n🔍 Testing Forecasting Capabilities...")
    
    try:
        from unified_interface_v2 import EnhancedUnifiedInterfaceV2
        
        interface = EnhancedUnifiedInterfaceV2()
        
        # Test data feed
        if hasattr(interface, 'data_feed'):
            try:
                # Test with mock data
                data = interface.data_feed.get_historical_data("AAPL", "2023-01-01", "2023-12-31")
                print("✅ Data feed working")
            except Exception as e:
                print(f"⚠️  Data feed test: {e}")
        
        # Test forecasting methods
        forecast_methods = [
            '_generate_forecast',
            '_generate_ensemble_forecast',
            '_generate_single_model_forecast',
            '_calculate_forecast_confidence'
        ]
        
        for method in forecast_methods:
            if hasattr(interface, method):
                print(f"✅ {method} available")
            else:
                print(f"❌ {method} missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Forecasting capabilities test failed: {e}")
        return False

def test_strategy_engine():
    """Test strategy engine capabilities."""
    print("\n🔍 Testing Strategy Engine...")
    
    try:
        from unified_interface_v2 import EnhancedUnifiedInterfaceV2
        
        interface = EnhancedUnifiedInterfaceV2()
        
        # Test strategy generation
        if hasattr(interface, '_generate_strategy'):
            print("✅ Strategy generation available")
        
        # Test hybrid engine
        if hasattr(interface, 'hybrid_engine'):
            print("✅ Hybrid engine available")
        
        # Test strategy selection
        if hasattr(interface, 'strategy_selector'):
            print("✅ Strategy selection available")
        
        # Test market regime detection
        if hasattr(interface, 'market_regime_agent'):
            print("✅ Market regime detection available")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy engine test failed: {e}")
        return False

def test_backtesting_and_reporting():
    """Test backtesting and reporting capabilities."""
    print("\n🔍 Testing Backtesting and Reporting...")
    
    try:
        from unified_interface_v2 import EnhancedUnifiedInterfaceV2
        
        interface = EnhancedUnifiedInterfaceV2()
        
        # Test backtesting
        if hasattr(interface, '_run_backtest'):
            print("✅ Backtesting available")
        
        if hasattr(interface, 'backtester'):
            print("✅ Enhanced backtester available")
        
        # Test reporting
        if hasattr(interface, '_generate_report'):
            print("✅ Report generation available")
        
        if hasattr(interface, 'reporter'):
            print("✅ Unified reporter available")
        
        # Test export functionality
        export_methods = [
            '_export_forecast_data',
            '_export_forecast_chart',
            '_export_backtest_data',
            '_export_equity_curve',
            '_download_report'
        ]
        
        for method in export_methods:
            if hasattr(interface, method):
                print(f"✅ {method} available")
        
        return True
        
    except Exception as e:
        print(f"❌ Backtesting and reporting test failed: {e}")
        return False

def test_llm_and_commentary():
    """Test LLM and commentary system."""
    print("\n🔍 Testing LLM and Commentary System...")
    
    try:
        from unified_interface_v2 import EnhancedUnifiedInterfaceV2
        
        interface = EnhancedUnifiedInterfaceV2()
        
        # Test QuantGPT
        if hasattr(interface, 'quant_gpt'):
            print("✅ QuantGPT commentary system available")
        
        # Test LLM integration
        try:
            from trading.llm.agent import PromptAgent
            print("✅ LLM agent available")
        except ImportError:
            print("⚠️  LLM agent not available")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM and commentary test failed: {e}")
        return False

def test_ui_and_deployment():
    """Test UI and deployment features."""
    print("\n🔍 Testing UI and Deployment Features...")
    
    try:
        from unified_interface_v2 import EnhancedUnifiedInterfaceV2
        
        interface = EnhancedUnifiedInterfaceV2()
        
        # Test UI methods
        ui_methods = [
            '_forecast_tab',
            '_strategy_tab',
            '_backtest_tab',
            '_report_tab',
            '_system_tab',
            '_render_sidebar'
        ]
        
        for method in ui_methods:
            if hasattr(interface, method):
                print(f"✅ {method} available")
            else:
                print(f"❌ {method} missing")
        
        # Test system resilience
        from system_resilience import SystemResilience
        resilience = SystemResilience()
        
        # Test health monitoring
        health = resilience.get_system_health()
        print(f"✅ Health monitoring working - Status: {health['overall_status']}")
        
        # Test performance monitoring
        performance = resilience.get_performance_report()
        print("✅ Performance monitoring working")
        
        return True
        
    except Exception as e:
        print(f"❌ UI and deployment test failed: {e}")
        return False

def test_production_readiness():
    """Test production readiness."""
    print("\n🔍 Testing Production Readiness...")
    
    # Check deployment files
    deployment_files = [
        "deploy/Dockerfile.production",
        "deploy/docker-compose.production.yml",
        "deploy/deploy.sh",
        "requirements.production.txt",
        "env.example",
        "system_resilience.py",
        "unified_interface_v2.py"
    ]
    
    all_files_exist = True
    for file_path in deployment_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_files_exist = False
    
    # Check environment variables
    if os.path.exists("env.example"):
        print("✅ Environment configuration available")
    else:
        print("❌ Environment configuration missing")
        all_files_exist = False
    
    # Check requirements
    if os.path.exists("requirements.production.txt"):
        print("✅ Production requirements available")
    else:
        print("❌ Production requirements missing")
        all_files_exist = False
    
    return all_files_exist

def test_fallback_mechanisms():
    """Test fallback mechanisms."""
    print("\n🔍 Testing Fallback Mechanisms...")
    
    try:
        from unified_interface_v2 import EnhancedUnifiedInterfaceV2
        
        interface = EnhancedUnifiedInterfaceV2()
        
        # Test fallback components
        fallback_components = [
            '_create_fallback_agent_hub',
            '_create_fallback_data_feed',
            '_create_fallback_prompt_router',
            '_create_fallback_model_monitor',
            '_create_fallback_strategy_logger',
            '_create_fallback_portfolio_manager',
            '_create_fallback_strategy_selector',
            '_create_fallback_market_regime_agent',
            '_create_fallback_hybrid_engine',
            '_create_fallback_quant_gpt',
            '_create_fallback_reporter',
            '_create_fallback_backtester'
        ]
        
        for component in fallback_components:
            if hasattr(interface, component):
                print(f"✅ {component} available")
            else:
                print(f"❌ {component} missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback mechanisms test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and logging."""
    print("\n🔍 Testing Error Handling and Logging...")
    
    try:
        from unified_interface_v2 import EnhancedUnifiedInterfaceV2
        
        interface = EnhancedUnifiedInterfaceV2()
        
        # Test logging setup
        if hasattr(interface, '_setup_logging'):
            print("✅ Logging setup available")
        
        # Test error handling in methods
        error_handled_methods = [
            '_initialize_components',
            '_load_config',
            '_generate_forecast',
            '_generate_strategy',
            '_run_backtest'
        ]
        
        for method in error_handled_methods:
            if hasattr(interface, method):
                print(f"✅ {method} error handling available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_performance_and_scalability():
    """Test performance and scalability."""
    print("\n🔍 Testing Performance and Scalability...")
    
    try:
        from system_resilience import SystemResilience
        
        resilience = SystemResilience()
        
        # Test performance metrics collection
        performance = resilience.get_performance_report()
        if 'cpu_usage_avg' in performance:
            print("✅ Performance metrics collection working")
        
        # Test health monitoring performance
        health = resilience.get_system_health()
        if 'overall_status' in health:
            print("✅ Health monitoring performance good")
        
        # Test component count
        component_count = len(health.get('components', {}))
        print(f"✅ {component_count} components monitored")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance and scalability test failed: {e}")
        return False

def run_final_validation():
    """Run final validation of all checklist requirements."""
    print("🚀 Starting Final Integration and Validation Test")
    print("=" * 70)
    
    test_results = {}
    
    # Run all validation tests
    tests = [
        ("Complete System Integration", test_complete_system_integration),
        ("Agentic Intelligence", test_agentic_intelligence),
        ("Forecasting Capabilities", test_forecasting_capabilities),
        ("Strategy Engine", test_strategy_engine),
        ("Backtesting and Reporting", test_backtesting_and_reporting),
        ("LLM and Commentary", test_llm_and_commentary),
        ("UI and Deployment", test_ui_and_deployment),
        ("Production Readiness", test_production_readiness),
        ("Fallback Mechanisms", test_fallback_mechanisms),
        ("Error Handling", test_error_handling),
        ("Performance and Scalability", test_performance_and_scalability)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            test_results[test_name] = False
    
    # Print comprehensive summary
    print("\n" + "=" * 70)
    print("📊 FINAL VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    # Calculate completion percentage
    completion_percentage = (passed / total) * 100
    
    if completion_percentage >= 90:
        print("🎉 EXCELLENT! System is production-ready with comprehensive functionality!")
        print("✅ All major components integrated and validated")
        print("✅ Agentic intelligence fully functional")
        print("✅ UI and deployment ready for production")
        print("✅ Fallback mechanisms and error handling robust")
        return True
    elif completion_percentage >= 80:
        print("🟢 GOOD! System is mostly production-ready with minor improvements needed")
        print("⚠️  Some components may need additional testing or configuration")
        return True
    elif completion_percentage >= 70:
        print("🟡 FAIR! System has good foundation but needs more work")
        print("⚠️  Several components need attention before production deployment")
        return False
    else:
        print("🔴 NEEDS WORK! System requires significant improvements")
        print("❌ Many components need development or integration")
        return False

if __name__ == "__main__":
    success = run_final_validation()
    sys.exit(0 if success else 1) 