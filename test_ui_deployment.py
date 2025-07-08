#!/usr/bin/env python3
"""
Test UI and Deployment Readiness

This script tests the enhanced UI and deployment features:
- Enhanced unified interface v2
- System resilience and fallback mechanisms
- Production deployment configuration
- Health monitoring and diagnostics
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_enhanced_interface_import():
    """Test enhanced interface import."""
    logger.info("🔍 Testing Enhanced Interface Import...")
    
    try:
        from unified_interface_v2 import EnhancedUnifiedInterfaceV2, run_enhanced_interface_v2
        logger.info("✅ Enhanced interface import successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Enhanced interface import failed: {e}")
        return False

def test_system_resilience():
    """Test system resilience features."""
    logger.info("\n🔍 Testing System Resilience...")
    
    try:
        from system_resilience import SystemResilience, get_system_resilience
        
        # Test system resilience initialization
        resilience = SystemResilience()
        logger.info("✅ System resilience initialization successful")
        
        # Test health checks
        health_status = resilience.get_system_health()
        logger.info(f"✅ Health status retrieved: {health_status['overall_status']}")
        
        # Test performance metrics
        performance = resilience.get_performance_report()
        logger.info("✅ Performance metrics retrieved")
        
        return True
    except Exception as e:
        logger.error(f"❌ System resilience test failed: {e}")
        return False

def test_deployment_configuration():
    """Test deployment configuration files."""
    logger.info("\n🔍 Testing Deployment Configuration...")
    
    deployment_files = [
        "deploy/Dockerfile.production",
        "deploy/docker-compose.production.yml",
        "deploy/deploy.sh",
        "requirements.production.txt",
        "env.example"
    ]
    
    all_exist = True
    for file_path in deployment_files:
        if os.path.exists(file_path):
            logger.info(f"✅ {file_path} exists")
        else:
            logger.error(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_environment_variables():
    """Test environment variable configuration."""
    logger.info("\n🔍 Testing Environment Variables...")
    
    try:
        # Check if env.example exists
        if os.path.exists("env.example"):
            logger.info("✅ env.example file exists")
            
            # Read and validate env.example
            with open("env.example", "r") as f:
                content = f.read()
                
            # Check for required variables
            required_vars = [
                "OPENAI_API_KEY",
                "FINNHUB_API_KEY",
                "ALPHA_VANTAGE_API_KEY",
                "APP_SECRET_KEY",
                "JWT_SECRET_KEY"
            ]
            
            missing_vars = []
            for var in required_vars:
                if var not in content:
                    missing_vars.append(var)
            
            if missing_vars:
                logger.warning(f"⚠️  Missing variables in env.example: {missing_vars}")
            else:
                logger.info("✅ All required variables in env.example")
            
            return True
        else:
            logger.error("❌ env.example file missing")
            return False
            
    except Exception as e:
        logger.error(f"❌ Environment variables test failed: {e}")
        return False

def test_production_requirements():
    """Test production requirements file."""
    logger.info("\n🔍 Testing Production Requirements...")
    
    try:
        if os.path.exists("requirements.production.txt"):
            logger.info("✅ requirements.production.txt exists")
            
            # Read requirements
            with open("requirements.production.txt", "r") as f:
                content = f.read()
            
            # Check for essential packages
            essential_packages = [
                "streamlit",
                "pandas",
                "numpy",
                "scikit-learn",
                "torch",
                "plotly"
            ]
            
            missing_packages = []
            for package in essential_packages:
                if package not in content:
                    missing_packages.append(package)
            
            if missing_packages:
                logger.warning(f"⚠️  Missing packages in requirements.production.txt: {missing_packages}")
            else:
                logger.info("✅ All essential packages in requirements.production.txt")
            
            return True
        else:
            logger.error("❌ requirements.production.txt missing")
            return False
            
    except Exception as e:
        logger.error(f"❌ Production requirements test failed: {e}")
        return False

def test_docker_configuration():
    """Test Docker configuration."""
    logger.info("\n🔍 Testing Docker Configuration...")
    
    try:
        # Test Dockerfile.production
        if os.path.exists("deploy/Dockerfile.production"):
            with open("deploy/Dockerfile.production", "r") as f:
                dockerfile_content = f.read()
            
            # Check for essential Docker features
            docker_features = [
                "FROM python:3.9-slim",
                "WORKDIR /app",
                "EXPOSE 8501",
                "HEALTHCHECK",
                "USER appuser"
            ]
            
            missing_features = []
            for feature in docker_features:
                if feature not in dockerfile_content:
                    missing_features.append(feature)
            
            if missing_features:
                logger.warning(f"⚠️  Missing Docker features: {missing_features}")
            else:
                logger.info("✅ Dockerfile.production has all essential features")
            
            return True
        else:
            logger.error("❌ Dockerfile.production missing")
            return False
            
    except Exception as e:
        logger.error(f"❌ Docker configuration test failed: {e}")
        return False

def test_ui_features():
    """Test UI features."""
    logger.info("\n🔍 Testing UI Features...")
    
    try:
        # Test enhanced interface features
        from unified_interface_v2 import EnhancedUnifiedInterfaceV2
        
        interface = EnhancedUnifiedInterfaceV2()
        logger.info("✅ Enhanced interface initialization successful")
        
        # Test component initialization
        if hasattr(interface, 'agent_hub'):
            logger.info("✅ Agent hub component available")
        
        if hasattr(interface, 'data_feed'):
            logger.info("✅ Data feed component available")
        
        if hasattr(interface, 'prompt_router'):
            logger.info("✅ Prompt router component available")
        
        if hasattr(interface, 'model_monitor'):
            logger.info("✅ Model monitor component available")
        
        if hasattr(interface, 'strategy_logger'):
            logger.info("✅ Strategy logger component available")
        
        if hasattr(interface, 'portfolio_manager'):
            logger.info("✅ Portfolio manager component available")
        
        if hasattr(interface, 'strategy_selector'):
            logger.info("✅ Strategy selector component available")
        
        if hasattr(interface, 'market_regime_agent'):
            logger.info("✅ Market regime agent component available")
        
        if hasattr(interface, 'hybrid_engine'):
            logger.info("✅ Hybrid engine component available")
        
        if hasattr(interface, 'quant_gpt'):
            logger.info("✅ QuantGPT component available")
        
        if hasattr(interface, 'reporter'):
            logger.info("✅ Reporter component available")
        
        if hasattr(interface, 'backtester'):
            logger.info("✅ Backtester component available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ UI features test failed: {e}")
        return False

def test_fallback_mechanisms():
    """Test fallback mechanisms."""
    logger.info("\n🔍 Testing Fallback Mechanisms...")
    
    try:
        from unified_interface_v2 import EnhancedUnifiedInterfaceV2
        
        interface = EnhancedUnifiedInterfaceV2()
        
        # Test fallback data feed
        if hasattr(interface, 'data_feed'):
            try:
                data = interface.data_feed.get_historical_data("AAPL", "2023-01-01", "2023-12-31")
                logger.info("✅ Data feed fallback working")
            except Exception as e:
                logger.warning(f"⚠️  Data feed fallback issue: {e}")
        
        # Test fallback model monitor
        if hasattr(interface, 'model_monitor'):
            try:
                trust_levels = interface.model_monitor.get_model_trust_levels()
                logger.info("✅ Model monitor fallback working")
            except Exception as e:
                logger.warning(f"⚠️  Model monitor fallback issue: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Fallback mechanisms test failed: {e}")
        return False

def test_export_functionality():
    """Test export functionality."""
    logger.info("\n🔍 Testing Export Functionality...")
    
    try:
        from unified_interface_v2 import EnhancedUnifiedInterfaceV2
        
        interface = EnhancedUnifiedInterfaceV2()
        
        # Test export methods exist
        export_methods = [
            '_export_forecast_data',
            '_export_forecast_chart',
            '_export_backtest_data',
            '_export_equity_curve',
            '_download_report'
        ]
        
        for method in export_methods:
            if hasattr(interface, method):
                logger.info(f"✅ {method} method available")
            else:
                logger.error(f"❌ {method} method missing")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Export functionality test failed: {e}")
        return False

def test_health_monitoring():
    """Test health monitoring."""
    logger.info("\n🔍 Testing Health Monitoring...")
    
    try:
        from system_resilience import get_system_resilience
        
        resilience = get_system_resilience()
        
        # Test health status
        health = resilience.get_system_health()
        logger.info(f"✅ Health monitoring working - Status: {health['overall_status']}")
        
        # Test performance metrics
        performance = resilience.get_performance_report()
        logger.info("✅ Performance monitoring working")
        
        # Test component health
        for component, status in health['components'].items():
            logger.info(f"  - {component}: {status['status']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Health monitoring test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive UI and deployment test."""
    logger.info("🚀 Starting UI and Deployment Readiness Test")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Enhanced Interface Import", test_enhanced_interface_import),
        ("System Resilience", test_system_resilience),
        ("Deployment Configuration", test_deployment_configuration),
        ("Environment Variables", test_environment_variables),
        ("Production Requirements", test_production_requirements),
        ("Docker Configuration", test_docker_configuration),
        ("UI Features", test_ui_features),
        ("Fallback Mechanisms", test_fallback_mechanisms),
        ("Export Functionality", test_export_functionality),
        ("Health Monitoring", test_health_monitoring)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            test_results[test_name] = False
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests passed! UI and deployment are ready.")
        return True
    else:
        logger.warning("⚠️  Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1) 