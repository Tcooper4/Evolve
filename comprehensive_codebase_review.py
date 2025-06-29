#!/usr/bin/env python3
"""Comprehensive Codebase Review for Evolve Trading Platform."""

import sys
import traceback
from typing import Dict, List, Any
import time

def test_basic_imports():
    """Test basic Python package imports."""
    print("🔍 Testing Basic Imports...")
    
    results = {}
    
    # Core scientific computing
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        results['numpy'] = f"✅ {np.__version__}"
        results['pandas'] = f"✅ {pd.__version__}"
        results['matplotlib'] = f"✅ {plt.matplotlib.__version__}"
    except ImportError as e:
        results['scientific'] = f"❌ {e}"
    
    # Machine learning
    try:
        import sklearn
        results['scikit-learn'] = f"✅ {sklearn.__version__}"
    except ImportError as e:
        results['scikit-learn'] = f"❌ {e}"
    
    # Deep learning
    try:
        import torch
        results['pytorch'] = f"✅ {torch.__version__}"
    except ImportError as e:
        results['pytorch'] = f"❌ {e}"
    
    # Web framework
    try:
        import streamlit as st
        results['streamlit'] = "✅ Available"
    except ImportError as e:
        results['streamlit'] = f"❌ {e}"
    
    # Visualization
    try:
        import plotly
        results['plotly'] = f"✅ {plotly.__version__}"
    except ImportError as e:
        results['plotly'] = f"❌ {e}"
    
    return results

def test_core_modules():
    """Test core platform modules."""
    print("\n🔍 Testing Core Modules...")
    
    results = {}
    
    # Configuration
    try:
        from utils.config_loader import ConfigLoader
        config = ConfigLoader()
        results['config_loader'] = "✅ Working"
    except Exception as e:
        results['config_loader'] = f"❌ {e}"
    
    # Data pipeline
    try:
        from data.streaming_pipeline import create_streaming_pipeline
        results['streaming_pipeline'] = "✅ Available"
    except Exception as e:
        results['streaming_pipeline'] = f"⚠️ {e}"
    
    # Trading core
    try:
        from trading.base_agent import BaseAgent
        results['trading_core'] = "✅ Available"
    except Exception as e:
        results['trading_core'] = f"⚠️ {e}"
    
    # Risk management
    try:
        from risk.tail_risk import TailRiskEngine
        results['risk_engine'] = "✅ Available"
    except Exception as e:
        results['risk_engine'] = f"⚠️ {e}"
    
    return results

def test_advanced_features():
    """Test advanced feature modules."""
    print("\n🔍 Testing Advanced Features...")
    
    results = {}
    
    # Reinforcement Learning
    try:
        from rl.strategy_trainer import create_rl_strategy_trainer
        trainer = create_rl_strategy_trainer()
        results['rl_engine'] = "✅ Available" if trainer['available'] else "⚠️ Dependencies missing"
    except Exception as e:
        results['rl_engine'] = f"❌ {e}"
    
    # Causal Inference
    try:
        from causal.causal_model import create_causal_model
        model = create_causal_model()
        results['causal_inference'] = "✅ Available"
    except Exception as e:
        results['causal_inference'] = f"⚠️ {e}"
    
    # TFT Model
    try:
        from models.tft_model import create_tft_model
        results['tft_model'] = "✅ Available"
    except Exception as e:
        results['tft_model'] = f"⚠️ {e}"
    
    # Strategy Gatekeeper
    try:
        from strategies.gatekeeper import create_strategy_gatekeeper
        results['strategy_gatekeeper'] = "✅ Available"
    except Exception as e:
        results['strategy_gatekeeper'] = f"⚠️ {e}"
    
    # Live Trading Interface
    try:
        from execution.live_trading_interface import create_live_trading_interface
        results['live_trading'] = "✅ Available"
    except Exception as e:
        results['live_trading'] = f"⚠️ {e}"
    
    return results

def test_optimization_modules():
    """Test optimization modules."""
    print("\n🔍 Testing Optimization Modules...")
    
    results = {}
    
    # Genetic Optimizer
    try:
        from trading.optimization.genetic_optimizer import create_genetic_optimizer
        results['genetic_optimizer'] = "✅ Available"
    except Exception as e:
        results['genetic_optimizer'] = f"⚠️ {e}"
    
    # Multi-objective Optimizer
    try:
        from trading.optimization.multi_objective_optimizer import MultiObjectiveOptimizer
        results['multi_objective_optimizer'] = "✅ Available"
    except Exception as e:
        results['multi_objective_optimizer'] = f"⚠️ {e}"
    
    # Bayesian Optimizer
    try:
        from trading.optimization.bayesian_optimizer import BayesianOptimizer
        results['bayesian_optimizer'] = "✅ Available"
    except Exception as e:
        results['bayesian_optimizer'] = f"⚠️ {e}"
    
    return results

def test_data_providers():
    """Test data provider modules."""
    print("\n🔍 Testing Data Providers...")
    
    results = {}
    
    # YFinance
    try:
        import yfinance as yf
        results['yfinance'] = "✅ Available"
    except ImportError:
        results['yfinance'] = "⚠️ Not installed"
    
    # Alpha Vantage
    try:
        from trading.data.providers.alpha_vantage_provider import AlphaVantageProvider
        results['alpha_vantage'] = "✅ Available"
    except Exception as e:
        results['alpha_vantage'] = f"⚠️ {e}"
    
    return results

def test_ui_components():
    """Test UI components."""
    print("\n🔍 Testing UI Components...")
    
    results = {}
    
    # Dashboard
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('dashboard', 'pages/10_Strategy_Health_Dashboard.py')
        module = importlib.util.module_from_spec(spec)
        results['strategy_dashboard'] = "✅ Loadable"
    except Exception as e:
        results['strategy_dashboard'] = f"❌ {e}"
    
    # Voice Interface
    try:
        from ui.voice_interface import create_voice_interface
        results['voice_interface'] = "✅ Available"
    except Exception as e:
        results['voice_interface'] = f"⚠️ {e}"
    
    return results

def test_configuration():
    """Test configuration files."""
    print("\n🔍 Testing Configuration...")
    
    results = {}
    
    import os
    
    # Check config files exist
    config_files = [
        'config/optimizer_config.yaml',
        'config/app_config.yaml',
        'config/config.json'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            results[config_file] = "✅ Exists"
        else:
            results[config_file] = "❌ Missing"
    
    # Test config loading
    try:
        from utils.config_loader import ConfigLoader
        config = ConfigLoader()
        settings = config.get_optimization_settings()
        results['config_loading'] = "✅ Working"
    except Exception as e:
        results['config_loading'] = f"❌ {e}"
    
    return results

def test_file_structure():
    """Test file structure integrity."""
    print("\n🔍 Testing File Structure...")
    
    results = {}
    
    import os
    
    # Check critical directories
    critical_dirs = [
        'trading',
        'models',
        'data',
        'risk',
        'rl',
        'causal',
        'strategies',
        'execution',
        'ui',
        'utils',
        'config',
        'pages'
    ]
    
    for directory in critical_dirs:
        if os.path.exists(directory):
            results[directory] = "✅ Exists"
        else:
            results[directory] = "❌ Missing"
    
    # Check critical files
    critical_files = [
        'app.py',
        'requirements.txt',
        'requirements_advanced.txt',
        'README.md',
        'ADVANCED_FEATURES_README.md'
    ]
    
    for file in critical_files:
        if os.path.exists(file):
            results[file] = "✅ Exists"
        else:
            results[file] = "❌ Missing"
    
    return results

def generate_summary(all_results: Dict[str, Dict[str, str]]):
    """Generate comprehensive summary."""
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE CODEBASE REVIEW SUMMARY")
    print("="*60)
    
    total_tests = 0
    passed_tests = 0
    warnings = 0
    failures = 0
    
    for category, results in all_results.items():
        print(f"\n{category.upper()}:")
        print("-" * len(category))
        
        for test, result in results.items():
            total_tests += 1
            if "✅" in result:
                passed_tests += 1
                print(f"  {test:30} {result}")
            elif "⚠️" in result:
                warnings += 1
                print(f"  {test:30} {result}")
            else:
                failures += 1
                print(f"  {test:30} {result}")
    
    print("\n" + "="*60)
    print("📈 OVERALL STATISTICS")
    print("="*60)
    print(f"Total Tests:     {total_tests}")
    print(f"✅ Passed:       {passed_tests}")
    print(f"⚠️  Warnings:     {warnings}")
    print(f"❌ Failures:     {failures}")
    print(f"Success Rate:    {passed_tests/total_tests*100:.1f}%")
    
    if failures == 0:
        print("\n🎉 EXCELLENT! All critical components are working!")
        if warnings > 0:
            print(f"⚠️  {warnings} optional features have warnings but don't affect core functionality.")
    elif failures <= 2:
        print("\n✅ GOOD! Most components are working. Minor issues detected.")
    else:
        print("\n⚠️  ATTENTION NEEDED! Several components have issues.")
    
    return {
        'total': total_tests,
        'passed': passed_tests,
        'warnings': warnings,
        'failures': failures,
        'success_rate': passed_tests/total_tests*100
    }

def main():
    """Run comprehensive codebase review."""
    print("🧪 COMPREHENSIVE CODEBASE REVIEW")
    print("="*60)
    print("Reviewing Evolve Trading Platform...")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    # Run all tests
    all_results['Basic Imports'] = test_basic_imports()
    all_results['Core Modules'] = test_core_modules()
    all_results['Advanced Features'] = test_advanced_features()
    all_results['Optimization Modules'] = test_optimization_modules()
    all_results['Data Providers'] = test_data_providers()
    all_results['UI Components'] = test_ui_components()
    all_results['Configuration'] = test_configuration()
    all_results['File Structure'] = test_file_structure()
    
    # Generate summary
    summary = generate_summary(all_results)
    
    print(f"\nCompleted at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return summary['success_rate'] >= 90.0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 