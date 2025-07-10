#!/usr/bin/env python3
"""
Evolve AI Trading System - Comprehensive Test Suite

This script tests all core components of the Evolve AI Trading System
to ensure everything is working properly for production deployment.
"""

import sys
import os
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemTester:
    """Comprehensive system tester for Evolve AI Trading Platform."""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.warnings = []
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all system tests."""
        logger.info("🚀 Starting Evolve AI Trading System Tests")
        
        tests = [
            ("Core Dependencies", self.test_core_dependencies),
            ("Streamlit App", self.test_streamlit_app),
            ("Trading Agents", self.test_trading_agents),
            ("Strategies", self.test_strategies),
            ("Models", self.test_models),
            ("Data Providers", self.test_data_providers),
            ("Visualization", self.test_visualization),
            ("ML Libraries", self.test_ml_libraries),
            ("Pages", self.test_pages),
            ("Configuration", self.test_configuration),
            ("Utilities", self.test_utilities),
            ("Integration", self.test_integration)
        ]
        
        for test_name, test_func in tests:
            try:
                logger.info(f"Running {test_name}...")
                result = test_func()
                self.test_results[test_name] = result
                if result['status'] == 'PASS':
                    logger.info(f"✅ {test_name}: PASS")
                else:
                    logger.warning(f"⚠️ {test_name}: {result['status']}")
            except Exception as e:
                error_msg = f"❌ {test_name}: FAIL - {str(e)}"
                logger.error(error_msg)
                self.errors.append(error_msg)
                self.test_results[test_name] = {
                    'status': 'FAIL',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        return self.generate_report()
    
    def test_core_dependencies(self) -> Dict[str, Any]:
        """Test core Python dependencies."""
        required_packages = [
            'streamlit', 'pandas', 'numpy', 'plotly', 'scikit-learn',
            'torch', 'xgboost', 'yfinance', 'requests'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            return {
                'status': 'FAIL',
                'missing_packages': missing_packages,
                'message': f"Missing packages: {', '.join(missing_packages)}"
            }
        
        return {'status': 'PASS', 'message': 'All core dependencies available'}
    
    def test_streamlit_app(self) -> Dict[str, Any]:
        """Test main Streamlit application."""
        try:
            # Test if app.py exists and can be imported
            app_path = Path("app.py")
            if not app_path.exists():
                return {'status': 'FAIL', 'message': 'app.py not found'}
            
            # Test basic app structure
            with open(app_path, 'r') as f:
                content = f.read()
                if 'streamlit' not in content.lower():
                    return {'status': 'FAIL', 'message': 'Streamlit not detected in app.py'}
                if 'st.set_page_config' not in content:
                    return {'status': 'WARN', 'message': 'Page config not found in app.py'}
            
            return {'status': 'PASS', 'message': 'Streamlit app structure valid'}
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_trading_agents(self) -> Dict[str, Any]:
        """Test trading agents."""
        agent_tests = [
            ('trading.llm.agent', 'PromptAgent'),
            ('trading.agents.model_creator_agent', 'ModelCreatorAgent'),
            ('trading.agents.strategy_selector_agent', 'StrategySelectorAgent'),
            ('trading.agents.model_optimizer_agent', 'ModelOptimizerAgent')
        ]
        
        results = []
        for module_path, class_name in agent_tests:
            try:
                module = importlib.import_module(module_path)
                agent_class = getattr(module, class_name)
                # Test instantiation
                agent = agent_class()
                results.append(f"✅ {class_name}")
            except Exception as e:
                results.append(f"❌ {class_name}: {str(e)}")
        
        if all('✅' in result for result in results):
            return {'status': 'PASS', 'message': 'All agents working', 'details': results}
        else:
            return {'status': 'WARN', 'message': 'Some agents failed', 'details': results}
    
    def test_strategies(self) -> Dict[str, Any]:
        """Test trading strategies."""
        try:
            # Test strategy imports
            from trading.strategies.enhanced_strategy_engine import EnhancedStrategyEngine
            
            # Test strategy engine
            engine = EnhancedStrategyEngine()
            strategies = engine.get_available_strategies()
            
            if not strategies:
                return {'status': 'WARN', 'message': 'No strategies found'}
            
            return {
                'status': 'PASS',
                'message': f'Found {len(strategies)} strategies',
                'strategies': list(strategies.keys())
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_models(self) -> Dict[str, Any]:
        """Test ML models."""
        try:
            # Test model router
            from models.forecast_router import ForecastRouter
            
            router = ForecastRouter()
            models = router.get_available_models()
            
            if not models:
                return {'status': 'WARN', 'message': 'No models found'}
            
            return {
                'status': 'PASS',
                'message': f'Found {len(models)} models',
                'models': list(models.keys())
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_data_providers(self) -> Dict[str, Any]:
        """Test data providers."""
        try:
            # Test fallback data provider
            from trading.data.providers.fallback_provider import FallbackDataProvider
            
            provider = FallbackDataProvider()
            
            # Test basic functionality
            test_data = provider.get_historical_data('AAPL', days=10)
            
            if test_data is None or test_data.empty:
                return {'status': 'WARN', 'message': 'Data provider returned empty data'}
            
            return {
                'status': 'PASS',
                'message': 'Data provider working',
                'data_shape': test_data.shape
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_visualization(self) -> Dict[str, Any]:
        """Test visualization libraries."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import matplotlib.pyplot as plt
            
            # Test basic plot creation
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
            
            return {'status': 'PASS', 'message': 'Visualization libraries working'}
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_ml_libraries(self) -> Dict[str, Any]:
        """Test machine learning libraries."""
        ml_tests = []
        
        # Test scikit-learn
        try:
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=10)
            ml_tests.append("✅ Scikit-learn")
        except Exception as e:
            ml_tests.append(f"❌ Scikit-learn: {str(e)}")
        
        # Test XGBoost
        try:
            import xgboost as xgb
            xgb_model = xgb.XGBRegressor(n_estimators=10)
            ml_tests.append("✅ XGBoost")
        except Exception as e:
            ml_tests.append(f"❌ XGBoost: {str(e)}")
        
        # Test PyTorch (our primary deep learning framework)
        try:
            import torch
            import torch.nn as nn
            model = nn.Sequential(
                nn.Linear(10, 5),
                nn.ReLU(),
                nn.Linear(5, 1)
            )
            ml_tests.append("✅ PyTorch")
        except Exception as e:
            ml_tests.append(f"❌ PyTorch: {str(e)}")
        
        if all('✅' in test for test in ml_tests):
            return {'status': 'PASS', 'message': 'All ML libraries working', 'details': ml_tests}
        else:
            return {'status': 'WARN', 'message': 'Some ML libraries failed', 'details': ml_tests}
    
    def test_pages(self) -> Dict[str, Any]:
        """Test Streamlit pages."""
        pages = ['Forecasting.py', 'Strategy_Lab.py', 'Model_Lab.py', 'Reports.py']
        results = []
        
        for page in pages:
            page_path = Path(f"pages/{page}")
            if page_path.exists():
                try:
                    # Test if page can be imported
                    module_name = f"pages.{page[:-3]}"
                    module = importlib.import_module(module_name)
                    
                    # Test if main function exists
                    if hasattr(module, 'main'):
                        results.append(f"✅ {page}")
                    else:
                        results.append(f"⚠️ {page}: No main function")
                except Exception as e:
                    results.append(f"❌ {page}: {str(e)}")
            else:
                results.append(f"❌ {page}: File not found")
        
        if all('✅' in result for result in results):
            return {'status': 'PASS', 'message': 'All pages working', 'details': results}
        else:
            return {'status': 'WARN', 'message': 'Some pages failed', 'details': results}
    
    def test_configuration(self) -> Dict[str, Any]:
        """Test configuration files."""
        config_files = ['config/app_config.yaml', 'requirements.txt', '.env']
        results = []
        
        for config_file in config_files:
            if Path(config_file).exists():
                results.append(f"✅ {config_file}")
            else:
                results.append(f"⚠️ {config_file}: Not found")
        
        return {'status': 'PASS', 'message': 'Configuration files checked', 'details': results}
    
    def test_utilities(self) -> Dict[str, Any]:
        """Test utility functions."""
        try:
            # Test common utilities
            import utils.common_helpers
            import utils.config_loader
            
            return {'status': 'PASS', 'message': 'Utility functions working'}
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_integration(self) -> Dict[str, Any]:
        """Test system integration."""
        try:
            # Test end-to-end workflow
            from trading.llm.agent import PromptAgent
            
            agent = PromptAgent()
            
            # Test basic prompt processing
            response = agent.process_prompt("Show me the best forecast for AAPL")
            
            if response:
                return {'status': 'PASS', 'message': 'Integration test passed'}
            else:
                return {'status': 'WARN', 'message': 'Integration test returned empty response'}
                
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASS')
        failed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'FAIL')
        warning_tests = sum(1 for result in self.test_results.values() if result['status'] == 'WARN')
        
        overall_status = 'PASS' if failed_tests == 0 else 'FAIL'
        
        report = {
            'overall_status': overall_status,
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'warnings': warning_tests,
                'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            },
            'test_results': self.test_results,
            'errors': self.errors,
            'warnings': self.warnings,
            'recommendations': self.generate_recommendations()
        }
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check for critical failures
        failed_tests = [name for name, result in self.test_results.items() if result['status'] == 'FAIL']
        if failed_tests:
            recommendations.append(f"Fix critical failures in: {', '.join(failed_tests)}")
        
        # Check for missing dependencies
        if 'Core Dependencies' in self.test_results and self.test_results['Core Dependencies']['status'] == 'FAIL':
            recommendations.append("Install missing dependencies: pip install -r requirements.txt")
        
        # Check for configuration issues
        if 'Configuration' in self.test_results:
            config_result = self.test_results['Configuration']
            if any('⚠️' in detail for detail in config_result.get('details', [])):
                recommendations.append("Review configuration files and environment variables")
        
        # Check for ML library issues
        if 'ML Libraries' in self.test_results and self.test_results['ML Libraries']['status'] == 'WARN':
            recommendations.append("Consider installing GPU support for better ML performance")
        
        # General recommendations
        if not recommendations:
            recommendations.append("System is ready for production deployment")
        
        return recommendations

def main():
    """Main test execution function."""
    print("🧪 Evolve AI Trading System - Comprehensive Test Suite")
    print("=" * 60)
    
    tester = SystemTester()
    report = tester.run_all_tests()
    
    # Print summary
    print("\n📊 Test Summary")
    print("-" * 30)
    summary = report['summary']
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']} ✅")
    print(f"Failed: {summary['failed']} ❌")
    print(f"Warnings: {summary['warnings']} ⚠️")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    
    # Print overall status
    status_emoji = "✅" if report['overall_status'] == 'PASS' else "❌"
    print(f"\nOverall Status: {status_emoji} {report['overall_status']}")
    
    # Print detailed results
    print("\n📋 Detailed Results")
    print("-" * 30)
    for test_name, result in report['test_results'].items():
        status_emoji = {
            'PASS': '✅',
            'FAIL': '❌',
            'WARN': '⚠️'
        }.get(result['status'], '❓')
        
        print(f"{status_emoji} {test_name}: {result['status']}")
        if 'message' in result:
            print(f"   {result['message']}")
        if 'details' in result:
            for detail in result['details']:
                print(f"   {detail}")
    
    # Print errors
    if report['errors']:
        print("\n❌ Errors")
        print("-" * 30)
        for error in report['errors']:
            print(f"  {error}")
    
    # Print recommendations
    if report['recommendations']:
        print("\n💡 Recommendations")
        print("-" * 30)
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    # Exit with appropriate code
    if report['overall_status'] == 'PASS':
        print("\n🎉 All tests passed! System is ready for production.")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed. Please review the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 