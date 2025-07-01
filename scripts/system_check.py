#!/usr/bin/env python3
"""
Comprehensive System Check for Evolve Trading Platform
Evaluates all modules, imports, and functionality
"""

import os
import sys
import importlib
import traceback
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemChecker:
    """Comprehensive system checker for Evolve platform."""
    
    def __init__(self):
        self.results = {
            'success': 0,
            'failure': 0,
            'warning': 0,
            'total': 0,
            'details': []
        }
        self.critical_modules = [
            'pandas',
            'numpy',
            'yfinance',
            'streamlit',
            'plotly',
            'scikit-learn',
            'tensorflow',
            'torch',
            'transformers'
        ]

    def check_imports(self) -> Dict[str, Any]:
        """Check all critical imports."""
        logger.info("Checking critical imports...")
        
        for module in self.critical_modules:
            try:
                importlib.import_module(module)
                self._record_success(f"Import: {module}")
            except ImportError as e:
                self._record_failure(f"Import: {module} - {str(e)}")
            except Exception as e:
                self._record_warning(f"Import: {module} - {str(e)}")

    def check_core_modules(self) -> None:
        """Check core Evolve modules."""
        logger.info("Checking core modules...")
        
        core_modules = [
            'trading.agents.base_agent',
            'trading.models.forecast_router',
            'trading.strategies.bollinger_strategy',
            'trading.data.data_loader',
            'trading.execution.execution_engine',
            'trading.optimization.bayesian_optimizer',
            'trading.risk.risk_analyzer',
            'trading.portfolio.portfolio_manager',
            'trading.evaluation.metrics',
            'trading.feature_engineering.feature_engineer'
        ]
        
        for module in core_modules:
            try:
                importlib.import_module(module)
                self._record_success(f"Core module: {module}")
            except ImportError as e:
                self._record_failure(f"Core module: {module} - {str(e)}")
            except Exception as e:
                self._record_warning(f"Core module: {module} - {str(e)}")

    def check_advanced_modules(self) -> None:
        """Check advanced feature modules."""
        logger.info("Checking advanced modules...")
        
        advanced_modules = [
            'causal.causal_model',
            'rl.strategy_trainer',
            'trading.models.advanced.tcn.tcn_model',
            'trading.models.advanced.transformer.transformer_model',
            'trading.models.advanced.lstm.lstm_model',
            'trading.models.advanced.gnn.gnn_model',
            'trading.models.advanced.rl.rl_model',
            'trading.models.advanced.ensemble.ensemble_model',
            'trading.nlp.llm_processor',
            'trading.meta_agents.agents.agent_router'
        ]
        
        for module in advanced_modules:
            try:
                importlib.import_module(module)
                self._record_success(f"Advanced module: {module}")
            except ImportError as e:
                self._record_warning(f"Advanced module: {module} - {str(e)}")
            except Exception as e:
                self._record_warning(f"Advanced module: {module} - {str(e)}")

    def check_ui_modules(self) -> None:
        """Check UI and Streamlit modules."""
        logger.info("Checking UI modules...")
        
        ui_modules = [
            'pages.1_Forecast_Trade',
            'pages.2_Strategy_Backtest',
            'pages.3_Trade_Execution',
            'pages.4_Portfolio_Management',
            'pages.5_Risk_Analysis',
            'pages.6_Model_Optimization',
            'pages.7_Market_Analysis',
            'pages.8_Agent_Management',
            'pages.9_System_Monitoring',
            'pages.10_Strategy_Health_Dashboard'
        ]
        
        for module in ui_modules:
            try:
                importlib.import_module(module)
                self._record_success(f"UI module: {module}")
            except ImportError as e:
                self._record_failure(f"UI module: {module} - {str(e)}")
            except Exception as e:
                self._record_warning(f"UI module: {module} - {str(e)}")

    def check_config_files(self) -> None:
        """Check configuration files exist."""
        logger.info("Checking configuration files...")
        
        config_files = [
            'config/app_config.yaml',
            'config/config.json',
            'trading/config/configuration.py',
            'trading/config/enhanced_settings.py',
            'trading/agents/agent_config.json',
            'trading/nlp/config/entity_patterns.json',
            'trading/nlp/config/intent_patterns.json',
            'trading/nlp/config/response_templates.json'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                self._record_success(f"Config file: {config_file}")
            else:
                self._record_warning(f"Config file: {config_file} - Missing")

    def check_data_sources(self) -> None:
        """Check data source connectivity."""
        logger.info("Checking data sources...")
        
        try:
            import yfinance as yf
            # Test basic data fetch
            ticker = yf.Ticker("AAPL")
            hist = ticker.history(period="1d")
            if not hist.empty:
                self._record_success("Data source: yfinance")
            else:
                self._record_warning("Data source: yfinance - No data returned")
        except Exception as e:
            self._record_warning(f"Data source: yfinance - {str(e)}")

    def check_model_functionality(self) -> None:
        """Check model functionality."""
        logger.info("Checking model functionality...")
        
        try:
            from trading.models.forecast_router import ForecastRouter
            router = ForecastRouter()
            self._record_success("Model: ForecastRouter")
        except Exception as e:
            self._record_failure(f"Model: ForecastRouter - {str(e)}")
        
        try:
            from trading.strategies.bollinger_strategy import BollingerStrategy
            strategy = BollingerStrategy()
            self._record_success("Strategy: BollingerStrategy")
        except Exception as e:
            self._record_failure(f"Strategy: BollingerStrategy - {str(e)}")

    def check_streamlit_app(self) -> None:
        """Check Streamlit app functionality."""
        logger.info("Checking Streamlit app...")
        
        try:
            import streamlit as st
            self._record_success("Streamlit: Core functionality")
        except Exception as e:
            self._record_failure(f"Streamlit: Core functionality - {str(e)}")
        
        # Check if app.py exists and is valid
        if os.path.exists('app.py'):
            try:
                with open('app.py', 'r') as f:
                    content = f.read()
                if 'streamlit' in content and 'st.' in content:
                    self._record_success("Streamlit: app.py structure")
                else:
                    self._record_warning("Streamlit: app.py - Missing streamlit code")
            except Exception as e:
                self._record_warning(f"Streamlit: app.py - {str(e)}")
        else:
            self._record_failure("Streamlit: app.py - File missing")

    def check_file_structure(self) -> None:
        """Check critical file structure."""
        logger.info("Checking file structure...")
        
        critical_dirs = [
            'trading',
            'pages',
            'config',
            'data',
            'models',
            'strategies',
            'utils',
            'scripts'
        ]
        
        for dir_name in critical_dirs:
            if os.path.exists(dir_name):
                self._record_success(f"Directory: {dir_name}")
            else:
                self._record_failure(f"Directory: {dir_name} - Missing")

    def _record_success(self, message: str) -> None:
        """Record successful check."""
        self.results['success'] += 1
        self.results['total'] += 1
        self.results['details'].append(f"✅ {message}")
        logger.info(f"✅ {message}")

    def _record_failure(self, message: str) -> None:
        """Record failed check."""
        self.results['failure'] += 1
        self.results['total'] += 1
        self.results['details'].append(f"❌ {message}")
        logger.error(f"❌ {message}")

    def _record_warning(self, message: str) -> None:
        """Record warning check."""
        self.results['warning'] += 1
        self.results['total'] += 1
        self.results['details'].append(f"⚠️ {message}")
        logger.warning(f"⚠️ {message}")

    def run_full_check(self) -> Dict[str, Any]:
        """Run complete system check."""
        logger.info("Starting comprehensive system check...")
        
        self.check_imports()
        self.check_core_modules()
        self.check_advanced_modules()
        self.check_ui_modules()
        self.check_config_files()
        self.check_data_sources()
        self.check_model_functionality()
        self.check_streamlit_app()
        self.check_file_structure()
        
        # Calculate success rate
        if self.results['total'] > 0:
            success_rate = (self.results['success'] / self.results['total']) * 100
        else:
            success_rate = 0
        
        self.results['success_rate'] = success_rate
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SYSTEM CHECK COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Total Checks: {self.results['total']}")
        logger.info(f"✅ Success: {self.results['success']}")
        logger.info(f"❌ Failures: {self.results['failure']}")
        logger.info(f"⚠️ Warnings: {self.results['warning']}")
        logger.info(f"{'='*60}")
        
        return self.results

def main():
    """Main function to run system check."""
    checker = SystemChecker()
    results = checker.run_full_check()
    
    # Print detailed results
    print("\nDETAILED RESULTS:")
    print("="*60)
    for detail in results['details']:
        print(detail)
    
    print(f"\nSUMMARY:")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    print(f"Total: {results['total']} | Success: {results['success']} | Failures: {results['failure']} | Warnings: {results['warning']}")
    
    return results

if __name__ == "__main__":
    main() 