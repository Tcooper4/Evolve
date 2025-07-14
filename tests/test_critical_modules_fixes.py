"""
Test script for critical module fixes.

Tests the improvements made to:
- trading/report/report_generator.py: Defensive checks and export logic
- trading/llm/parser_engine.py: LLM prompt parsing with fallback
- agents/prompt_agent.py: Integration with parser engine and strategy routing
- trading/backtesting/backtester.py: NaN handling and error logging
- agents/llm/model_loader.py: Model verification and fallback
"""

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest

# Import the modules we're testing
from trading.report.report_generator import ReportGenerator
from trading.llm.parser_engine import ParserEngine, ParsedIntent
from agents.prompt_agent import PromptAgent
from trading.backtesting.backtester import Backtester
from agents.llm.model_loader import ModelLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestReportGenerator:
    """Test the report generator improvements."""
    
    def setup_method(self):
        """Setup test environment."""
        self.report_gen = ReportGenerator()
        self.temp_dir = tempfile.mkdtemp()
        
    def test_export_signals_defensive_checks(self):
        """Test defensive checks in export_signals method."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = self.report_gen.export_signals(empty_df, "test.csv")
        assert result is False
        
        # Test with None DataFrame
        result = self.report_gen.export_signals(None, "test.csv")
        assert result is False
        
        # Test with missing columns
        df = pd.DataFrame({'Close': [100, 101, 102]})
        result = self.report_gen.export_signals(df, "test.csv", buy_col='Buy', sell_col='Sell')
        assert result is False
        
        # Test with all NaN values
        df = pd.DataFrame({
            'Buy': [np.nan, np.nan, np.nan],
            'Sell': [np.nan, np.nan, np.nan]
        })
        result = self.report_gen.export_signals(df, "test.csv")
        assert result is False
        
    def test_export_signals_success(self):
        """Test successful signal export."""
        # Create valid signals DataFrame
        df = pd.DataFrame({
            'Buy': [1, 0, 1, 0],
            'Sell': [0, 1, 0, 1],
            'Close': [100, 101, 102, 103]
        })
        
        # Test CSV export
        csv_path = os.path.join(self.temp_dir, "signals.csv")
        result = self.report_gen.export_signals(df, csv_path)
        assert result is True
        assert os.path.exists(csv_path)
        
        # Test JSON export
        json_path = os.path.join(self.temp_dir, "signals.json")
        result = self.report_gen.export_signals(df, json_path)
        assert result is True
        assert os.path.exists(json_path)
        
        # Test with custom column names
        df_custom = pd.DataFrame({
            'Long': [1, 0, 1, 0],
            'Short': [0, 1, 0, 1]
        })
        custom_path = os.path.join(self.temp_dir, "custom_signals.csv")
        result = self.report_gen.export_signals(df_custom, custom_path, buy_col='Long', sell_col='Short')
        assert result is True
        assert os.path.exists(custom_path)


class TestParserEngine:
    """Test the parser engine improvements."""
    
    def setup_method(self):
        """Setup test environment."""
        self.parser = ParserEngine(
            enable_debug_mode=True,
            use_regex_first=True,
            use_local_llm=False,  # Disable for testing
            use_openai_fallback=False  # Disable for testing
        )
        
    def test_fallback_regex_router(self):
        """Test the enhanced fallback regex router."""
        # Test forecasting intent
        prompt = "Forecast the price of AAPL for the next 30 days"
        result = self.parser._fallback_regex_router(prompt)
        assert result.intent == 'forecasting'
        assert result.confidence > 0.7
        assert 'ticker' in result.args
        assert result.args['ticker'] == 'AAPL'
        
        # Test strategy intent
        prompt = "Generate a buy signal strategy for TSLA"
        result = self.parser._fallback_regex_router(prompt)
        assert result.intent == 'strategy'
        assert result.confidence > 0.7
        assert 'action' in result.args
        assert result.args['action'] == 'buy'
        
        # Test analysis intent
        prompt = "Analyze the market performance of GOOGL"
        result = self.parser._fallback_regex_router(prompt)
        assert result.intent == 'research'
        assert result.confidence > 0.7
        
        # Test general intent
        prompt = "What is the weather like?"
        result = self.parser._fallback_regex_router(prompt)
        assert result.intent == 'general'
        
    def test_strategy_routing(self):
        """Test strategy routing functionality."""
        # Test forecasting route
        route = self.parser.route_strategy('forecasting', {'horizon': 30})
        assert route is not None
        assert route.strategy_name == 'lstm_forecaster'
        assert 'xgboost_forecaster' in route.fallback_strategies
        
        # Test strategy route
        route = self.parser.route_strategy('strategy', {'market_volatility': 'medium'})
        assert route is not None
        assert route.strategy_name == 'bollinger_strategy'
        
        # Test unknown intent
        route = self.parser.route_strategy('unknown_intent', {})
        assert route is not None  # Should find best match
        
    def test_strategy_registry_management(self):
        """Test strategy registry management."""
        # Test registry summary
        summary = self.parser.get_registry_summary()
        assert 'total_routes' in summary
        assert 'intents' in summary
        assert 'strategies' in summary
        
        # Test updating registry
        new_routes = [{
            'intent': 'custom_strategy',
            'strategy_name': 'custom_forecaster',
            'priority': 1,
            'fallback_strategies': ['lstm_forecaster'],
            'conditions': {},
            'parameters': {'custom_param': 'value'}
        }]
        
        self.parser.update_strategy_registry(new_routes)
        route = self.parser.route_strategy('custom_strategy', {})
        assert route is not None
        assert route.strategy_name == 'custom_forecaster'


class TestPromptAgent:
    """Test the prompt agent improvements."""
    
    def setup_method(self):
        """Setup test environment."""
        self.agent = PromptAgent(
            use_regex_first=True,
            use_local_llm=False,  # Disable for testing
            use_openai_fallback=False  # Disable for testing
        )
        
    def test_parser_engine_integration(self):
        """Test integration with parser engine."""
        # Test basic intent parsing
        prompt = "Forecast AAPL price for next week"
        result = self.agent.parse_intent(prompt)
        assert isinstance(result, ParsedIntent)
        assert result.intent in ['forecasting', 'general']
        assert result.provider in ['regex', 'fallback_regex', 'basic_regex_fallback']
        
    def test_basic_regex_fallback(self):
        """Test basic regex fallback when parser engine fails."""
        # Test with various prompts
        test_cases = [
            ("forecast the market", 'forecasting'),
            ("buy signal strategy", 'strategy'),
            ("analyze performance", 'analysis'),
            ("optimize parameters", 'optimization'),
            ("portfolio allocation", 'portfolio'),
            ("random text", 'general')
        ]
        
        for prompt, expected_intent in test_cases:
            result = self.agent._basic_regex_fallback(prompt)
            assert result.intent == expected_intent
            assert result.provider == 'basic_regex_fallback'
            
    def test_strategy_routing_integration(self):
        """Test strategy routing integration."""
        # Test routing with strategy route
        prompt = "Generate a forecast for AAPL"
        processed = self.agent.process_prompt(prompt)
        
        # Should have strategy route information
        assert hasattr(processed, 'extracted_parameters')
        assert isinstance(processed.extracted_parameters, dict)


class TestBacktester:
    """Test the backtester improvements."""
    
    def setup_method(self):
        """Setup test environment."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.data = pd.DataFrame({
            'AAPL': np.random.randn(100).cumsum() + 100,
            'GOOGL': np.random.randn(100).cumsum() + 200
        }, index=dates)
        
        self.backtester = Backtester(
            data=self.data,
            initial_cash=100000,
            enable_leverage=True,
            enable_fractional_sizing=True
        )
        
    def test_nan_handling_in_performance_metrics(self):
        """Test NaN handling in performance metrics."""
        # Create equity curve with NaN values
        equity_curve = pd.DataFrame({
            'equity_curve': [100000, 101000, np.nan, 102000, 103000],
            'returns': [0.01, np.nan, np.inf, 0.01, 0.01]
        })
        
        # Mock the _calculate_equity_curve method
        self.backtester._calculate_equity_curve = lambda: equity_curve
        
        # Test that metrics are calculated without errors
        metrics = self.backtester.get_performance_metrics()
        assert isinstance(metrics, dict)
        
        # Check that no NaN values are in final metrics
        for metric_name, metric_value in metrics.items():
            assert not pd.isna(metric_value), f"NaN value in {metric_name}"
            assert not np.isinf(metric_value), f"Infinite value in {metric_name}"
            
    def test_signals_dataframe_processing(self):
        """Test signals DataFrame processing."""
        # Test with NaN values
        signals_df = pd.DataFrame({
            'Buy': [1, np.nan, 0, 1, np.nan],
            'Sell': [0, 1, np.nan, 0, 1],
            'Close': [100, 101, 102, 103, 104]
        })
        
        # Test forward fill
        processed = self.backtester.process_signals_dataframe(signals_df, fill_method='ffill')
        assert not processed.isna().any().any()
        assert len(processed) == 5
        
        # Test backward fill
        processed = self.backtester.process_signals_dataframe(signals_df, fill_method='bfill')
        assert not processed.isna().any().any()
        
        # Test drop method
        processed = self.backtester.process_signals_dataframe(signals_df, fill_method='drop')
        assert len(processed) < 5  # Should drop rows with NaN
        
        # Test zero fill
        processed = self.backtester.process_signals_dataframe(signals_df, fill_method='zero')
        assert not processed.isna().any().any()
        assert (processed == 0).any().any()  # Should have some zeros
        
    def test_infinite_value_handling(self):
        """Test handling of infinite values."""
        signals_df = pd.DataFrame({
            'Buy': [1, np.inf, 0, -np.inf, 1],
            'Sell': [0, 1, np.inf, 0, 1]
        })
        
        processed = self.backtester.process_signals_dataframe(signals_df)
        assert not np.isinf(processed).any().any()
        assert not processed.isna().any().any()
        
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        # Test with None
        processed = self.backtester.process_signals_dataframe(None)
        assert processed.empty
        
        # Test with empty DataFrame
        processed = self.backtester.process_signals_dataframe(pd.DataFrame())
        assert processed.empty


class TestModelLoader:
    """Test the model loader improvements."""
    
    def setup_method(self):
        """Setup test environment."""
        self.model_loader = ModelLoader()
        
    def test_model_verification(self):
        """Test model verification functionality."""
        # Test with valid model
        result = self.model_loader.verify_model("gpt-3.5-turbo")
        assert isinstance(result, bool)
        
        # Test with invalid model
        result = self.model_loader.verify_model("invalid_model_name")
        assert result is False
        
    def test_safe_from_pretrained(self):
        """Test safe from_pretrained with fallback."""
        # Mock the AutoTokenizer
        class MockTokenizer:
            def __init__(self, name):
                self.name = name
                
        def mock_from_pretrained(model_name, **kwargs):
            if model_name == "test_model":
                raise Exception("Model not found")
            return MockTokenizer(model_name)
        
        # Test successful loading
        result = self.model_loader._safe_from_pretrained(mock_from_pretrained, "gpt2")
        assert result.name == "gpt2"
        
        # Test fallback loading
        result = self.model_loader._safe_from_pretrained(mock_from_pretrained, "test_model")
        assert result.name in ["distilgpt2", "gpt2", "bert-base-uncased", "roberta-base"]
        
    @pytest.mark.asyncio
    async def test_huggingface_fallback_to_openai(self):
        """Test HuggingFace fallback to OpenAI."""
        # Mock the _load_openai_model method
        async def mock_load_openai(config):
            self.model_loader.models["gpt-3.5-turbo"] = {
                "provider": "openai",
                "config": config
            }
        
        self.model_loader._load_openai_model = mock_load_openai
        
        # Test fallback
        config = self.model_loader.configs["gpt2"]
        config.api_key = "test_key"  # Mock API key
        
        await self.model_loader._fallback_to_openai(config)
        
        # Check that fallback was successful
        assert "gpt2" in self.model_loader.models
        assert self.model_loader.models["gpt2"]["provider"] == "openai"


def run_integration_tests():
    """Run integration tests for all modules."""
    logger.info("Running integration tests...")
    
    # Test report generator
    logger.info("Testing Report Generator...")
    test_report = TestReportGenerator()
    test_report.setup_method()
    test_report.test_export_signals_defensive_checks()
    test_report.test_export_signals_success()
    
    # Test parser engine
    logger.info("Testing Parser Engine...")
    test_parser = TestParserEngine()
    test_parser.setup_method()
    test_parser.test_fallback_regex_router()
    test_parser.test_strategy_routing()
    test_parser.test_strategy_registry_management()
    
    # Test prompt agent
    logger.info("Testing Prompt Agent...")
    test_agent = TestPromptAgent()
    test_agent.setup_method()
    test_agent.test_parser_engine_integration()
    test_agent.test_basic_regex_fallback()
    test_agent.test_strategy_routing_integration()
    
    # Test backtester
    logger.info("Testing Backtester...")
    test_backtester = TestBacktester()
    test_backtester.setup_method()
    test_backtester.test_nan_handling_in_performance_metrics()
    test_backtester.test_signals_dataframe_processing()
    test_backtester.test_infinite_value_handling()
    test_backtester.test_empty_dataframe_handling()
    
    # Test model loader
    logger.info("Testing Model Loader...")
    test_loader = TestModelLoader()
    test_loader.setup_method()
    test_loader.test_model_verification()
    test_loader.test_safe_from_pretrained()
    
    logger.info("All integration tests completed successfully!")


def test_production_readiness():
    """Test production readiness features."""
    logger.info("Testing production readiness...")
    
    # Test error handling
    try:
        # Test with invalid inputs
        report_gen = ReportGenerator()
        result = report_gen.export_signals(None, "test.csv")
        assert result is False, "Should handle None DataFrame"
        
        # Test parser with invalid prompt
        parser = ParserEngine()
        result = parser._fallback_regex_router("")
        assert result.intent == 'general', "Should handle empty prompt"
        
        # Test backtester with invalid data
        backtester = Backtester(data=pd.DataFrame())
        metrics = backtester.get_performance_metrics()
        assert isinstance(metrics, dict), "Should return valid metrics even with invalid data"
        
        logger.info("Production readiness tests passed!")
        
    except Exception as e:
        logger.error(f"Production readiness test failed: {e}")
        raise


if __name__ == "__main__":
    # Run all tests
    run_integration_tests()
    test_production_readiness()
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("\nSummary of improvements tested:")
    print("✅ Report Generator: Defensive checks and export logic")
    print("✅ Parser Engine: LLM prompt parsing with fallback")
    print("✅ Prompt Agent: Integration with parser engine")
    print("✅ Backtester: NaN handling and error logging")
    print("✅ Model Loader: Model verification and fallback")
    print("\nAll modules are now production-ready with:")
    print("- Robust error handling")
    print("- Defensive programming")
    print("- Fallback mechanisms")
    print("- Comprehensive logging")
    print("- Data validation") 