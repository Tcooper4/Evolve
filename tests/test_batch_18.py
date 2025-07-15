"""
Batch 18 Tests
Tests for fault tolerance, clarity, and fallback logic improvements
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Import Batch 18 modules
from agents.model_discovery_agent import ModelDiscoveryAgent
from trading.ensemble.diagnostic_tools import EnsembleDiagnosticTools, DiagnosticLevel
from trading.strategies.strategy_fallback import StrategyFallback, FallbackStrategy
from trading.agents.prompt_response_validator import PromptResponseValidator, ValidationLevel


class TestModelDiscoveryAgent(unittest.TestCase):
    """Test ModelDiscoveryAgent fallback functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = ModelDiscoveryAgent()
    
    def test_fallback_to_predefined_models(self):
        """Test fallback to pre-defined models when none discovered."""
        # Mock _discover_models to return empty list
        with patch.object(self.agent, '_discover_models', return_value=[]):
            with patch.object(self.agent, '_validate_models') as mock_validate:
                mock_validate.return_value = [{'id': 'test', 'type': 'lstm'}]
                
                result = self.agent.execute()
                
                # Check that fallback was used
                self.assertTrue(result.success)
                self.assertTrue(result.data['used_fallback'])
                self.assertGreater(result.data['discovered_models'], 0)
    
    def test_predefined_models_structure(self):
        """Test that predefined models have correct structure."""
        predefined_models = self.agent._get_predefined_models()
        
        self.assertIsInstance(predefined_models, list)
        self.assertGreater(len(predefined_models), 0)
        
        for model in predefined_models:
            self.assertIn('id', model)
            self.assertIn('type', model)
            self.assertIn('config', model)
            self.assertIn('status', model)
            self.assertEqual(model['status'], 'predefined')
    
    def test_warning_logged_when_no_models_discovered(self):
        """Test that warning is logged when no models discovered."""
        with patch.object(self.agent, '_discover_models', return_value=[]):
            with patch.object(self.agent.logger, 'warning') as mock_warning:
                with patch.object(self.agent, '_validate_models', return_value=[]):
                    self.agent.execute()
                    mock_warning.assert_called_with("No models discovered, falling back to pre-defined model list")


class TestEnsembleDiagnosticTools(unittest.TestCase):
    """Test EnsembleDiagnosticTools confidence filtering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tools = EnsembleDiagnosticTools(confidence_threshold=0.2)
    
    def test_confidence_filtering(self):
        """Test filtering of low confidence models."""
        ensemble_data = {
            'ensemble_id': 'test_ensemble',
            'models': [
                {
                    'model_name': 'high_conf_model',
                    'model_type': 'lstm',
                    'confidence_score': 0.8,
                    'performance_metrics': {'accuracy': 0.85},
                    'prediction_quality': 0.8
                },
                {
                    'model_name': 'low_conf_model',
                    'model_type': 'xgboost',
                    'confidence_score': 0.1,
                    'performance_metrics': {'accuracy': 0.6},
                    'prediction_quality': 0.5
                }
            ]
        }
        
        diagnostic = self.tools.analyze_ensemble(ensemble_data)
        
        # Should filter out low confidence model
        self.assertEqual(len(diagnostic.models), 1)
        self.assertEqual(diagnostic.models[0].model_name, 'high_conf_model')
    
    def test_debug_mode_includes_low_confidence(self):
        """Test that debug mode includes low confidence models."""
        self.tools.enable_debug_mode(True)
        
        ensemble_data = {
            'ensemble_id': 'test_ensemble',
            'models': [
                {
                    'model_name': 'low_conf_model',
                    'model_type': 'xgboost',
                    'confidence_score': 0.1,
                    'performance_metrics': {'accuracy': 0.6},
                    'prediction_quality': 0.5
                }
            ]
        }
        
        diagnostic = self.tools.analyze_ensemble(ensemble_data)
        
        # Should include low confidence model in debug mode
        self.assertEqual(len(diagnostic.models), 1)
        self.assertEqual(diagnostic.models[0].model_name, 'low_conf_model')
    
    def test_diagnostic_report_filtering(self):
        """Test diagnostic report filtering."""
        ensemble_data = {
            'ensemble_id': 'test_ensemble',
            'models': [
                {
                    'model_name': 'high_conf_model',
                    'model_type': 'lstm',
                    'confidence_score': 0.8,
                    'performance_metrics': {'accuracy': 0.85},
                    'prediction_quality': 0.8
                },
                {
                    'model_name': 'low_conf_model',
                    'model_type': 'xgboost',
                    'confidence_score': 0.1,
                    'performance_metrics': {'accuracy': 0.6},
                    'prediction_quality': 0.5
                }
            ]
        }
        
        diagnostic = self.tools.analyze_ensemble(ensemble_data)
        report = self.tools.generate_diagnostic_report(diagnostic)
        
        # Should show filtered models count
        self.assertEqual(report['total_models'], 2)
        self.assertEqual(report['reported_models'], 1)
        self.assertEqual(report['filtered_models'], 1)


class TestStrategyFallback(unittest.TestCase):
    """Test StrategyFallback ranked fallback functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fallback = StrategyFallback(fallback_pool=["RSI", "SMA", "MACD"])
    
    def test_ranked_fallbacks(self):
        """Test that fallbacks are ranked by performance."""
        ranked_fallbacks = self.fallback.get_ranked_fallbacks()
        
        self.assertIsInstance(ranked_fallbacks, list)
        self.assertEqual(len(ranked_fallbacks), 3)
        
        # Check that strategies are ranked (scores should be descending)
        scores = [score for _, score in ranked_fallbacks]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_fallback_execution_order(self):
        """Test that fallbacks are executed in ranked order."""
        market_data = pd.DataFrame({'close': [100, 101, 102]})
        context = {'rsi': 25, 'current_price': 100}
        
        with patch.object(self.fallback, '_execute_strategy') as mock_execute:
            mock_execute.return_value = Mock(
                strategy_name="RSI",
                signal="BUY",
                confidence=0.8,
                performance_metrics={},
                execution_time=0.1,
                fallback_rank=1
            )
            
            result = self.fallback.execute_fallback(market_data, context)
            
            # Should call _execute_strategy for each fallback in order
            self.assertEqual(mock_execute.call_count, 3)
    
    def test_performance_update(self):
        """Test strategy performance update."""
        trade_result = {
            'pnl': 0.02,
            'win': True,
            'strategy_used': 'RSI'
        }
        
        self.fallback.update_strategy_performance('RSI', trade_result)
        
        # Check that performance was updated
        self.assertIn('RSI', self.fallback.strategy_performance)
        self.assertEqual(len(self.fallback.trade_history), 1)
    
    def test_fallback_pool_customization(self):
        """Test custom fallback pool."""
        custom_fallback = StrategyFallback(fallback_pool=["Bollinger", "Momentum"])
        
        ranked_fallbacks = custom_fallback.get_ranked_fallbacks()
        strategy_names = [name for name, _ in ranked_fallbacks]
        
        self.assertEqual(set(strategy_names), {"Bollinger", "Momentum"})


class TestPromptResponseValidator(unittest.TestCase):
    """Test PromptResponseValidator schema validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = PromptResponseValidator()
    
    def test_valid_strategy_result(self):
        """Test validation of valid strategy result."""
        valid_data = {
            'buy': pd.Series([True, False, True]),
            'sell': pd.Series([False, True, False]),
            'price': pd.Series([100.0, 101.0, 102.0]),
            'confidence': 0.8,
            'strategy_name': 'test_strategy'
        }
        
        result = self.validator.validate_strategy_result(valid_data)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        invalid_data = {
            'buy': pd.Series([True, False]),
            # Missing 'sell' and 'price'
            'confidence': 0.8
        }
        
        result = self.validator.validate_strategy_result(invalid_data)
        
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
        self.assertTrue(any("Missing required field" in error for error in result.errors))
    
    def test_invalid_field_types(self):
        """Test validation with invalid field types."""
        invalid_data = {
            'buy': [True, False],  # Should be pd.Series
            'sell': [False, True],  # Should be pd.Series
            'price': [100.0, 101.0],  # Should be pd.Series
            'confidence': "0.8"  # Should be float
        }
        
        result = self.validator.validate_strategy_result(invalid_data)
        
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
    
    def test_auto_correction(self):
        """Test auto-correction functionality."""
        self.validator.enable_auto_correction(True)
        
        invalid_data = {
            'buy': [True, False],  # Wrong type
            'sell': [False, True],  # Wrong type
            'price': [100.0, 101.0],  # Wrong type
            'confidence': "0.8"  # Wrong type
        }
        
        result = self.validator.validate_strategy_result(invalid_data)
        
        # Should be valid after auto-correction
        self.assertTrue(result.is_valid)
        self.assertIn("Data was auto-corrected", result.warnings)
    
    def test_validation_levels(self):
        """Test different validation levels."""
        # Test strict validation
        strict_validator = PromptResponseValidator(ValidationLevel.STRICT)
        
        data_with_warnings = {
            'buy': pd.Series([True, False]),
            'sell': pd.Series([False, True]),
            'price': pd.Series([100.0, 101.0]),
            # Missing optional fields
        }
        
        result = strict_validator.validate_strategy_result(data_with_warnings)
        
        # Should have warnings but still be valid
        self.assertTrue(result.is_valid)
        self.assertGreater(len(result.warnings), 0)
    
    def test_json_response_validation(self):
        """Test JSON response validation."""
        valid_json = '{"action": "buy", "confidence": 0.8}'
        invalid_json = '{"action": "buy", "confidence": 0.8'  # Missing closing brace
        
        # Test valid JSON
        result = self.validator.validate_prompt_response(valid_json, "json")
        self.assertTrue(result.is_valid)
        
        # Test invalid JSON
        result = self.validator.validate_prompt_response(invalid_json, "json")
        self.assertFalse(result.is_valid)
        self.assertIn("Invalid JSON format", result.errors[0])


class TestBatch18Integration(unittest.TestCase):
    """Integration tests for Batch 18 modules."""
    
    def test_end_to_end_fallback_flow(self):
        """Test complete fallback flow with validation."""
        # Create instances
        discovery_agent = ModelDiscoveryAgent()
        diagnostic_tools = EnsembleDiagnosticTools(confidence_threshold=0.2)
        strategy_fallback = StrategyFallback()
        validator = PromptResponseValidator()
        
        # Simulate model discovery with fallback
        with patch.object(discovery_agent, '_discover_models', return_value=[]):
            discovery_result = discovery_agent.execute()
            self.assertTrue(discovery_result.success)
            self.assertTrue(discovery_result.data['used_fallback'])
        
        # Simulate ensemble analysis with filtering
        ensemble_data = {
            'ensemble_id': 'test',
            'models': [
                {'model_name': 'high_conf', 'model_type': 'lstm', 'confidence_score': 0.8, 
                 'performance_metrics': {}, 'prediction_quality': 0.8},
                {'model_name': 'low_conf', 'model_type': 'xgboost', 'confidence_score': 0.1,
                 'performance_metrics': {}, 'prediction_quality': 0.5}
            ]
        }
        
        diagnostic = diagnostic_tools.analyze_ensemble(ensemble_data)
        self.assertEqual(len(diagnostic.models), 1)  # Low confidence filtered out
        
        # Simulate strategy fallback execution
        market_data = pd.DataFrame({'close': [100, 101, 102]})
        fallback_result = strategy_fallback.execute_fallback(market_data)
        self.assertIsNotNone(fallback_result)
        
        # Validate strategy result
        strategy_data = {
            'buy': pd.Series([True, False]),
            'sell': pd.Series([False, True]),
            'price': pd.Series([100.0, 101.0])
        }
        
        validation_result = validator.validate_strategy_result(strategy_data)
        self.assertTrue(validation_result.is_valid)


if __name__ == '__main__':
    unittest.main() 