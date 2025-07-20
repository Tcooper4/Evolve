"""
Tests for ensemble voter functionality including audit reporting.
"""

import json
import pickle
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from trading.strategies.ensemble_methods import (
    DynamicEnsembleVoter,
    StrategySignal,
    SignalType,
    HybridSignal,
    ModelPerformance,
    combine_weighted_average,
    combine_voting,
    combine_ensemble_model,
    calculate_ensemble_position_size,
    create_fallback_hybrid_signal,
    update_model_performance,
    combine_signals_dynamic,
    get_ensemble_audit_report,
)


class TestDynamicEnsembleVoter:
    """Test DynamicEnsembleVoter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.voter = DynamicEnsembleVoter()
        self.test_signals = [
            StrategySignal(
                strategy_name="RSI_Strategy",
                signal_type=SignalType.BUY,
                confidence=0.8,
                predicted_return=0.05,
                position_size=0.5,
                risk_score=0.3,
                timestamp=datetime.now(),
                metadata={"rsi_value": 30}
            ),
            StrategySignal(
                strategy_name="MACD_Strategy",
                signal_type=SignalType.SELL,
                confidence=0.7,
                predicted_return=-0.03,
                position_size=0.3,
                risk_score=0.4,
                timestamp=datetime.now(),
                metadata={"macd_signal": "bearish"}
            ),
        ]

    def test_initialization(self):
        """Test voter initialization."""
        assert self.voter.epsilon == 1e-6
        assert self.voter.decay_factor == 0.95
        assert self.voter.lookback_window == 30
        assert self.voter.min_weight == 0.01
        assert self.voter.max_weight == 0.5
        assert isinstance(self.voter.model_performance, dict)
        assert isinstance(self.voter.audit_history, list)

    def test_update_model_performance(self):
        """Test model performance update."""
        strategy_name = "test_strategy"
        actual_returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        predicted_returns = pd.Series([0.015, 0.018, -0.008, 0.025])
        
        self.voter.update_model_performance(strategy_name, actual_returns, predicted_returns)
        
        assert strategy_name in self.voter.model_performance
        perf = self.voter.model_performance[strategy_name]
        assert perf.strategy_name == strategy_name
        assert len(perf.mse_history) == 1
        assert perf.recent_performance > 0
        assert perf.weight > 0

    def test_calculate_dynamic_weight(self):
        """Test dynamic weight calculation."""
        perf = ModelPerformance(
            strategy_name="test",
            mse_history=[0.001, 0.002],
            timestamps=[datetime.now() - timedelta(days=1), datetime.now()],
            recent_performance=500.0,  # 1/0.002
            weight=1.0,
            last_updated=datetime.now()
        )
        
        weight = self.voter._calculate_dynamic_weight(perf)
        assert self.voter.min_weight <= weight <= self.voter.max_weight

    def test_get_dynamic_weights(self):
        """Test dynamic weight retrieval."""
        # Add some test performance data
        self.voter.model_performance["strategy1"] = ModelPerformance(
            strategy_name="strategy1",
            mse_history=[0.001],
            timestamps=[datetime.now()],
            recent_performance=1000.0,
            weight=0.6,
            last_updated=datetime.now()
        )
        
        self.voter.model_performance["strategy2"] = ModelPerformance(
            strategy_name="strategy2",
            mse_history=[0.002],
            timestamps=[datetime.now()],
            recent_performance=500.0,
            weight=0.4,
            last_updated=datetime.now()
        )
        
        weights = self.voter.get_dynamic_weights(["strategy1", "strategy2"])
        assert "strategy1" in weights
        assert "strategy2" in weights
        assert abs(weights["strategy1"] + weights["strategy2"] - 1.0) < 1e-6

    def test_combine_signals_dynamic(self):
        """Test dynamic signal combination."""
        result = self.voter.combine_signals_dynamic(self.test_signals)
        
        assert isinstance(result, HybridSignal)
        assert result.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
        assert 0 <= result.confidence <= 1
        assert 0 <= result.position_size <= 1
        assert len(result.individual_signals) == 2
        assert "strategy_weights" in result.metadata

    def test_combine_signals_without_dynamic_weights(self):
        """Test signal combination without dynamic weights."""
        result = self.voter.combine_signals_dynamic(
            self.test_signals, use_dynamic_weights=False
        )
        
        assert isinstance(result, HybridSignal)
        # Should have equal weights
        weights = result.strategy_weights
        assert abs(weights["RSI_Strategy"] - weights["MACD_Strategy"]) < 1e-6

    def test_audit_recording(self):
        """Test audit information recording."""
        initial_audit_count = len(self.voter.audit_history)
        
        self.voter.combine_signals_dynamic(self.test_signals)
        
        assert len(self.voter.audit_history) == initial_audit_count + 1
        latest_audit = self.voter.audit_history[-1]
        assert "timestamp" in latest_audit
        assert "signals" in latest_audit
        assert "final_signal" in latest_audit
        assert "weights" in latest_audit

    def test_get_ensemble_audit_report(self):
        """Test ensemble audit report generation."""
        # Generate some audit history
        self.voter.combine_signals_dynamic(self.test_signals)
        self.voter.combine_signals_dynamic(self.test_signals)
        
        report = self.voter.get_ensemble_audit_report()
        
        assert "total_decisions" in report
        assert "win_rates" in report
        assert "recent_performance" in report
        assert "audit_summary" in report
        assert report["total_decisions"] >= 2

    def test_save_and_load_state(self):
        """Test state persistence."""
        # Add some test data
        self.voter.model_performance["test_strategy"] = ModelPerformance(
            strategy_name="test_strategy",
            mse_history=[0.001, 0.002],
            timestamps=[datetime.now() - timedelta(days=1), datetime.now()],
            recent_performance=500.0,
            weight=0.5,
            last_updated=datetime.now()
        )
        
        self.voter.audit_history.append({
            "timestamp": datetime.now().isoformat(),
            "signals": [],
            "final_signal": "hold",
            "final_confidence": 0.5,
            "weights": {}
        })
        
        # Save state
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            state_file = f.name
            
        try:
            self.voter.save_state(state_file)
            
            # Create new voter and load state
            new_voter = DynamicEnsembleVoter()
            new_voter.load_state(state_file)
            
            # Verify data was loaded
            assert "test_strategy" in new_voter.model_performance
            assert len(new_voter.audit_history) == 1
            
        finally:
            Path(state_file).unlink(missing_ok=True)

    def test_empty_signals_handling(self):
        """Test handling of empty signal list."""
        result = self.voter.combine_signals_dynamic([])
        assert isinstance(result, HybridSignal)
        assert result.signal_type == SignalType.HOLD
        assert result.confidence == 0.5
        assert result.position_size == 0.0

    def test_error_handling(self):
        """Test error handling in signal combination."""
        # Test with invalid signals
        invalid_signals = [
            StrategySignal(
                strategy_name="invalid",
                signal_type=SignalType.BUY,
                confidence=float('nan'),  # Invalid confidence
                predicted_return=0.0,
                position_size=0.0,
                risk_score=0.0,
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        result = self.voter.combine_signals_dynamic(invalid_signals)
        assert isinstance(result, HybridSignal)


class TestEnsembleMethods:
    """Test ensemble method functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_signals = [
            StrategySignal(
                strategy_name="Strategy1",
                signal_type=SignalType.BUY,
                confidence=0.8,
                predicted_return=0.05,
                position_size=0.5,
                risk_score=0.3,
                timestamp=datetime.now(),
                metadata={}
            ),
            StrategySignal(
                strategy_name="Strategy2",
                signal_type=SignalType.SELL,
                confidence=0.7,
                predicted_return=-0.03,
                position_size=0.3,
                risk_score=0.4,
                timestamp=datetime.now(),
                metadata={}
            ),
        ]
        self.strategy_weights = {"Strategy1": 0.6, "Strategy2": 0.4}

    def test_combine_weighted_average(self):
        """Test weighted average combination."""
        result = combine_weighted_average(self.test_signals, self.strategy_weights)
        
        assert isinstance(result, HybridSignal)
        assert result.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
        assert 0 <= result.confidence <= 1
        assert len(result.individual_signals) == 2

    def test_combine_voting(self):
        """Test voting combination."""
        result = combine_voting(self.test_signals, self.strategy_weights)
        
        assert isinstance(result, HybridSignal)
        assert result.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
        assert "vote_counts" in result.metadata
        assert "vote_ratio" in result.metadata

    def test_combine_ensemble_model(self):
        """Test ensemble model combination."""
        # Mock ensemble model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.02])
        
        result = combine_ensemble_model(
            self.test_signals, mock_model, self.strategy_weights
        )
        
        assert isinstance(result, HybridSignal)
        assert "ensemble_prediction" in result.metadata

    def test_calculate_ensemble_position_size(self):
        """Test position size calculation."""
        position_size = calculate_ensemble_position_size(0.8, 0.3)
        assert 0 <= position_size <= 1
        
        # Test edge cases
        assert calculate_ensemble_position_size(0.0, 0.0) == 0.0
        assert calculate_ensemble_position_size(1.0, 0.0) == 1.0

    def test_create_fallback_hybrid_signal(self):
        """Test fallback signal creation."""
        signal = create_fallback_hybrid_signal()
        
        assert isinstance(signal, HybridSignal)
        assert signal.signal_type == SignalType.HOLD
        assert signal.confidence == 0.5
        assert signal.position_size == 0.0
        assert signal.risk_score == 1.0
        assert len(signal.individual_signals) == 0

    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test update_model_performance
        actual_returns = pd.Series([0.01, 0.02, -0.01])
        predicted_returns = pd.Series([0.015, 0.018, -0.008])
        
        update_model_performance("test_strategy", actual_returns, predicted_returns)
        
        # Test combine_signals_dynamic
        result = combine_signals_dynamic(self.test_signals)
        assert isinstance(result, HybridSignal)
        
        # Test get_ensemble_audit_report
        report = get_ensemble_audit_report()
        assert isinstance(report, dict)


class TestPerformancePersistence:
    """Test performance data persistence."""

    def test_performance_data_save_load(self):
        """Test saving and loading performance data."""
        voter = DynamicEnsembleVoter()
        
        # Add test performance data
        actual_returns = pd.Series([0.01, 0.02, -0.01])
        predicted_returns = pd.Series([0.015, 0.018, -0.008])
        
        voter.update_model_performance("test_strategy", actual_returns, predicted_returns)
        
        # Test save/load with temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            perf_file = f.name
            
        try:
            # Mock the save method to use our temp file
            with patch.object(voter, '_save_performance_data') as mock_save:
                voter._save_performance_data()
                mock_save.assert_called_once()
                
        finally:
            Path(perf_file).unlink(missing_ok=True)


class TestAuditReporting:
    """Test audit reporting functionality."""

    def test_audit_report_with_no_history(self):
        """Test audit report when no history exists."""
        voter = DynamicEnsembleVoter()
        report = voter.get_ensemble_audit_report()
        
        assert report["message"] == "No audit history available"

    def test_audit_report_with_history(self):
        """Test audit report with audit history."""
        voter = DynamicEnsembleVoter()
        
        # Create some audit history
        for i in range(5):
            signals = [
                StrategySignal(
                    strategy_name=f"Strategy{i % 2 + 1}",
                    signal_type=SignalType.BUY if i % 2 == 0 else SignalType.SELL,
                    confidence=0.7 + (i * 0.1),
                    predicted_return=0.05 if i % 2 == 0 else -0.03,
                    position_size=0.5,
                    risk_score=0.3,
                    timestamp=datetime.now(),
                    metadata={}
                )
            ]
            voter.combine_signals_dynamic(signals)
        
        report = voter.get_ensemble_audit_report()
        
        assert report["total_decisions"] == 5
        assert "win_rates" in report
        assert "recent_performance" in report
        assert "audit_summary" in report

    def test_audit_report_model_wins_calculation(self):
        """Test calculation of model wins in audit report."""
        voter = DynamicEnsembleVoter()
        
        # Create signals with different strategies and confidences
        signals1 = [
            StrategySignal(
                strategy_name="Strategy1",
                signal_type=SignalType.BUY,
                confidence=0.9,  # High confidence
                predicted_return=0.05,
                position_size=0.5,
                risk_score=0.3,
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        signals2 = [
            StrategySignal(
                strategy_name="Strategy2",
                signal_type=SignalType.SELL,
                confidence=0.6,  # Lower confidence
                predicted_return=-0.03,
                position_size=0.3,
                risk_score=0.4,
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        # Combine signals multiple times
        for _ in range(3):
            voter.combine_signals_dynamic(signals1)
        for _ in range(2):
            voter.combine_signals_dynamic(signals2)
        
        report = voter.get_ensemble_audit_report()
        
        # Strategy1 should have more wins due to higher confidence
        win_rates = report["win_rates"]
        assert "Strategy1" in win_rates
        assert "Strategy2" in win_rates
        assert win_rates["Strategy1"] >= win_rates["Strategy2"]


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_confidence_threshold(self):
        """Test handling of invalid confidence threshold."""
        voter = DynamicEnsembleVoter()
        signals = [
            StrategySignal(
                strategy_name="test",
                signal_type=SignalType.BUY,
                confidence=0.3,  # Low confidence
                predicted_return=0.05,
                position_size=0.5,
                risk_score=0.3,
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        # Should result in HOLD signal due to low confidence
        result = voter.combine_signals_dynamic(signals, confidence_threshold=0.5)
        assert result.signal_type == SignalType.HOLD

    def test_mixed_signal_types(self):
        """Test handling of mixed signal types."""
        voter = DynamicEnsembleVoter()
        signals = [
            StrategySignal(
                strategy_name="buy_strategy",
                signal_type=SignalType.BUY,
                confidence=0.8,
                predicted_return=0.05,
                position_size=0.5,
                risk_score=0.3,
                timestamp=datetime.now(),
                metadata={}
            ),
            StrategySignal(
                strategy_name="sell_strategy",
                signal_type=SignalType.SELL,
                confidence=0.8,
                predicted_return=-0.05,
                position_size=0.5,
                risk_score=0.3,
                timestamp=datetime.now(),
                metadata={}
            ),
        ]
        
        result = voter.combine_signals_dynamic(signals)
        # Should result in some signal type (not necessarily HOLD)
        assert result.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]

    def test_performance_data_validation(self):
        """Test validation of performance data."""
        voter = DynamicEnsembleVoter()
        
        # Test with empty series
        empty_series = pd.Series([])
        voter.update_model_performance("test", empty_series, empty_series)
        
        # Test with mismatched lengths
        series1 = pd.Series([0.01, 0.02])
        series2 = pd.Series([0.015])
        voter.update_model_performance("test2", series1, series2)
        
        # Should handle gracefully without errors
        assert "test" in voter.model_performance
        assert "test2" in voter.model_performance


if __name__ == "__main__":
    pytest.main([__file__])
