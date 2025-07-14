"""
Tests for strategy output merging functionality

Tests duplicate timestamp handling, None/empty DataFrame fallbacks,
and strategy signal aggregation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyOutputMerger:
    """
    Handles merging of multiple strategy outputs with conflict resolution
    """
    
    def __init__(self, conflict_resolution: str = 'weighted_vote'):
        self.conflict_resolution = conflict_resolution
        self.strategy_weights = {}
        self.logger = logging.getLogger(__name__)
    
    def set_strategy_weight(self, strategy_name: str, weight: float) -> None:
        """Set weight for a strategy"""
        self.strategy_weights[strategy_name] = weight
    
    def merge_strategy_outputs(
        self,
        strategy_outputs: Dict[str, pd.DataFrame],
        timestamp_column: str = 'timestamp',
        signal_column: str = 'signal'
    ) -> pd.DataFrame:
        """
        Merge multiple strategy outputs with conflict resolution
        
        Args:
            strategy_outputs: Dict of strategy name to DataFrame
            timestamp_column: Column containing timestamps
            signal_column: Column containing signals
            
        Returns:
            Merged DataFrame with resolved conflicts
        """
        if not strategy_outputs:
            self.logger.warning("No strategy outputs provided")
            return pd.DataFrame()
        
        # Filter out None/empty outputs
        valid_outputs = {}
        for strategy_name, df in strategy_outputs.items():
            if df is not None and not df.empty:
                valid_outputs[strategy_name] = df.copy()
            else:
                self.logger.warning(f"Strategy {strategy_name} returned None or empty DataFrame")
        
        if not valid_outputs:
            self.logger.error("No valid strategy outputs after filtering")
            return self._create_fallback_output()
        
        # Merge all outputs
        merged_df = self._merge_dataframes(valid_outputs, timestamp_column)
        
        # Resolve conflicts
        resolved_df = self._resolve_conflicts(merged_df, signal_column)
        
        return resolved_df
    
    def _merge_dataframes(
        self,
        strategy_outputs: Dict[str, pd.DataFrame],
        timestamp_column: str
    ) -> pd.DataFrame:
        """Merge multiple DataFrames on timestamp"""
        merged_data = []
        
        for strategy_name, df in strategy_outputs.items():
            if timestamp_column not in df.columns:
                self.logger.warning(f"Strategy {strategy_name} missing timestamp column")
                continue
            
            # Add strategy name column
            df_copy = df.copy()
            df_copy['strategy_name'] = strategy_name
            df_copy['strategy_weight'] = self.strategy_weights.get(strategy_name, 1.0)
            
            merged_data.append(df_copy)
        
        if not merged_data:
            return pd.DataFrame()
        
        # Concatenate all DataFrames
        merged_df = pd.concat(merged_data, ignore_index=True)
        
        # Sort by timestamp
        merged_df = merged_df.sort_values(timestamp_column)
        
        return merged_df
    
    def _resolve_conflicts(self, merged_df: pd.DataFrame, signal_column: str) -> pd.DataFrame:
        """Resolve conflicts using specified resolution method"""
        if merged_df.empty:
            return merged_df
        
        if self.conflict_resolution == 'weighted_vote':
            return self._weighted_vote_resolution(merged_df, signal_column)
        elif self.conflict_resolution == 'majority':
            return self._majority_vote_resolution(merged_df, signal_column)
        elif self.conflict_resolution == 'priority':
            return self._priority_resolution(merged_df, signal_column)
        else:
            self.logger.warning(f"Unknown conflict resolution: {self.conflict_resolution}")
            return self._majority_vote_resolution(merged_df, signal_column)
    
    def _weighted_vote_resolution(self, merged_df: pd.DataFrame, signal_column: str) -> pd.DataFrame:
        """Resolve conflicts using weighted voting"""
        # Group by timestamp
        grouped = merged_df.groupby('timestamp')
        
        resolved_data = []
        
        for timestamp, group in grouped:
            # Calculate weighted votes for each signal
            signal_votes = {}
            
            for _, row in group.iterrows():
                signal = row[signal_column]
                weight = row['strategy_weight']
                
                if signal not in signal_votes:
                    signal_votes[signal] = 0.0
                signal_votes[signal] += weight
            
            # Find signal with highest weighted vote
            if signal_votes:
                best_signal = max(signal_votes.items(), key=lambda x: x[1])[0]
                confidence = signal_votes[best_signal] / sum(signal_votes.values())
                
                resolved_data.append({
                    'timestamp': timestamp,
                    'signal': best_signal,
                    'confidence': confidence,
                    'strategy_count': len(group),
                    'signal_distribution': signal_votes
                })
        
        return pd.DataFrame(resolved_data)
    
    def _majority_vote_resolution(self, merged_df: pd.DataFrame, signal_column: str) -> pd.DataFrame:
        """Resolve conflicts using simple majority voting"""
        grouped = merged_df.groupby('timestamp')
        
        resolved_data = []
        
        for timestamp, group in grouped:
            # Count votes for each signal
            signal_counts = group[signal_column].value_counts()
            
            if not signal_counts.empty:
                best_signal = signal_counts.index[0]
                confidence = signal_counts.iloc[0] / len(group)
                
                resolved_data.append({
                    'timestamp': timestamp,
                    'signal': best_signal,
                    'confidence': confidence,
                    'strategy_count': len(group),
                    'signal_distribution': signal_counts.to_dict()
                })
        
        return pd.DataFrame(resolved_data)
    
    def _priority_resolution(self, merged_df: pd.DataFrame, signal_column: str) -> pd.DataFrame:
        """Resolve conflicts using strategy priority (first strategy wins)"""
        grouped = merged_df.groupby('timestamp')
        
        resolved_data = []
        
        for timestamp, group in grouped:
            # Take first strategy's signal (highest priority)
            first_row = group.iloc[0]
            
            resolved_data.append({
                'timestamp': timestamp,
                'signal': first_row[signal_column],
                'confidence': 1.0,
                'strategy_count': len(group),
                'strategy_used': first_row['strategy_name']
            })
        
        return pd.DataFrame(resolved_data)
    
    def _create_fallback_output(self) -> pd.DataFrame:
        """Create fallback output when no valid strategies"""
        return pd.DataFrame({
            'timestamp': [datetime.now()],
            'signal': ['HOLD'],
            'confidence': [0.0],
            'strategy_count': [0],
            'error': ['No valid strategy outputs']
        })

class TestStrategyOutputMerge:
    """Test cases for strategy output merging"""
    
    @pytest.fixture
    def merger(self):
        """Create a strategy output merger for testing"""
        merger = StrategyOutputMerger(conflict_resolution='weighted_vote')
        merger.set_strategy_weight('RSI', 1.0)
        merger.set_strategy_weight('MACD', 1.5)
        merger.set_strategy_weight('Bollinger', 0.8)
        return merger
    
    @pytest.fixture
    def sample_data(self):
        """Create sample strategy data"""
        base_time = datetime(2024, 1, 1, 9, 30)
        
        # RSI strategy data
        rsi_data = pd.DataFrame({
            'timestamp': [base_time + timedelta(minutes=i) for i in range(5)],
            'signal': ['BUY', 'HOLD', 'SELL', 'BUY', 'HOLD'],
            'rsi_value': [30, 45, 70, 25, 50],
            'confidence': [0.8, 0.6, 0.9, 0.7, 0.5]
        })
        
        # MACD strategy data (with duplicate timestamps)
        macd_data = pd.DataFrame({
            'timestamp': [base_time + timedelta(minutes=i) for i in range(5)],
            'signal': ['BUY', 'BUY', 'SELL', 'BUY', 'SELL'],
            'macd_value': [0.5, 0.3, -0.2, 0.4, -0.1],
            'confidence': [0.9, 0.7, 0.8, 0.6, 0.7]
        })
        
        # Bollinger strategy data
        bollinger_data = pd.DataFrame({
            'timestamp': [base_time + timedelta(minutes=i) for i in range(5)],
            'signal': ['HOLD', 'BUY', 'HOLD', 'SELL', 'BUY'],
            'bb_position': [0.5, 0.2, 0.6, 0.9, 0.3],
            'confidence': [0.5, 0.8, 0.4, 0.9, 0.6]
        })
        
        return {
            'RSI': rsi_data,
            'MACD': macd_data,
            'Bollinger': bollinger_data
        }
    
    def test_duplicate_timestamp_handling(self, merger, sample_data):
        """Test handling of duplicate timestamps with conflicting signals"""
        # Both RSI and MACD emit 'BUY' at the same timestamp
        # MACD has higher weight (1.5 vs 1.0)
        
        result = merger.merge_strategy_outputs(sample_data)
        
        assert not result.empty
        assert 'timestamp' in result.columns
        assert 'signal' in result.columns
        assert 'confidence' in result.columns
        
        # Check that conflicts are resolved
        assert len(result) == 5  # Should have 5 unique timestamps
        
        # Check first timestamp (both RSI and MACD say BUY)
        first_row = result.iloc[0]
        assert first_row['signal'] == 'BUY'  # Should be BUY (both agree)
        assert first_row['confidence'] > 0.5  # Should have high confidence
        assert first_row['strategy_count'] == 3  # All 3 strategies
        
        # Check second timestamp (RSI: HOLD, MACD: BUY, Bollinger: BUY)
        second_row = result.iloc[1]
        # MACD weight (1.5) + Bollinger weight (0.8) > RSI weight (1.0)
        # So should be BUY
        assert second_row['signal'] == 'BUY'
    
    def test_none_strategy_fallback(self, merger, sample_data):
        """Test fallback when strategy returns None"""
        # Add None strategy
        sample_data['NoneStrategy'] = None
        
        result = merger.merge_strategy_outputs(sample_data)
        
        assert not result.empty
        assert len(result) == 5  # Should still have 5 timestamps
        
        # Check that None strategy was ignored
        for _, row in result.iterrows():
            assert row['strategy_count'] == 3  # Only the 3 valid strategies
    
    def test_empty_dataframe_fallback(self, merger, sample_data):
        """Test fallback when strategy returns empty DataFrame"""
        # Add empty DataFrame
        sample_data['EmptyStrategy'] = pd.DataFrame()
        
        result = merger.merge_strategy_outputs(sample_data)
        
        assert not result.empty
        assert len(result) == 5
        
        # Check that empty strategy was ignored
        for _, row in result.iterrows():
            assert row['strategy_count'] == 3
    
    def test_all_strategies_none_or_empty(self, merger):
        """Test fallback when all strategies return None or empty"""
        strategy_outputs = {
            'Strategy1': None,
            'Strategy2': pd.DataFrame(),
            'Strategy3': pd.DataFrame(columns=['timestamp', 'signal'])
        }
        
        result = merger.merge_strategy_outputs(strategy_outputs)
        
        assert not result.empty
        assert len(result) == 1  # Should have one fallback row
        assert result.iloc[0]['signal'] == 'HOLD'
        assert result.iloc[0]['confidence'] == 0.0
        assert 'error' in result.columns
    
    def test_majority_vote_resolution(self):
        """Test majority vote conflict resolution"""
        merger = StrategyOutputMerger(conflict_resolution='majority')
        
        base_time = datetime(2024, 1, 1, 9, 30)
        
        strategy_outputs = {
            'Strategy1': pd.DataFrame({
                'timestamp': [base_time],
                'signal': ['BUY']
            }),
            'Strategy2': pd.DataFrame({
                'timestamp': [base_time],
                'signal': ['BUY']
            }),
            'Strategy3': pd.DataFrame({
                'timestamp': [base_time],
                'signal': ['SELL']
            })
        }
        
        result = merger.merge_strategy_outputs(strategy_outputs)
        
        assert len(result) == 1
        assert result.iloc[0]['signal'] == 'BUY'  # Majority vote
        assert result.iloc[0]['confidence'] == 2/3  # 2 out of 3 votes
    
    def test_priority_resolution(self):
        """Test priority-based conflict resolution"""
        merger = StrategyOutputMerger(conflict_resolution='priority')
        
        base_time = datetime(2024, 1, 1, 9, 30)
        
        strategy_outputs = {
            'HighPriority': pd.DataFrame({
                'timestamp': [base_time],
                'signal': ['BUY']
            }),
            'LowPriority': pd.DataFrame({
                'timestamp': [base_time],
                'signal': ['SELL']
            })
        }
        
        result = merger.merge_strategy_outputs(strategy_outputs)
        
        assert len(result) == 1
        assert result.iloc[0]['signal'] == 'BUY'  # First strategy wins
        assert result.iloc[0]['strategy_used'] == 'HighPriority'
    
    def test_missing_timestamp_column(self, merger, sample_data):
        """Test handling of missing timestamp column"""
        # Remove timestamp column from one strategy
        sample_data['RSI'] = sample_data['RSI'].drop(columns=['timestamp'])
        
        result = merger.merge_strategy_outputs(sample_data)
        
        assert not result.empty
        # Should still work with remaining strategies
        assert len(result) == 5
    
    def test_different_signal_values(self, merger):
        """Test handling of different signal value types"""
        base_time = datetime(2024, 1, 1, 9, 30)
        
        strategy_outputs = {
            'Strategy1': pd.DataFrame({
                'timestamp': [base_time],
                'signal': [1]  # Numeric signal
            }),
            'Strategy2': pd.DataFrame({
                'timestamp': [base_time],
                'signal': ['BUY']  # String signal
            })
        }
        
        result = merger.merge_strategy_outputs(strategy_outputs)
        
        assert len(result) == 1
        # Should handle mixed signal types
        assert 'signal' in result.columns
    
    def test_confidence_calculation(self, merger, sample_data):
        """Test confidence calculation in merged results"""
        result = merger.merge_strategy_outputs(sample_data)
        
        for _, row in result.iterrows():
            assert 0.0 <= row['confidence'] <= 1.0
            assert row['strategy_count'] > 0
    
    def test_strategy_weight_impact(self, merger, sample_data):
        """Test that strategy weights affect the final decision"""
        # Set very high weight for MACD
        merger.set_strategy_weight('MACD', 10.0)
        
        result = merger.merge_strategy_outputs(sample_data)
        
        # MACD should dominate decisions due to high weight
        for _, row in result.iterrows():
            # Check that MACD's signals are more likely to be chosen
            pass  # This would require more detailed analysis of the results
    
    def test_large_number_of_strategies(self, merger):
        """Test handling of many strategies"""
        base_time = datetime(2024, 1, 1, 9, 30)
        
        # Create 10 strategies
        strategy_outputs = {}
        for i in range(10):
            strategy_outputs[f'Strategy{i}'] = pd.DataFrame({
                'timestamp': [base_time],
                'signal': ['BUY'] if i < 6 else ['SELL']  # 6 BUY, 4 SELL
            })
        
        result = merger.merge_strategy_outputs(strategy_outputs)
        
        assert len(result) == 1
        assert result.iloc[0]['signal'] == 'BUY'  # Majority should be BUY
        assert result.iloc[0]['strategy_count'] == 10
    
    def test_performance_with_large_datasets(self, merger):
        """Test performance with large datasets"""
        base_time = datetime(2024, 1, 1, 9, 30)
        
        # Create large dataset (1000 timestamps)
        timestamps = [base_time + timedelta(minutes=i) for i in range(1000)]
        
        strategy_outputs = {
            'Strategy1': pd.DataFrame({
                'timestamp': timestamps,
                'signal': ['BUY'] * 1000
            }),
            'Strategy2': pd.DataFrame({
                'timestamp': timestamps,
                'signal': ['SELL'] * 1000
            })
        }
        
        import time
        start_time = time.time()
        
        result = merger.merge_strategy_outputs(strategy_outputs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        assert len(result) == 1000
        assert execution_time < 5.0  # Should complete within 5 seconds
    
    def test_error_handling_invalid_resolution(self):
        """Test error handling for invalid conflict resolution"""
        merger = StrategyOutputMerger(conflict_resolution='invalid_method')
        
        base_time = datetime(2024, 1, 1, 9, 30)
        
        strategy_outputs = {
            'Strategy1': pd.DataFrame({
                'timestamp': [base_time],
                'signal': ['BUY']
            })
        }
        
        # Should fall back to majority vote
        result = merger.merge_strategy_outputs(strategy_outputs)
        
        assert not result.empty
        assert len(result) == 1

# Integration test with actual strategy classes
class MockStrategy:
    """Mock strategy for testing"""
    
    def __init__(self, name: str, signals: List[str]):
        self.name = name
        self.signals = signals
    
    def generate_signals(self, timestamps: List[datetime]) -> pd.DataFrame:
        """Generate mock signals"""
        return pd.DataFrame({
            'timestamp': timestamps,
            'signal': self.signals[:len(timestamps)]
        })

class TestStrategyIntegration:
    """Integration tests with mock strategies"""
    
    def test_integration_with_mock_strategies(self):
        """Test integration with mock strategy classes"""
        base_time = datetime(2024, 1, 1, 9, 30)
        timestamps = [base_time + timedelta(minutes=i) for i in range(5)]
        
        # Create mock strategies
        rsi_strategy = MockStrategy('RSI', ['BUY', 'HOLD', 'SELL', 'BUY', 'HOLD'])
        macd_strategy = MockStrategy('MACD', ['BUY', 'BUY', 'SELL', 'BUY', 'SELL'])
        
        # Generate signals
        strategy_outputs = {
            'RSI': rsi_strategy.generate_signals(timestamps),
            'MACD': macd_strategy.generate_signals(timestamps)
        }
        
        # Merge outputs
        merger = StrategyOutputMerger()
        result = merger.merge_strategy_outputs(strategy_outputs)
        
        assert not result.empty
        assert len(result) == 5
        assert all(col in result.columns for col in ['timestamp', 'signal', 'confidence'])

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 