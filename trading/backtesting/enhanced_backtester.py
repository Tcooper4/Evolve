"""
Enhanced Backtesting Engine

This module provides an enhanced backtesting system that:
- Automatically runs backtests on forecasted signals
- Integrates with the unified trade reporting engine
- Provides comprehensive performance analysis
- Supports multiple strategies and model combinations
- Generates detailed trade logs and equity curves
- Includes advanced risk management features
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np

# Local imports
from .backtester import Backtester
from .performance_analysis import PerformanceAnalyzer
from .risk_metrics import RiskMetricsEngine
from .trade_models import Trade, TradeType
from ..report.unified_trade_reporter import UnifiedTradeReporter, generate_unified_report
from ..models.base_model import BaseModel
from ..strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class EnhancedBacktester:
    """
    Enhanced backtesting engine with automatic signal integration and comprehensive reporting.
    
    This class extends the basic backtester with:
    - Automatic integration with forecasting models
    - Signal generation from model predictions
    - Comprehensive performance analysis
    - Unified reporting integration
    - Advanced risk management
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_cash: float = 100000.0,
                 risk_free_rate: float = 0.02,
                 trading_days_per_year: int = 252,
                 output_dir: str = "backtest_results",
                 **kwargs):
        """
        Initialize the EnhancedBacktester.
        
        Args:
            data: Historical price data
            initial_cash: Starting cash amount
            risk_free_rate: Risk-free rate for calculations
            trading_days_per_year: Trading days per year
            output_dir: Directory for output files
            **kwargs: Additional backtester parameters
        """
        self.data = data
        self.initial_cash = initial_cash
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.backtester = Backtester(data, initial_cash=initial_cash, **kwargs)
        self.performance_analyzer = PerformanceAnalyzer()
        self.risk_metrics_engine = RiskMetricsEngine(
            risk_free_rate=risk_free_rate,
            period=trading_days_per_year
        )
        self.reporter = UnifiedTradeReporter(
            output_dir=str(self.output_dir / "reports"),
            risk_free_rate=risk_free_rate,
            trading_days_per_year=trading_days_per_year
        )
        
        # Results storage
        self.backtest_results: Dict[str, Any] = {}
        self.trade_logs: Dict[str, List[Dict[str, Any]]] = {}
        self.equity_curves: Dict[str, pd.DataFrame] = {}
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        
        logger.info("EnhancedBacktester initialized")
    
    def run_forecast_backtest(self,
                             model: BaseModel,
                             strategy: BaseStrategy,
                             symbol: str,
                             timeframe: str = "1d",
                             forecast_period: int = 30,
                             confidence_threshold: float = 0.6,
                             **kwargs) -> Dict[str, Any]:
        """
        Run backtest using forecasted signals from a model.
        
        Args:
            model: Forecasting model to generate signals
            strategy: Trading strategy to execute
            symbol: Trading symbol
            timeframe: Data timeframe
            forecast_period: Number of periods to forecast
            confidence_threshold: Minimum confidence for signal execution
            **kwargs: Additional parameters
            
        Returns:
            Comprehensive backtest results
        """
        try:
            logger.info(f"Running forecast backtest for {symbol} with {model.__class__.__name__}")
            
            # Generate forecasts
            forecasts = self._generate_forecasts(model, forecast_period)
            
            # Generate signals from forecasts
            signals = self._generate_signals_from_forecasts(forecasts, strategy, confidence_threshold)
            
            # Execute backtest with signals
            results = self._execute_backtest_with_signals(signals, strategy, symbol, **kwargs)
            
            # Generate comprehensive report
            report = self._generate_backtest_report(results, model, strategy, symbol, timeframe)
            
            # Store results
            test_id = f"{symbol}_{model.__class__.__name__}_{strategy.__class__.__name__}_{int(time.time())}"
            self.backtest_results[test_id] = {
                'results': results,
                'report': report,
                'model': model.__class__.__name__,
                'strategy': strategy.__class__.__name__,
                'symbol': symbol,
                'timeframe': timeframe,
                'forecast_period': forecast_period,
                'confidence_threshold': confidence_threshold
            }
            
            logger.info(f"Forecast backtest completed: {test_id}")
            return self.backtest_results[test_id]
            
        except Exception as e:
            logger.error(f"Error in forecast backtest: {e}")
            return self._generate_error_result(str(e))
    
    def run_multi_model_backtest(self,
                                models: List[BaseModel],
                                strategy: BaseStrategy,
                                symbol: str,
                                timeframe: str = "1d",
                                forecast_period: int = 30,
                                ensemble_method: str = "weighted",
                                **kwargs) -> Dict[str, Any]:
        """
        Run backtest using multiple models with ensemble forecasting.
        
        Args:
            models: List of forecasting models
            strategy: Trading strategy to execute
            symbol: Trading symbol
            timeframe: Data timeframe
            forecast_period: Number of periods to forecast
            ensemble_method: Method for combining forecasts ('weighted', 'voting', 'average')
            **kwargs: Additional parameters
            
        Returns:
            Comprehensive backtest results with ensemble analysis
        """
        try:
            logger.info(f"Running multi-model backtest for {symbol} with {len(models)} models")
            
            # Generate ensemble forecasts
            ensemble_forecasts = self._generate_ensemble_forecasts(
                models, forecast_period, ensemble_method
            )
            
            # Generate signals from ensemble forecasts
            signals = self._generate_signals_from_forecasts(
                ensemble_forecasts, strategy, confidence_threshold=0.5
            )
            
            # Execute backtest
            results = self._execute_backtest_with_signals(signals, strategy, symbol, **kwargs)
            
            # Generate comprehensive report
            report = self._generate_ensemble_backtest_report(
                results, models, strategy, symbol, timeframe, ensemble_method
            )
            
            # Store results
            test_id = f"{symbol}_ensemble_{strategy.__class__.__name__}_{int(time.time())}"
            self.backtest_results[test_id] = {
                'results': results,
                'report': report,
                'models': [m.__class__.__name__ for m in models],
                'strategy': strategy.__class__.__name__,
                'symbol': symbol,
                'timeframe': timeframe,
                'ensemble_method': ensemble_method
            }
            
            logger.info(f"Multi-model backtest completed: {test_id}")
            return self.backtest_results[test_id]
            
        except Exception as e:
            logger.error(f"Error in multi-model backtest: {e}")
            return self._generate_error_result(str(e))
    
    def run_strategy_comparison(self,
                               models: List[BaseModel],
                               strategies: List[BaseStrategy],
                               symbol: str,
                               timeframe: str = "1d",
                               forecast_period: int = 30,
                               **kwargs) -> Dict[str, Any]:
        """
        Run comprehensive comparison of multiple model-strategy combinations.
        
        Args:
            models: List of forecasting models
            strategies: List of trading strategies
            symbol: Trading symbol
            timeframe: Data timeframe
            forecast_period: Number of periods to forecast
            **kwargs: Additional parameters
            
        Returns:
            Comprehensive comparison results
        """
        try:
            logger.info(f"Running strategy comparison for {symbol}")
            
            comparison_results = {}
            
            # Run all combinations
            for model in models:
                for strategy in strategies:
                    try:
                        result = self.run_forecast_backtest(
                            model, strategy, symbol, timeframe, forecast_period, **kwargs
                        )
                        key = f"{model.__class__.__name__}_{strategy.__class__.__name__}"
                        comparison_results[key] = result
                    except Exception as e:
                        logger.warning(f"Failed to run {model.__class__.__name__} + {strategy.__class__.__name__}: {e}")
                        comparison_results[f"{model.__class__.__name__}_{strategy.__class__.__name__}"] = {
                            'error': str(e)
                        }
            
            # Generate comparison report
            comparison_report = self._generate_comparison_report(comparison_results, symbol)
            
            # Store comparison results
            test_id = f"{symbol}_comparison_{int(time.time())}"
            self.backtest_results[test_id] = {
                'comparison_results': comparison_results,
                'comparison_report': comparison_report,
                'models': [m.__class__.__name__ for m in models],
                'strategies': [s.__class__.__name__ for s in strategies],
                'symbol': symbol,
                'timeframe': timeframe
            }
            
            logger.info(f"Strategy comparison completed: {test_id}")
            return self.backtest_results[test_id]
            
        except Exception as e:
            logger.error(f"Error in strategy comparison: {e}")
            return self._generate_error_result(str(e))
    
    def _generate_forecasts(self, model: BaseModel, forecast_period: int) -> Dict[str, Any]:
        """Generate forecasts using the specified model."""
        try:
            # Prepare data for forecasting
            if hasattr(model, 'prepare_data'):
                prepared_data = model.prepare_data(self.data)
            else:
                prepared_data = self.data
            
            # Generate forecasts
            if hasattr(model, 'forecast'):
                forecasts = model.forecast(prepared_data, periods=forecast_period)
            elif hasattr(model, 'predict'):
                # For models that use predict instead of forecast
                forecasts = model.predict(prepared_data)
            else:
                raise ValueError(f"Model {model.__class__.__name__} does not support forecasting")
            
            return {
                'forecasts': forecasts,
                'model_name': model.__class__.__name__,
                'forecast_period': forecast_period,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating forecasts: {e}")
            return {'error': str(e)}
    
    def _generate_ensemble_forecasts(self,
                                   models: List[BaseModel],
                                   forecast_period: int,
                                   ensemble_method: str) -> Dict[str, Any]:
        """Generate ensemble forecasts from multiple models."""
        try:
            individual_forecasts = []
            
            # Generate forecasts from each model
            for model in models:
                try:
                    forecast = self._generate_forecasts(model, forecast_period)
                    if 'error' not in forecast:
                        individual_forecasts.append(forecast)
                except Exception as e:
                    logger.warning(f"Failed to generate forecast for {model.__class__.__name__}: {e}")
            
            if not individual_forecasts:
                raise ValueError("No valid forecasts generated from any model")
            
            # Combine forecasts based on ensemble method
            if ensemble_method == 'weighted':
                combined_forecasts = self._weighted_ensemble(individual_forecasts)
            elif ensemble_method == 'voting':
                combined_forecasts = self._voting_ensemble(individual_forecasts)
            elif ensemble_method == 'average':
                combined_forecasts = self._average_ensemble(individual_forecasts)
            else:
                raise ValueError(f"Unknown ensemble method: {ensemble_method}")
            
            return {
                'ensemble_forecasts': combined_forecasts,
                'individual_forecasts': individual_forecasts,
                'ensemble_method': ensemble_method,
                'forecast_period': forecast_period,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating ensemble forecasts: {e}")
            return {'error': str(e)}
    
    def _weighted_ensemble(self, individual_forecasts: List[Dict[str, Any]]) -> np.ndarray:
        """Combine forecasts using weighted average based on model performance."""
        try:
            # Simple equal weighting for now - could be enhanced with performance-based weights
            weights = np.ones(len(individual_forecasts)) / len(individual_forecasts)
            
            combined = np.zeros_like(individual_forecasts[0]['forecasts'])
            for i, forecast in enumerate(individual_forecasts):
                combined += weights[i] * forecast['forecasts']
            
            return combined
            
        except Exception as e:
            logger.error(f"Error in weighted ensemble: {e}")
            return individual_forecasts[0]['forecasts'] if individual_forecasts else np.array([])
    
    def _voting_ensemble(self, individual_forecasts: List[Dict[str, Any]]) -> np.ndarray:
        """Combine forecasts using voting mechanism."""
        try:
            # Convert forecasts to binary signals (positive/negative)
            signals = []
            for forecast in individual_forecasts:
                signal = np.where(forecast['forecasts'] > 0, 1, -1)
                signals.append(signal)
            
            # Majority vote
            combined_signal = np.mean(signals, axis=0)
            combined_forecast = np.where(combined_signal > 0, 1, -1)
            
            return combined_forecast
            
        except Exception as e:
            logger.error(f"Error in voting ensemble: {e}")
            return individual_forecasts[0]['forecasts'] if individual_forecasts else np.array([])
    
    def _average_ensemble(self, individual_forecasts: List[Dict[str, Any]]) -> np.ndarray:
        """Combine forecasts using simple average."""
        try:
            combined = np.mean([f['forecasts'] for f in individual_forecasts], axis=0)
            return combined
            
        except Exception as e:
            logger.error(f"Error in average ensemble: {e}")
            return individual_forecasts[0]['forecasts'] if individual_forecasts else np.array([])
    
    def _generate_signals_from_forecasts(self,
                                       forecasts: Dict[str, Any],
                                       strategy: BaseStrategy,
                                       confidence_threshold: float) -> List[Dict[str, Any]]:
        """Generate trading signals from forecasts using the strategy."""
        try:
            if 'error' in forecasts:
                return []
            
            forecast_data = forecasts.get('forecasts') or forecasts.get('ensemble_forecasts')
            if forecast_data is None:
                return []
            
            signals = []
            
            # Generate signals for each forecast period
            for i, forecast_value in enumerate(forecast_data):
                try:
                    # Create signal data structure
                    signal_data = {
                        'timestamp': datetime.now() + timedelta(days=i),
                        'forecast_value': forecast_value,
                        'confidence': abs(forecast_value),  # Simple confidence measure
                        'period': i
                    }
                    
                    # Apply strategy logic
                    if hasattr(strategy, 'generate_signal'):
                        signal = strategy.generate_signal(signal_data)
                    else:
                        # Default signal generation
                        signal = self._default_signal_generation(signal_data, confidence_threshold)
                    
                    if signal and signal.get('confidence', 0) >= confidence_threshold:
                        signals.append(signal)
                        
                except Exception as e:
                    logger.warning(f"Error generating signal for period {i}: {e}")
                    continue
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals from forecasts: {e}")
            return []
    
    def _default_signal_generation(self,
                                 signal_data: Dict[str, Any],
                                 confidence_threshold: float) -> Optional[Dict[str, Any]]:
        """Default signal generation logic."""
        try:
            forecast_value = signal_data['forecast_value']
            confidence = signal_data['confidence']
            
            if confidence < confidence_threshold:
                return None
            
            # Simple signal logic: positive forecast = buy, negative = sell
            if forecast_value > 0:
                signal_type = 'BUY'
            elif forecast_value < 0:
                signal_type = 'SELL'
            else:
                return None
            
            return {
                'timestamp': signal_data['timestamp'],
                'type': signal_type,
                'confidence': confidence,
                'forecast_value': forecast_value,
                'period': signal_data['period']
            }
            
        except Exception as e:
            logger.error(f"Error in default signal generation: {e}")
            return None
    
    def _execute_backtest_with_signals(self,
                                     signals: List[Dict[str, Any]],
                                     strategy: BaseStrategy,
                                     symbol: str,
                                     **kwargs) -> Dict[str, Any]:
        """Execute backtest using generated signals."""
        try:
            # Reset backtester
            self.backtester.reset()
            
            # Execute trades based on signals
            for signal in signals:
                try:
                    # Get current price (simplified - would need actual price data)
                    current_price = self._get_price_at_timestamp(signal['timestamp'])
                    if current_price is None:
                        continue
                    
                    # Calculate position size
                    position_size = self._calculate_position_size(signal, current_price)
                    
                    # Execute trade
                    if signal['type'] == 'BUY':
                        trade_type = TradeType.BUY
                    elif signal['type'] == 'SELL':
                        trade_type = TradeType.SELL
                    else:
                        continue
                    
                    trade = self.backtester.execute_trade(
                        timestamp=signal['timestamp'],
                        asset=symbol,
                        quantity=position_size,
                        price=current_price,
                        trade_type=trade_type,
                        strategy=strategy.__class__.__name__,
                        signal=signal['confidence']
                    )
                    
                except Exception as e:
                    logger.warning(f"Error executing trade for signal: {e}")
                    continue
            
            # Get backtest results
            performance_metrics = self.backtester.get_performance_metrics()
            trade_summary = self.backtester.get_trade_summary()
            equity_curve = self.backtester._calculate_equity_curve()
            
            return {
                'performance_metrics': performance_metrics,
                'trade_summary': trade_summary,
                'equity_curve': equity_curve,
                'trades': self.backtester.trade_log,
                'signals': signals,
                'strategy': strategy.__class__.__name__,
                'symbol': symbol
            }
            
        except Exception as e:
            logger.error(f"Error executing backtest with signals: {e}")
            return {'error': str(e)}
    
    def _get_price_at_timestamp(self, timestamp: datetime) -> Optional[float]:
        """Get price at specific timestamp (simplified implementation)."""
        try:
            # This is a simplified implementation
            # In a real system, you would look up the actual price data
            if len(self.data) > 0:
                # Use the last available price as approximation
                return self.data.iloc[-1].iloc[0] if len(self.data.columns) > 0 else 100.0
            return 100.0  # Default price
        except Exception as e:
            logger.error(f"Error getting price at timestamp: {e}")
            return None
    
    def _calculate_position_size(self, signal: Dict[str, Any], price: float) -> float:
        """Calculate position size based on signal and current price."""
        try:
            # Simple position sizing: use confidence as position size factor
            confidence = signal.get('confidence', 0.5)
            base_position = self.initial_cash * 0.1  # 10% of initial cash per trade
            position_size = (base_position * confidence) / price
            return max(position_size, 0)  # Ensure non-negative
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def _generate_backtest_report(self,
                                results: Dict[str, Any],
                                model: BaseModel,
                                strategy: BaseStrategy,
                                symbol: str,
                                timeframe: str) -> Dict[str, Any]:
        """Generate comprehensive backtest report."""
        try:
            if 'error' in results:
                return {'error': results['error']}
            
            # Prepare trade data for reporting
            trade_data = {
                'trades': results.get('trades', []),
                'equity_curve': results.get('equity_curve', pd.DataFrame())
            }
            
            # Prepare model data
            model_data = {
                'model_name': model.__class__.__name__,
                'model_type': 'forecasting',
                'performance_metrics': results.get('performance_metrics', {})
            }
            
            # Prepare strategy data
            strategy_data = {
                'strategy_name': strategy.__class__.__name__,
                'symbol': symbol,
                'timeframe': timeframe,
                'signals': results.get('signals', []),
                'performance': results.get('trade_summary', {})
            }
            
            # Generate unified report
            report = self.reporter.generate_comprehensive_report(
                trade_data=trade_data,
                model_data=model_data,
                strategy_data=strategy_data,
                symbol=symbol,
                timeframe=timeframe,
                period=f"Forecast backtest - {len(results.get('signals', []))} signals"
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating backtest report: {e}")
            return {'error': str(e)}
    
    def _generate_ensemble_backtest_report(self,
                                         results: Dict[str, Any],
                                         models: List[BaseModel],
                                         strategy: BaseStrategy,
                                         symbol: str,
                                         timeframe: str,
                                         ensemble_method: str) -> Dict[str, Any]:
        """Generate comprehensive ensemble backtest report."""
        try:
            # Similar to regular backtest report but with ensemble information
            report = self._generate_backtest_report(results, models[0], strategy, symbol, timeframe)
            
            # Add ensemble-specific information
            if 'error' not in report:
                report['ensemble_info'] = {
                    'models': [m.__class__.__name__ for m in models],
                    'ensemble_method': ensemble_method,
                    'model_count': len(models)
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating ensemble backtest report: {e}")
            return {'error': str(e)}
    
    def _generate_comparison_report(self,
                                  comparison_results: Dict[str, Any],
                                  symbol: str) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        try:
            # Extract performance metrics for comparison
            comparison_data = {}
            
            for key, result in comparison_results.items():
                if 'error' not in result and 'results' in result:
                    metrics = result['results'].get('performance_metrics', {})
                    comparison_data[key] = {
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'total_return': metrics.get('total_return', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'total_trades': metrics.get('num_trades', 0)
                    }
            
            # Find best performing combination
            if comparison_data:
                best_sharpe = max(comparison_data.keys(), 
                                key=lambda k: comparison_data[k].get('sharpe_ratio', 0))
                best_return = max(comparison_data.keys(),
                                key=lambda k: comparison_data[k].get('total_return', 0))
                
                comparison_summary = {
                    'best_sharpe': {
                        'combination': best_sharpe,
                        'metrics': comparison_data[best_sharpe]
                    },
                    'best_return': {
                        'combination': best_return,
                        'metrics': comparison_data[best_return]
                    },
                    'all_combinations': comparison_data
                }
            else:
                comparison_summary = {'error': 'No valid results for comparison'}
            
            return {
                'comparison_summary': comparison_summary,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating comparison report: {e}")
            return {'error': str(e)}
    
    def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        """Generate error result structure."""
        return {
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'results': {},
            'report': {'error': error_message}
        }
    
    def get_all_results(self) -> Dict[str, Any]:
        """Get all backtest results."""
        return self.backtest_results
    
    def save_results(self, filepath: Optional[str] = None) -> str:
        """Save all backtest results to file."""
        try:
            filepath = filepath or str(self.output_dir / f"backtest_results_{int(time.time())}.json")
            
            # Convert results to serializable format
            serializable_results = {}
            for key, value in self.backtest_results.items():
                try:
                    # Convert numpy arrays and other non-serializable objects
                    serializable_results[key] = self._make_serializable(value)
                except Exception as e:
                    logger.warning(f"Could not serialize result {key}: {e}")
                    serializable_results[key] = {'error': f'Serialization failed: {e}'}
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Backtest results saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return ""
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj

# Convenience functions
def run_forecast_backtest(data: pd.DataFrame,
                         model: BaseModel,
                         strategy: BaseStrategy,
                         symbol: str,
                         **kwargs) -> Dict[str, Any]:
    """Convenience function to run forecast backtest."""
    backtester = EnhancedBacktester(data, **kwargs)
    return backtester.run_forecast_backtest(model, strategy, symbol, **kwargs)

def run_multi_model_backtest(data: pd.DataFrame,
                           models: List[BaseModel],
                           strategy: BaseStrategy,
                           symbol: str,
                           **kwargs) -> Dict[str, Any]:
    """Convenience function to run multi-model backtest."""
    backtester = EnhancedBacktester(data, **kwargs)
    return backtester.run_multi_model_backtest(models, strategy, symbol, **kwargs)

def run_strategy_comparison(data: pd.DataFrame,
                          models: List[BaseModel],
                          strategies: List[BaseStrategy],
                          symbol: str,
                          **kwargs) -> Dict[str, Any]:
    """Convenience function to run strategy comparison."""
    backtester = EnhancedBacktester(data, **kwargs)
    return backtester.run_strategy_comparison(models, strategies, symbol, **kwargs) 