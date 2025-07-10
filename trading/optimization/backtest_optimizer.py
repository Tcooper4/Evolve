"""
Backtest Optimizer with Walk-Forward Analysis and Regime Detection

Advanced backtesting framework with walk-forward optimization, regime detection,
and robust performance evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

@dataclass
class WalkForwardResult:
    """Result from walk-forward analysis."""
    period_start: str
    period_end: str
    training_period: str
    validation_period: str
    best_params: Dict[str, Any]
    validation_performance: Dict[str, float]
    out_of_sample_performance: Dict[str, float]
    regime: str
    regime_confidence: float
    timestamp: str

@dataclass
class RegimeInfo:
    """Information about detected market regime."""
    regime_id: int
    regime_name: str
    start_date: str
    end_date: str
    characteristics: Dict[str, float]
    volatility: float
    trend_strength: float
    correlation_structure: Dict[str, float]
    duration_days: int
    confidence: float

class RegimeDetector:
    """Market regime detection using clustering and statistical methods."""
    
    def __init__(self, n_regimes: int = 3, lookback_window: int = 252):
        """Initialize regime detector.
        
        Args:
            n_regimes: Number of regimes to detect
            lookback_window: Rolling window for feature calculation
        """
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        self.regime_history = []
        self.logger = logging.getLogger(__name__)
    
    def calculate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for regime detection.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            DataFrame with regime features
        """
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        if 'close' in data.columns:
            returns = data['close'].pct_change()
            features['volatility'] = returns.rolling(self.lookback_window).std() * np.sqrt(252)
            features['returns'] = returns.rolling(self.lookback_window).mean() * 252
            features['skewness'] = returns.rolling(self.lookback_window).skew()
            features['kurtosis'] = returns.rolling(self.lookback_window).kurt()
            
            # Trend features
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean()
            features['trend_strength'] = (sma_20 - sma_50) / sma_50
            
            # Momentum features
            features['momentum'] = data['close'].pct_change(20)
            features['rsi'] = self._calculate_rsi(data['close'])
        
        # Volume-based features
        if 'volume' in data.columns:
            features['volume_ratio'] = data['volume'].rolling(self.lookback_window).mean() / data['volume'].rolling(self.lookback_window * 2).mean()
            features['volume_volatility'] = data['volume'].pct_change().rolling(self.lookback_window).std()
        
        # Volatility regime features
        if 'high' in data.columns and 'low' in data.columns:
            features['volatility_regime'] = (data['high'] - data['low']) / data['close']
            features['volatility_regime'] = features['volatility_regime'].rolling(self.lookback_window).mean()
        
        # Remove NaN values
        features = features.dropna()
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def detect_regimes(self, data: pd.DataFrame) -> List[RegimeInfo]:
        """Detect market regimes in the data.
        
        Args:
            data: Market data
            
        Returns:
            List of detected regimes
        """
        # Calculate features
        features = self.calculate_regime_features(data)
        
        if len(features) < self.lookback_window:
            self.logger.warning("Insufficient data for regime detection")
            return []
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Cluster to detect regimes
        cluster_labels = self.kmeans.fit_predict(features_scaled)
        
        # Create regime information
        regimes = []
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            regime_mask = cluster_labels == label
            regime_dates = features.index[regime_mask]
            
            if len(regime_dates) == 0:
                continue
            
            # Calculate regime characteristics
            regime_features = features.loc[regime_dates]
            regime_characteristics = {
                'mean_volatility': regime_features['volatility'].mean(),
                'mean_returns': regime_features['returns'].mean(),
                'mean_trend_strength': regime_features['trend_strength'].mean(),
                'mean_momentum': regime_features['momentum'].mean(),
                'mean_rsi': regime_features['rsi'].mean()
            }
            
            # Determine regime name based on characteristics
            regime_name = self._classify_regime(regime_characteristics)
            
            # Calculate confidence based on cluster stability
            confidence = self._calculate_regime_confidence(features_scaled, cluster_labels, label)
            
            regime = RegimeInfo(
                regime_id=int(label),
                regime_name=regime_name,
                start_date=regime_dates[0].isoformat(),
                end_date=regime_dates[-1].isoformat(),
                characteristics=regime_characteristics,
                volatility=regime_characteristics['mean_volatility'],
                trend_strength=regime_characteristics['mean_trend_strength'],
                correlation_structure={},  # Could be enhanced
                duration_days=len(regime_dates),
                confidence=confidence
            )
            regimes.append(regime)
        
        self.regime_history = regimes
        return regimes
    
    def _classify_regime(self, characteristics: Dict[str, float]) -> str:
        """Classify regime based on characteristics."""
        volatility = characteristics['mean_volatility']
        returns = characteristics['mean_returns']
        trend = characteristics['mean_trend_strength']
        
        if volatility > 0.25:
            if returns < -0.05:
                return "High Volatility Bear Market"
            elif returns > 0.05:
                return "High Volatility Bull Market"
            else:
                return "High Volatility Sideways"
        elif volatility < 0.15:
            if trend > 0.02:
                return "Low Volatility Bull Market"
            elif trend < -0.02:
                return "Low Volatility Bear Market"
            else:
                return "Low Volatility Sideways"
        else:
            if returns > 0.05:
                return "Moderate Bull Market"
            elif returns < -0.05:
                return "Moderate Bear Market"
            else:
                return "Moderate Sideways"
    
    def _calculate_regime_confidence(self, features_scaled: np.ndarray, 
                                   cluster_labels: np.ndarray, 
                                   regime_label: int) -> float:
        """Calculate confidence in regime classification."""
        regime_mask = cluster_labels == regime_label
        regime_features = features_scaled[regime_mask]
        
        if len(regime_features) == 0:
            return 0.0
        
        # Calculate silhouette score-like metric
        regime_center = self.kmeans.cluster_centers_[regime_label]
        intra_cluster_dist = np.mean(np.linalg.norm(regime_features - regime_center, axis=1))
        
        # Calculate distance to nearest other cluster
        other_centers = [self.kmeans.cluster_centers_[i] for i in range(self.n_regimes) if i != regime_label]
        if other_centers:
            min_inter_cluster_dist = min(np.linalg.norm(regime_center - other_center) for other_center in other_centers)
            confidence = min_inter_cluster_dist / (min_inter_cluster_dist + intra_cluster_dist)
        else:
            confidence = 1.0
        
        return min(confidence, 1.0)
    
    def get_current_regime(self, data: pd.DataFrame, lookback_days: int = 30) -> Optional[RegimeInfo]:
        """Get the current market regime.
        
        Args:
            data: Recent market data
            lookback_days: Number of days to look back
            
        Returns:
            Current regime information
        """
        if len(data) < lookback_days:
            return None
        
        recent_data = data.tail(lookback_days)
        features = self.calculate_regime_features(recent_data)
        
        if len(features) == 0:
            return None
        
        # Use the last available features
        latest_features = features.iloc[-1:].values
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # Predict regime
        regime_label = self.kmeans.predict(latest_features_scaled)[0]
        
        # Find corresponding regime info
        for regime in self.regime_history:
            if regime.regime_id == regime_label:
                return regime
        
        return None

class WalkForwardOptimizer:
    """Walk-forward optimization framework."""
    
    def __init__(self, 
                 training_window: int = 252,
                 validation_window: int = 63,
                 step_size: int = 21,
                 regime_detector: Optional[RegimeDetector] = None):
        """Initialize walk-forward optimizer.
        
        Args:
            training_window: Training period length in days
            validation_window: Validation period length in days
            step_size: Step size for moving window in days
            regime_detector: Optional regime detector
        """
        self.training_window = training_window
        self.validation_window = validation_window
        self.step_size = step_size
        self.regime_detector = regime_detector or RegimeDetector()
        self.results = []
        self.logger = logging.getLogger(__name__)
    
    def run_walk_forward_analysis(self, 
                                data: pd.DataFrame,
                                strategy_class: Any,
                                param_space: Dict[str, Any],
                                optimization_method: str = "bayesian",
                                **kwargs) -> List[WalkForwardResult]:
        """Run walk-forward analysis.
        
        Args:
            data: Market data
            strategy_class: Strategy class to optimize
            param_space: Parameter space for optimization
            optimization_method: Optimization method to use
            **kwargs: Additional optimization parameters
            
        Returns:
            List of walk-forward results
        """
        self.logger.info("Starting walk-forward analysis")
        
        # Detect regimes
        regimes = self.regime_detector.detect_regimes(data)
        self.logger.info(f"Detected {len(regimes)} market regimes")
        
        # Generate time windows
        windows = self._generate_windows(data.index)
        
        results = []
        for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
            self.logger.info(f"Processing window {i+1}/{len(windows)}")
            
            # Get current regime
            current_regime = self._get_regime_for_period(regimes, train_start, train_end)
            
            # Optimize strategy for training period
            best_params = self._optimize_strategy(
                data.loc[train_start:train_end],
                strategy_class,
                param_space,
                optimization_method,
                **kwargs
            )
            
            # Evaluate on validation period
            val_performance = self._evaluate_strategy(
                data.loc[val_start:val_end],
                strategy_class,
                best_params
            )
            
            # Evaluate out-of-sample (if possible)
            oos_performance = {}
            if val_end < data.index[-1]:
                oos_end = min(val_end + timedelta(days=self.validation_window), data.index[-1])
                oos_performance = self._evaluate_strategy(
                    data.loc[val_end:oos_end],
                    strategy_class,
                    best_params
                )
            
            # Create result
            result = WalkForwardResult(
                period_start=train_start.isoformat(),
                period_end=val_end.isoformat(),
                training_period=f"{train_start.isoformat()} to {train_end.isoformat()}",
                validation_period=f"{val_start.isoformat()} to {val_end.isoformat()}",
                best_params=best_params,
                validation_performance=val_performance,
                out_of_sample_performance=oos_performance,
                regime=current_regime.regime_name if current_regime else "Unknown",
                regime_confidence=current_regime.confidence if current_regime else 0.0,
                timestamp=datetime.utcnow().isoformat()
            )
            
            results.append(result)
        
        self.results = results
        self.logger.info(f"Completed walk-forward analysis with {len(results)} periods")
        return results
    
    def _generate_windows(self, dates: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate training and validation windows."""
        windows = []
        current_start = dates[0]
        
        while current_start + timedelta(days=self.training_window + self.validation_window) <= dates[-1]:
            train_end = current_start + timedelta(days=self.training_window - 1)
            val_start = train_end + timedelta(days=1)
            val_end = val_start + timedelta(days=self.validation_window - 1)
            
            if val_end <= dates[-1]:
                windows.append((current_start, train_end, val_start, val_end))
            
            current_start += timedelta(days=self.step_size)
        
        return windows
    
    def _get_regime_for_period(self, regimes: List[RegimeInfo], 
                              start_date: pd.Timestamp, 
                              end_date: pd.Timestamp) -> Optional[RegimeInfo]:
        """Get regime for a specific time period."""
        for regime in regimes:
            regime_start = pd.Timestamp(regime.start_date)
            regime_end = pd.Timestamp(regime.end_date)
            
            # Check if period overlaps with regime
            if (start_date <= regime_end and end_date >= regime_start):
                return regime
        
        return None
    
    def _optimize_strategy(self, 
                          data: pd.DataFrame,
                          strategy_class: Any,
                          param_space: Dict[str, Any],
                          optimization_method: str,
                          **kwargs) -> Dict[str, Any]:
        """Optimize strategy parameters."""
        # This would integrate with the existing optimization framework
        # For now, return default parameters
        return {k: v[0] if isinstance(v, (list, tuple)) else v for k, v in param_space.items()}
    
    def _evaluate_strategy(self, 
                          data: pd.DataFrame,
                          strategy_class: Any,
                          params: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate strategy performance."""
        # This would integrate with the existing strategy evaluation framework
        # For now, return mock performance metrics
        return {
            'sharpe_ratio': np.random.normal(0.5, 0.3),
            'total_return': np.random.normal(0.1, 0.05),
            'max_drawdown': np.random.normal(-0.1, 0.05),
            'win_rate': np.random.normal(0.55, 0.1)
        }
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze walk-forward results."""
        if not self.results:
            return {'error': 'No results available'}
        
        # Performance analysis
        val_sharpes = [r.validation_performance.get('sharpe_ratio', 0) for r in self.results]
        oos_sharpes = [r.out_of_sample_performance.get('sharpe_ratio', 0) for r in self.results if r.out_of_sample_performance]
        
        # Regime analysis
        regime_performance = {}
        for result in self.results:
            regime = result.regime
            if regime not in regime_performance:
                regime_performance[regime] = []
            regime_performance[regime].append(result.validation_performance.get('sharpe_ratio', 0))
        
        # Parameter stability analysis
        param_stability = self._analyze_parameter_stability()
        
        return {
            'summary': {
                'total_periods': len(self.results),
                'avg_validation_sharpe': np.mean(val_sharpes),
                'avg_oos_sharpe': np.mean(oos_sharpes) if oos_sharpes else None,
                'sharpe_degradation': np.mean(val_sharpes) - np.mean(oos_sharpes) if oos_sharpes else None
            },
            'regime_analysis': {
                regime: {
                    'count': len(perfs),
                    'avg_sharpe': np.mean(perfs),
                    'std_sharpe': np.std(perfs)
                }
                for regime, perfs in regime_performance.items()
            },
            'parameter_stability': param_stability
        }
    
    def _analyze_parameter_stability(self) -> Dict[str, Any]:
        """Analyze parameter stability across periods."""
        if not self.results:
            return {}
        
        # Get all parameters
        all_params = [result.best_params for result in self.results]
        param_names = list(all_params[0].keys())
        
        stability_metrics = {}
        for param_name in param_names:
            param_values = [params.get(param_name, 0) for params in all_params]
            
            stability_metrics[param_name] = {
                'mean': np.mean(param_values),
                'std': np.std(param_values),
                'cv': np.std(param_values) / np.mean(param_values) if np.mean(param_values) != 0 else 0,
                'min': np.min(param_values),
                'max': np.max(param_values)
            }
        
        return stability_metrics
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """Plot walk-forward analysis results."""
        if not self.results:
            self.logger.warning("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Walk-Forward Analysis Results', fontsize=16)
        
        # Plot 1: Performance over time
        periods = range(len(self.results))
        val_sharpes = [r.validation_performance.get('sharpe_ratio', 0) for r in self.results]
        oos_sharpes = [r.out_of_sample_performance.get('sharpe_ratio', 0) for r in self.results if r.out_of_sample_performance]
        
        axes[0, 0].plot(periods, val_sharpes, 'b-', label='Validation', linewidth=2)
        if oos_sharpes:
            axes[0, 0].plot(range(len(oos_sharpes)), oos_sharpes, 'r--', label='Out-of-Sample', linewidth=2)
        axes[0, 0].set_title('Sharpe Ratio Over Time')
        axes[0, 0].set_xlabel('Period')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Regime distribution
        regimes = [r.regime for r in self.results]
        regime_counts = pd.Series(regimes).value_counts()
        axes[0, 1].pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Regime Distribution')
        
        # Plot 3: Performance by regime
        regime_performance = {}
        for result in self.results:
            regime = result.regime
            if regime not in regime_performance:
                regime_performance[regime] = []
            regime_performance[regime].append(result.validation_performance.get('sharpe_ratio', 0))
        
        regime_names = list(regime_performance.keys())
        regime_means = [np.mean(perfs) for perfs in regime_performance.values()]
        
        axes[1, 0].bar(regime_names, regime_means, alpha=0.7)
        axes[1, 0].set_title('Average Performance by Regime')
        axes[1, 0].set_xlabel('Regime')
        axes[1, 0].set_ylabel('Average Sharpe Ratio')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Parameter stability
        if self.results:
            param_stability = self._analyze_parameter_stability()
            if param_stability:
                param_names = list(param_stability.keys())
                param_cvs = [param_stability[p]['cv'] for p in param_names]
                
                axes[1, 1].bar(param_names, param_cvs, alpha=0.7)
                axes[1, 1].set_title('Parameter Stability (Coefficient of Variation)')
                axes[1, 1].set_xlabel('Parameter')
                axes[1, 1].set_ylabel('CV (Lower = More Stable)')
                axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Walk-forward analysis plot saved to {save_path}")
        
        plt.show()
    
    def export_results(self, filepath: str) -> Dict[str, Any]:
        """Export walk-forward results to file."""
        try:
            # Prepare data for export
            export_data = {
                'metadata': {
                    'training_window': self.training_window,
                    'validation_window': self.validation_window,
                    'step_size': self.step_size,
                    'total_periods': len(self.results),
                    'timestamp': datetime.utcnow().isoformat()
                },
                'results': [asdict(result) for result in self.results],
                'analysis': self.analyze_results()
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Walk-forward results exported to {filepath}")
            return {'success': True, 'filepath': filepath}
            
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
            return {'success': False, 'error': str(e)}

class BacktestOptimizer:
    """Main backtest optimizer class."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize backtest optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.regime_detector = RegimeDetector(
            n_regimes=self.config.get('n_regimes', 3),
            lookback_window=self.config.get('lookback_window', 252)
        )
        self.walk_forward_optimizer = WalkForwardOptimizer(
            training_window=self.config.get('training_window', 252),
            validation_window=self.config.get('validation_window', 63),
            step_size=self.config.get('step_size', 21),
            regime_detector=self.regime_detector
        )
        self.logger = logging.getLogger(__name__)
    
    def run_comprehensive_backtest(self, 
                                 data: pd.DataFrame,
                                 strategy_class: Any,
                                 param_space: Dict[str, Any],
                                 **kwargs) -> Dict[str, Any]:
        """Run comprehensive backtest with walk-forward analysis and regime detection.
        
        Args:
            data: Market data
            strategy_class: Strategy class to test
            param_space: Parameter space for optimization
            **kwargs: Additional parameters
            
        Returns:
            Comprehensive backtest results
        """
        self.logger.info("Starting comprehensive backtest")
        
        # Run walk-forward analysis
        walk_forward_results = self.walk_forward_optimizer.run_walk_forward_analysis(
            data, strategy_class, param_space, **kwargs
        )
        
        # Analyze results
        analysis = self.walk_forward_optimizer.analyze_results()
        
        # Generate plots
        self.walk_forward_optimizer.plot_results()
        
        return {
            'walk_forward_results': walk_forward_results,
            'analysis': analysis,
            'regimes': self.regime_detector.regime_history,
            'config': self.config
        }
    
    def get_regime_recommendations(self, data: pd.DataFrame) -> List[str]:
        """Get trading recommendations based on current regime.
        
        Args:
            data: Recent market data
            
        Returns:
            List of recommendations
        """
        current_regime = self.regime_detector.get_current_regime(data)
        
        if not current_regime:
            return ["Insufficient data for regime analysis"]
        
        recommendations = []
        regime_name = current_regime.regime_name.lower()
        
        if "high volatility" in regime_name:
            recommendations.extend([
                "Reduce position sizes",
                "Use wider stop losses",
                "Consider hedging strategies",
                "Monitor positions more frequently"
            ])
        elif "low volatility" in regime_name:
            recommendations.extend([
                "Consider increasing position sizes",
                "Use tighter stop losses",
                "Focus on trend-following strategies",
                "Look for breakout opportunities"
            ])
        
        if "bull market" in regime_name:
            recommendations.extend([
                "Favor long positions",
                "Use momentum strategies",
                "Consider trend-following indicators"
            ])
        elif "bear market" in regime_name:
            recommendations.extend([
                "Favor short positions or defensive strategies",
                "Use mean reversion strategies",
                "Consider safe-haven assets"
            ])
        
        return recommendations

# Global instance
_backtest_optimizer = None

def get_backtest_optimizer(config: Optional[Dict[str, Any]] = None) -> BacktestOptimizer:
    """Get global backtest optimizer instance."""
    global _backtest_optimizer
    if _backtest_optimizer is None:
        _backtest_optimizer = BacktestOptimizer(config)
    return _backtest_optimizer 