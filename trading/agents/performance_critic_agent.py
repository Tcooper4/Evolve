"""
Performance Critic Agent

This agent evaluates model performance based on financial metrics
including Sharpe ratio, drawdown, and win rate.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict, field
import pickle

# Local imports
from trading.agents.base_agent_interface import BaseAgent, AgentConfig, AgentResult
from trading.backtesting.backtester import Backtester
from trading.utils.common import calculate_sharpe_ratio, calculate_max_drawdown, calculate_win_rate, timer, handle_exceptions
from trading.strategies.strategy_manager import StrategyManager
from trading.memory.performance_memory import PerformanceMemory
from trading.memory.agent_memory import AgentMemory
from trading.utils.reward_function import RewardFunction

@dataclass
class ModelEvaluationRequest:
    """Request for model evaluation."""
    model_id: str
    model_path: str
    model_type: str
    test_data_path: str
    evaluation_period: int = 252  # Days
    benchmark_symbol: Optional[str] = None
    risk_free_rate: float = 0.02
    request_id: Optional[str] = None

@dataclass
class ModelEvaluationResult:
    """Result of model evaluation."""
    request_id: str
    model_id: str
    evaluation_timestamp: str
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    trading_metrics: Dict[str, Any]
    benchmark_comparison: Optional[Dict[str, float]] = None
    recommendations: List[str] = field(default_factory=list)
    evaluation_status: str = "success"
    error_message: Optional[str] = None

class PerformanceCriticAgent(BaseAgent):
    """Agent responsible for evaluating model performance."""
    
    # Agent metadata
    version = "1.0.0"
    description = "Evaluates model performance based on financial metrics including Sharpe ratio, drawdown, and win rate"
    author = "Evolve Trading System"
    tags = ["performance", "evaluation", "metrics", "risk"]
    capabilities = ["model_evaluation", "performance_analysis", "risk_assessment", "benchmark_comparison"]
    dependencies = ["trading.backtesting", "trading.evaluation", "trading.strategies"]
    
    def _setup(self) -> None:
        """Setup method called during initialization."""
        self.memory = PerformanceMemory()
        self.backtester = Backtester()
        self.strategy_manager = StrategyManager()
        self.agent_memory = AgentMemory("trading/agents/agent_memory.json")
        self.reward_function = RewardFunction()
        
        # Performance thresholds
        self.thresholds = {
            'min_sharpe_ratio': 0.5,
            'max_drawdown': -0.15,
            'min_win_rate': 0.45,
            'max_volatility': 0.25,
            'min_calmar_ratio': 0.5
        }
        
        # Evaluation history
        self.evaluation_history: Dict[str, List[ModelEvaluationResult]] = {}
        
        self.logger.info("PerformanceCriticAgent initialized")
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    async def execute(self, **kwargs) -> AgentResult:
        """Execute the model evaluation logic.
        
        Args:
            **kwargs: Must contain 'request' with ModelEvaluationRequest
            
        Returns:
            AgentResult: Result of the model evaluation execution
        """
        request = kwargs.get('request')
        if not request:
            return AgentResult(
                success=False,
                error_message="ModelEvaluationRequest is required"
            )
        
        if not isinstance(request, ModelEvaluationRequest):
            return AgentResult(
                success=False,
                error_message="Request must be a ModelEvaluationRequest instance"
            )
        
        try:
            result = self.evaluate_model(request)
            
            if result.evaluation_status == "success":
                return AgentResult(
                    success=True,
                    data={
                        "model_id": result.model_id,
                        "performance_metrics": result.performance_metrics,
                        "risk_metrics": result.risk_metrics,
                        "trading_metrics": result.trading_metrics,
                        "recommendations": result.recommendations,
                        "evaluation_timestamp": result.evaluation_timestamp
                    }
                )
            else:
                return AgentResult(
                    success=False,
                    error_message=result.error_message or "Model evaluation failed"
                )
                
        except Exception as e:
            return AgentResult(
                success=False,
                error_message=str(e)
            )
    
    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            bool: True if input is valid
        """
        request = kwargs.get('request')
        if not request:
            return False
        
        if not isinstance(request, ModelEvaluationRequest):
            return False
        
        # Validate required fields
        if not request.model_id or not request.model_path or not request.model_type or not request.test_data_path:
            return False
        
        # Validate model path exists
        if not Path(request.model_path).exists():
            return False
        
        # Validate test data path exists
        if not Path(request.test_data_path).exists():
            return False
        
        # Validate evaluation period
        if request.evaluation_period <= 0:
            return False
        
        return True
    
    @handle_exceptions
    def evaluate_model(self, request: ModelEvaluationRequest) -> ModelEvaluationResult:
        """Evaluate a model's performance.
        
        Args:
            request: Model evaluation request
            
        Returns:
            Model evaluation result
        """
        request_id = request.request_id or str(uuid.uuid4())
        self.logger.info(f"Evaluating model {request.model_id} with ID: {request_id}")
        
        try:
            # Load model and test data
            model = self._load_model(request.model_path, request.model_type)
            test_data = self._load_test_data(request.test_data_path, request.evaluation_period)
            
            # Generate predictions
            predictions = self._generate_predictions(model, test_data, request.model_type)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(predictions, test_data)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(predictions, test_data)
            
            # Calculate trading metrics
            trading_metrics = self._calculate_trading_metrics(predictions, test_data)
            
            # Compare with benchmark if provided
            benchmark_comparison = None
            if request.benchmark_symbol:
                benchmark_comparison = self._compare_with_benchmark(
                    predictions, test_data, request.benchmark_symbol
                )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                performance_metrics, risk_metrics, trading_metrics
            )
            
            result = ModelEvaluationResult(
                request_id=request_id,
                model_id=request.model_id,
                evaluation_timestamp=datetime.now().isoformat(),
                performance_metrics=performance_metrics,
                risk_metrics=risk_metrics,
                trading_metrics=trading_metrics,
                benchmark_comparison=benchmark_comparison,
                recommendations=recommendations
            )
            
            # Store evaluation result
            self._store_evaluation_result(result)
            
            # Compute reward score for the model (use both performance and trading metrics)
            reward = self.reward_function.compute({
                **result.performance_metrics,
                **result.trading_metrics,
                'max_drawdown': result.risk_metrics.get('max_drawdown', 0.0)
            })
            
            # Log outcome to agent memory (add reward)
            self.agent_memory.log_outcome(
                agent="PerformanceCriticAgent",
                run_type="evaluate",
                outcome={
                    "model_id": result.model_id,
                    "performance_metrics": result.performance_metrics,
                    "risk_metrics": result.risk_metrics,
                    "trading_metrics": result.trading_metrics,
                    "recommendations": result.recommendations,
                    "status": result.evaluation_status,
                    "reward": reward,
                    "timestamp": result.evaluation_timestamp
                }
            )
            
            self.logger.info(f"Successfully evaluated model {request.model_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate model {request.model_id}: {str(e)}")
            self.agent_memory.log_outcome(
                agent="PerformanceCriticAgent",
                run_type="evaluate",
                outcome={
                    "model_id": request.model_id,
                    "status": "failed",
                    "error_message": str(e),
                    "reward": 0.0,
                    "timestamp": datetime.now().isoformat()
                }
            )
            return ModelEvaluationResult(
                request_id=request_id,
                model_id=request.model_id,
                evaluation_timestamp=datetime.now().isoformat(),
                performance_metrics={},
                risk_metrics={},
                trading_metrics={},
                recommendations=[],
                evaluation_status="failed",
                error_message=str(e)
            )
    
    def _load_model(self, model_path: str, model_type: str) -> Any:
        """Load a trained model.
        
        Args:
            model_path: Path to model file
            model_type: Type of model
            
        Returns:
            Loaded model
        """
        if model_type == 'ensemble':
            # Load ensemble configuration
            with open(model_path, 'r') as f:
                config = json.load(f)
            
            # Load individual models
            models = []
            for model_info in config['models']:
                with open(model_info['path'], 'rb') as f:
                    models.append(pickle.load(f))
            
            return {'models': models, 'config': config}
        else:
            # Load single model
            with open(model_path, 'rb') as f:
                return pickle.load(f)
    
    def _load_test_data(self, data_path: str, evaluation_period: int) -> pd.DataFrame:
        """Load test data for evaluation.
        
        Args:
            data_path: Path to test data
            evaluation_period: Number of days to evaluate
            
        Returns:
            Test data
        """
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
        
        # Ensure datetime index
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
        
        # Take last evaluation_period days
        if len(data) > evaluation_period:
            data = data.tail(evaluation_period)
        
        return data
    
    def _generate_predictions(self, model: Any, test_data: pd.DataFrame, model_type: str) -> pd.Series:
        """Generate predictions using the model.
        
        Args:
            model: Trained model
            test_data: Test data
            model_type: Type of model
            
        Returns:
            Predictions
        """
        if model_type == 'ensemble':
            # Generate predictions from all models
            predictions = []
            for i, sub_model in enumerate(model['models']):
                pred = sub_model.predict(test_data)
                predictions.append(pred * model['config']['models'][i]['weight'])
            
            # Combine predictions
            return pd.Series(np.sum(predictions, axis=0), index=test_data.index)
        else:
            # Generate predictions from single model
            return pd.Series(model.predict(test_data), index=test_data.index)
    
    def _calculate_performance_metrics(self, predictions: pd.Series, test_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Args:
            predictions: Model predictions
            test_data: Test data
            
        Returns:
            Performance metrics
        """
        # Calculate returns
        actual_returns = test_data['close'].pct_change().dropna()
        predicted_returns = predictions.pct_change().dropna()
        
        # Align data
        common_index = actual_returns.index.intersection(predicted_returns.index)
        actual_returns = actual_returns.loc[common_index]
        predicted_returns = predicted_returns.loc[common_index]
        
        # Calculate metrics
        sharpe_ratio = calculate_sharpe_ratio(actual_returns)
        total_return = (1 + actual_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(actual_returns)) - 1
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': actual_returns.std() * np.sqrt(252),
            'information_ratio': (actual_returns.mean() - predicted_returns.mean()) / actual_returns.std()
        }
    
    def _calculate_risk_metrics(self, predictions: pd.Series, test_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk metrics.
        
        Args:
            predictions: Model predictions
            test_data: Test data
            
        Returns:
            Risk metrics
        """
        # Calculate returns
        actual_returns = test_data['close'].pct_change().dropna()
        
        # Calculate risk metrics
        max_drawdown = calculate_max_drawdown(actual_returns)
        var_95 = np.percentile(actual_returns, 5)
        cvar_95 = actual_returns[actual_returns <= var_95].mean()
        
        # Calculate downside deviation
        downside_returns = actual_returns[actual_returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        
        # Calculate Sortino ratio
        risk_free_rate = self.config.get('risk_free_rate', 0.02) / 252
        sortino_ratio = (actual_returns.mean() - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'downside_deviation': downside_deviation,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': self._calculate_calmar_ratio(actual_returns, max_drawdown)
        }
    
    def _calculate_trading_metrics(self, predictions: pd.Series, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trading-specific metrics.
        
        Args:
            predictions: Model predictions
            test_data: Test data
            
        Returns:
            Trading metrics
        """
        # Generate trading signals
        signals = self._generate_trading_signals(predictions, test_data)
        
        # Calculate trading metrics
        win_rate = calculate_win_rate(signals)
        profit_factor = self._calculate_profit_factor(signals)
        avg_trade = self._calculate_avg_trade(signals)
        
        # Calculate trade statistics
        trades = self._extract_trades(signals)
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'total_trades': len(trades),
            'avg_win': self._calculate_avg_win(trades),
            'avg_loss': self._calculate_avg_loss(trades),
            'largest_win': max(trades) if trades else 0,
            'largest_loss': min(trades) if trades else 0
        }
    
    def _generate_trading_signals(self, predictions: pd.Series, test_data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from predictions.
        
        Args:
            predictions: Model predictions
            test_data: Test data
            
        Returns:
            Trading signals
        """
        # Simple signal generation based on prediction direction
        signals = pd.Series(0, index=predictions.index)
        
        # Buy signal when prediction increases
        signals[predictions.diff() > 0] = 1
        
        # Sell signal when prediction decreases
        signals[predictions.diff() < 0] = -1
        
        return signals
    
    def _calculate_profit_factor(self, signals: pd.Series) -> float:
        """Calculate profit factor.
        
        Args:
            signals: Trading signals
            
        Returns:
            Profit factor
        """
        # This is a simplified calculation
        # In practice, you'd calculate actual profits/losses from trades
        return 1.5  # Placeholder, actual calculation needed
    
    def _calculate_avg_trade(self, signals: pd.Series) -> float:
        """Calculate average trade.
        
        Args:
            signals: Trading signals
            
        Returns:
            Average trade
        """
        # This is a simplified calculation
        return 0.02  # Placeholder, actual calculation needed
    
    def _extract_trades(self, signals: pd.Series) -> List[float]:
        """Extract individual trades from signals.
        
        Args:
            signals: Trading signals
            
        Returns:
            List of trade returns
        """
        # This is a simplified implementation
        # In practice, you'd track actual trade entries/exits
        return [0.01, -0.005, 0.02, -0.01, 0.015]  # Placeholder, actual implementation needed
    
    def _calculate_avg_win(self, trades: List[float]) -> float:
        """Calculate average winning trade.
        
        Args:
            trades: List of trade returns
            
        Returns:
            Average winning trade
        """
        winning_trades = [t for t in trades if t > 0]
        return np.mean(winning_trades) if winning_trades else 0
    
    def _calculate_avg_loss(self, trades: List[float]) -> float:
        """Calculate average losing trade.
        
        Args:
            trades: List of trade returns
            
        Returns:
            Average losing trade
        """
        losing_trades = [t for t in trades if t < 0]
        return np.mean(losing_trades) if losing_trades else 0
    
    def _calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """Calculate Calmar ratio.
        
        Args:
            returns: Returns series
            max_drawdown: Maximum drawdown
            
        Returns:
            Calmar ratio
        """
        annualized_return = (1 + returns.mean()) ** 252 - 1
        return annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    def _compare_with_benchmark(self, predictions: pd.Series, test_data: pd.DataFrame, benchmark_symbol: str) -> Dict[str, float]:
        """Compare model performance with benchmark.
        
        Args:
            predictions: Model predictions
            test_data: Test data
            benchmark_symbol: Benchmark symbol
            
        Returns:
            Benchmark comparison metrics
        """
        # This is a simplified implementation
        # In practice, you'd fetch actual benchmark data
        return {
            'benchmark_return': 0.08,
            'benchmark_sharpe': 0.6,
            'benchmark_drawdown': -0.10,
            'excess_return': 0.02,
            'information_ratio': 0.3
        }
    
    def _generate_recommendations(self, performance_metrics: Dict[str, float], 
                                risk_metrics: Dict[str, float], 
                                trading_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results.
        
        Args:
            performance_metrics: Performance metrics
            risk_metrics: Risk metrics
            trading_metrics: Trading metrics
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Performance recommendations
        if performance_metrics.get('sharpe_ratio', 0) < self.thresholds['min_sharpe_ratio']:
            recommendations.append("Sharpe ratio below threshold - consider model retraining or hyperparameter tuning")
        
        if performance_metrics.get('total_return', 0) < 0:
            recommendations.append("Negative total return - model may need fundamental changes")
        
        # Risk recommendations
        if risk_metrics.get('max_drawdown', 0) < self.thresholds['max_drawdown']:
            recommendations.append("Maximum drawdown exceeds threshold - implement better risk management")
        
        if risk_metrics.get('sortino_ratio', 0) < 0.5:
            recommendations.append("Low Sortino ratio - focus on reducing downside risk")
        
        # Trading recommendations
        if trading_metrics.get('win_rate', 0) < self.thresholds['min_win_rate']:
            recommendations.append("Win rate below threshold - review signal generation logic")
        
        if trading_metrics.get('profit_factor', 0) < 1.2:
            recommendations.append("Low profit factor - optimize entry/exit strategies")
        
        # Positive feedback
        if performance_metrics.get('sharpe_ratio', 0) > 1.0:
            recommendations.append("Excellent Sharpe ratio - consider increasing position sizes")
        
        if risk_metrics.get('calmar_ratio', 0) > 1.0:
            recommendations.append("Strong Calmar ratio - model shows good risk-adjusted returns")
        
        return recommendations
    
    def _store_evaluation_result(self, result: ModelEvaluationResult) -> None:
        """Store evaluation result in memory.
        
        Args:
            result: Evaluation result
        """
        # Store in local history
        if result.model_id not in self.evaluation_history:
            self.evaluation_history[result.model_id] = []
        
        self.evaluation_history[result.model_id].append(result)
        
        # Store in performance memory
        metadata = {
            'model_id': result.model_id,
            'evaluation_timestamp': result.evaluation_timestamp,
            'performance_metrics': result.performance_metrics,
            'risk_metrics': result.risk_metrics,
            'trading_metrics': result.trading_metrics,
            'recommendations': result.recommendations,
            'status': result.evaluation_status
        }
        
        self.memory.store_evaluation_result(result.model_id, metadata)
    
    def get_evaluation_history(self, model_id: str) -> List[ModelEvaluationResult]:
        """Get evaluation history for a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            List of evaluation results
        """
        return self.evaluation_history.get(model_id, [])
    
    def get_model_performance_summary(self, model_id: str) -> Dict[str, Any]:
        """Get performance summary for a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Performance summary
        """
        evaluations = self.get_evaluation_history(model_id)
        if not evaluations:
            return {}
        
        # Get latest evaluation
        latest = evaluations[-1]
        
        # Calculate trends
        if len(evaluations) > 1:
            previous = evaluations[-2]
            sharpe_trend = latest.performance_metrics.get('sharpe_ratio', 0) - previous.performance_metrics.get('sharpe_ratio', 0)
            return_trend = latest.performance_metrics.get('total_return', 0) - previous.performance_metrics.get('total_return', 0)
        else:
            sharpe_trend = 0
            return_trend = 0
        
        return {
            'model_id': model_id,
            'latest_evaluation': latest.evaluation_timestamp,
            'current_sharpe': latest.performance_metrics.get('sharpe_ratio', 0),
            'current_return': latest.performance_metrics.get('total_return', 0),
            'current_drawdown': latest.risk_metrics.get('max_drawdown', 0),
            'current_win_rate': latest.trading_metrics.get('win_rate', 0),
            'sharpe_trend': sharpe_trend,
            'return_trend': return_trend,
            'recommendations': latest.recommendations,
            'evaluation_count': len(evaluations)
        } 