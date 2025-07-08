# -*- coding: utf-8 -*-
"""
Self-Tuning Optimizer Agent for dynamic parameter adjustment and optimization.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from trading.optimization.bayesian_optimizer import BayesianOptimizer
from trading.optimization.core_optimizer import GeneticOptimizer
from trading.market.market_analyzer import MarketAnalyzer
from trading.utils.performance_metrics import calculate_sharpe_ratio, calculate_max_drawdown
from trading.memory.agent_memory import AgentMemory
from .base_agent_interface import BaseAgent, AgentConfig, AgentResult

class OptimizationTrigger(str, Enum):
    """Optimization trigger types."""
    PERFORMANCE_DECLINE = "performance_decline"
    MARKET_REGIME_CHANGE = "market_regime_change"
    SCHEDULED = "scheduled"
    VOLATILITY_SPIKE = "volatility_spike"
    MANUAL = "manual"

@dataclass
class OptimizationResult:
    """Optimization result."""
    strategy_name: str
    timestamp: datetime
    trigger: OptimizationTrigger
    old_parameters: Dict[str, Any]
    new_parameters: Dict[str, Any]
    performance_improvement: float
    optimization_time: float
    confidence: float

@dataclass
class ParameterConstraint:
    """Parameter constraint for optimization."""
    parameter_name: str
    min_value: float
    max_value: float
    step_size: Optional[float] = None
    parameter_type: str = "continuous"  # continuous, discrete, categorical

class SelfTuningOptimizerAgent(BaseAgent):
    """
    Self-Tuning Optimizer Agent with:
    - Dynamic parameter adjustment based on market conditions
    - Performance-based optimization triggers
    - Multi-objective optimization
    - Constraint handling
    - Optimization history tracking
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="SelfTuningOptimizerAgent",
                enabled=True,
                priority=1,
                max_concurrent_runs=1,
                timeout_seconds=300,
                retry_attempts=3,
                custom_config={}
            )
        super().__init__(config)
        self.config_dict = config.custom_config or {}
        self.logger = logging.getLogger(__name__)
        self.memory = AgentMemory()
        self.market_analyzer = MarketAnalyzer()
        self.bayesian_optimizer = BayesianOptimizer()
        self.genetic_optimizer = GeneticOptimizer()
        
        # Configuration
        self.optimization_frequency = self.config_dict.get('optimization_frequency', 'weekly')
        self.performance_threshold = self.config_dict.get('performance_threshold', 0.1)
        self.volatility_threshold = self.config_dict.get('volatility_threshold', 0.03)
        self.optimization_timeout = self.config_dict.get('optimization_timeout', 300)  # 5 minutes
        self.min_improvement = self.config_dict.get('min_improvement', 0.05)
        
        # Storage
        self.optimization_history: List[OptimizationResult] = []
        self.parameter_constraints: Dict[str, List[ParameterConstraint]] = {}
        self.strategy_performance: Dict[str, pd.Series] = {}
        self.market_regime_history: List[Dict[str, Any]] = []
        
        # Load existing data
        self._load_optimization_data()

    def _setup(self):
        pass

    async def execute(self, **kwargs) -> AgentResult:
        """Execute the self-tuning optimization logic.
        Args:
            **kwargs: strategy_name, strategy_performance, market_data, action, etc.
        Returns:
            AgentResult
        """
        try:
            action = kwargs.get('action', 'check_triggers')
            
            if action == 'check_triggers':
                strategy_name = kwargs.get('strategy_name')
                strategy_performance = kwargs.get('strategy_performance')
                market_data = kwargs.get('market_data')
                
                if strategy_name is None or strategy_performance is None or market_data is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameters: strategy_name, strategy_performance, market_data"
                    )
                
                triggers = await self.check_optimization_triggers(strategy_name, strategy_performance, market_data)
                return AgentResult(success=True, data={
                    "triggers": [trigger.value for trigger in triggers],
                    "trigger_count": len(triggers)
                })
                
            elif action == 'optimize_parameters':
                strategy_name = kwargs.get('strategy_name')
                current_parameters = kwargs.get('current_parameters')
                strategy_performance = kwargs.get('strategy_performance')
                market_data = kwargs.get('market_data')
                triggers = kwargs.get('triggers', [])
                
                if strategy_name is None or current_parameters is None or strategy_performance is None or market_data is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameters: strategy_name, current_parameters, strategy_performance, market_data"
                    )
                
                # Convert trigger strings to enum if needed
                if triggers and isinstance(triggers[0], str):
                    triggers = [OptimizationTrigger(t) for t in triggers]
                
                result = await self.optimize_strategy_parameters(
                    strategy_name, current_parameters, strategy_performance, market_data, triggers
                )
                return AgentResult(success=True, data={
                    "optimization_result": result.__dict__,
                    "performance_improvement": result.performance_improvement,
                    "confidence": result.confidence
                })
                
            elif action == 'get_optimization_summary':
                strategy_name = kwargs.get('strategy_name')
                summary = self.get_optimization_summary(strategy_name)
                return AgentResult(success=True, data={"optimization_summary": summary})
                
            elif action == 'set_constraints':
                strategy_name = kwargs.get('strategy_name')
                constraints_data = kwargs.get('constraints', [])
                
                if strategy_name is None or not constraints_data:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameters: strategy_name, constraints"
                    )
                
                constraints = [ParameterConstraint(**c) for c in constraints_data]
                self.set_parameter_constraints(strategy_name, constraints)
                return AgentResult(success=True, data={
                    "message": f"Set {len(constraints)} constraints for {strategy_name}"
                })
                
            else:
                return AgentResult(success=False, error_message=f"Unknown action: {action}")
                
        except Exception as e:
            return self.handle_error(e)
    
    async def check_optimization_triggers(self, 
                                        strategy_name: str,
                                        strategy_performance: pd.Series,
                                        market_data: pd.DataFrame) -> List[OptimizationTrigger]:
        """
        Check if optimization should be triggered.
        
        Args:
            strategy_name: Name of the strategy
            strategy_performance: Strategy performance series
            market_data: Current market data
            
        Returns:
            List of triggered optimization types
        """
        try:
            triggers = []
            
            # Check performance decline
            if self._check_performance_decline(strategy_performance):
                triggers.append(OptimizationTrigger.PERFORMANCE_DECLINE)
            
            # Check market regime change
            if self._check_market_regime_change(market_data):
                triggers.append(OptimizationTrigger.MARKET_REGIME_CHANGE)
            
            # Check volatility spike
            if self._check_volatility_spike(market_data):
                triggers.append(OptimizationTrigger.VOLATILITY_SPIKE)
            
            # Check scheduled optimization
            if self._check_scheduled_optimization(strategy_name):
                triggers.append(OptimizationTrigger.SCHEDULED)
            
            return triggers
            
        except Exception as e:
            self.logger.error(f"Error checking optimization triggers: {str(e)}")
            return []
    
    def _check_performance_decline(self, performance: pd.Series) -> bool:
        """Check if strategy performance has declined significantly."""
        try:
            if len(performance) < 20:
                return False
            
            # Calculate recent vs historical performance
            recent_performance = performance.tail(10).mean()
            historical_performance = performance.tail(50).mean()
            
            # Check for significant decline
            if historical_performance != 0:
                decline_ratio = (historical_performance - recent_performance) / abs(historical_performance)
                return decline_ratio > self.performance_threshold
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking performance decline: {str(e)}")
            return False
    
    def _check_market_regime_change(self, market_data: pd.DataFrame) -> bool:
        """Check if market regime has changed."""
        try:
            if market_data.empty or 'close' not in market_data.columns:
                return False
            
            # Get current market regime
            current_regime = self._detect_market_regime(market_data)
            
            # Check against recent regime history
            if self.market_regime_history:
                recent_regimes = [r['regime'] for r in self.market_regime_history[-5:]]
                if current_regime not in recent_regimes:
                    return True
            
            # Store current regime
            self.market_regime_history.append({
                'timestamp': datetime.now(),
                'regime': current_regime
            })
            
            # Keep only recent history
            cutoff_date = datetime.now() - timedelta(days=30)
            self.market_regime_history = [
                r for r in self.market_regime_history
                if r['timestamp'] > cutoff_date
            ]
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking market regime change: {str(e)}")
            return False
    
    def _detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """Detect current market regime."""
        try:
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std().iloc[-1]
            
            sma_short = market_data['close'].rolling(window=10).mean()
            sma_long = market_data['close'].rolling(window=50).mean()
            trend = (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
            
            if volatility > 0.03:
                return 'high_volatility'
            elif volatility < 0.01:
                return 'low_volatility'
            elif trend > 0.02:
                return 'trending_up'
            elif trend < -0.02:
                return 'trending_down'
            else:
                return 'sideways'
                
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return 'unknown'
    
    def _check_volatility_spike(self, market_data: pd.DataFrame) -> bool:
        """Check for volatility spike."""
        try:
            if market_data.empty or 'close' not in market_data.columns:
                return False
            
            returns = market_data['close'].pct_change().dropna()
            current_volatility = returns.tail(10).std()
            historical_volatility = returns.tail(50).std()
            
            # Check if current volatility is significantly higher
            if historical_volatility > 0:
                volatility_ratio = current_volatility / historical_volatility
                return volatility_ratio > 2.0  # 2x historical volatility
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking volatility spike: {str(e)}")
            return False
    
    def _check_scheduled_optimization(self, strategy_name: str) -> bool:
        """Check if scheduled optimization is due."""
        try:
            # Get last optimization time for this strategy
            last_optimization = None
            for result in self.optimization_history:
                if result.strategy_name == strategy_name:
                    if last_optimization is None or result.timestamp > last_optimization:
                        last_optimization = result.timestamp
            
            if last_optimization is None:
                return True  # Never optimized
            
            # Check if enough time has passed
            time_since_optimization = datetime.now() - last_optimization
            
            if self.optimization_frequency == 'daily':
                return time_since_optimization.days >= 1
            elif self.optimization_frequency == 'weekly':
                return time_since_optimization.days >= 7
            elif self.optimization_frequency == 'monthly':
                return time_since_optimization.days >= 30
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking scheduled optimization: {str(e)}")
            return False
    
    async def optimize_strategy_parameters(self, 
                                         strategy_name: str,
                                         current_parameters: Dict[str, Any],
                                         strategy_performance: pd.Series,
                                         market_data: pd.DataFrame,
                                         triggers: List[OptimizationTrigger]) -> OptimizationResult:
        """
        Optimize strategy parameters based on triggers and market conditions.
        
        Args:
            strategy_name: Name of the strategy to optimize
            current_parameters: Current strategy parameters
            strategy_performance: Strategy performance history
            market_data: Current market data
            triggers: List of optimization triggers
            
        Returns:
            Optimization result with new parameters
        """
        try:
            self.logger.info(f"Optimizing parameters for {strategy_name}")
            
            start_time = datetime.now()
            
            # Get parameter constraints
            constraints = self._get_parameter_constraints(strategy_name)
            
            # Define optimization objective
            objective = self._create_optimization_objective(
                strategy_name, strategy_performance, market_data, triggers
            )
            
            # Choose optimization method
            optimization_method = self._choose_optimization_method(triggers, constraints)
            
            # Run optimization
            if optimization_method == 'bayesian':
                new_parameters = await self._run_bayesian_optimization(
                    objective, current_parameters, constraints
                )
            else:
                new_parameters = await self._run_genetic_optimization(
                    objective, current_parameters, constraints
                )
            
            # Evaluate improvement
            performance_improvement = self._evaluate_improvement(
                strategy_name, current_parameters, new_parameters, strategy_performance
            )
            
            # Calculate confidence
            confidence = self._calculate_optimization_confidence(
                performance_improvement, triggers, market_data
            )
            
            # Create optimization result
            result = OptimizationResult(
                strategy_name=strategy_name,
                timestamp=datetime.now(),
                trigger=triggers[0] if triggers else OptimizationTrigger.MANUAL,
                old_parameters=current_parameters,
                new_parameters=new_parameters,
                performance_improvement=performance_improvement,
                optimization_time=(datetime.now() - start_time).total_seconds(),
                confidence=confidence
            )
            
            # Store result
            self.optimization_history.append(result)
            
            # Store in memory
            self._store_optimization_result(result)
            
            self.logger.info(f"Optimization completed for {strategy_name}: "
                           f"improvement = {performance_improvement:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error optimizing strategy parameters: {str(e)}")
            return False
    
    def _get_parameter_constraints(self, strategy_name: str) -> List[ParameterConstraint]:
        """Get parameter constraints for a strategy."""
        try:
            if strategy_name in self.parameter_constraints:
                return self.parameter_constraints[strategy_name]
            
            # Default constraints based on strategy type
            if 'rsi' in strategy_name.lower():
                return [
                    ParameterConstraint('period', 5, 30, 1, 'discrete'),
                    ParameterConstraint('overbought', 60, 90, 5, 'discrete'),
                    ParameterConstraint('oversold', 10, 40, 5, 'discrete')
                ]
            elif 'macd' in strategy_name.lower():
                return [
                    ParameterConstraint('fast_period', 8, 20, 1, 'discrete'),
                    ParameterConstraint('slow_period', 20, 40, 1, 'discrete'),
                    ParameterConstraint('signal_period', 5, 15, 1, 'discrete')
                ]
            elif 'bollinger' in strategy_name.lower():
                return [
                    ParameterConstraint('period', 10, 30, 1, 'discrete'),
                    ParameterConstraint('std_dev', 1.5, 3.0, 0.1, 'continuous')
                ]
            else:
                # Generic constraints
                return [
                    ParameterConstraint('period', 10, 50, 1, 'discrete'),
                    ParameterConstraint('threshold', 0.01, 0.1, 0.01, 'continuous')
                ]
                
        except Exception as e:
            self.logger.error(f"Error getting parameter constraints: {str(e)}")
            return []
    
    def _create_optimization_objective(self, 
                                     strategy_name: str,
                                     strategy_performance: pd.Series,
                                     market_data: pd.DataFrame,
                                     triggers: List[OptimizationTrigger]) -> callable:
        """Create optimization objective function."""
        try:
            def objective(parameters):
                try:
                    # Simulate strategy performance with new parameters
                    simulated_performance = self._simulate_strategy_performance(
                        strategy_name, parameters, market_data
                    )
                    
                    if simulated_performance is None or len(simulated_performance) < 10:
                        return 0.0
                    
                    # Calculate multiple objectives
                    sharpe_ratio = calculate_sharpe_ratio(simulated_performance)
                    max_drawdown = calculate_max_drawdown(simulated_performance)
                    total_return = simulated_performance.sum()
                    
                    # Weight objectives based on triggers
                    if OptimizationTrigger.PERFORMANCE_DECLINE in triggers:
                        # Focus on improving returns
                        score = 0.5 * total_return + 0.3 * sharpe_ratio - 0.2 * max_drawdown
                    elif OptimizationTrigger.VOLATILITY_SPIKE in triggers:
                        # Focus on risk management
                        score = 0.3 * total_return + 0.5 * sharpe_ratio - 0.2 * max_drawdown
                    else:
                        # Balanced approach
                        score = 0.4 * total_return + 0.4 * sharpe_ratio - 0.2 * max_drawdown
                    
                    return score
                    
                except Exception as e:
                    self.logger.error(f"Error in optimization objective: {str(e)}")
                    return 0.0
            
            return objective
            
        except Exception as e:
            self.logger.error(f"Error creating optimization objective: {str(e)}")
            return lambda x: 0.0
    
    def _simulate_strategy_performance(self, 
                                     strategy_name: str,
                                     parameters: Dict[str, Any],
                                     market_data: pd.DataFrame) -> Optional[pd.Series]:
        """Simulate strategy performance with given parameters."""
        try:
            # This is a simplified simulation
            # In practice, you'd run the actual strategy with the parameters
            
            if market_data.empty or 'close' not in market_data.columns:
                return None
            
            # Generate simple simulated returns based on parameters
            returns = market_data['close'].pct_change().dropna()
            
            # Apply parameter-based adjustments
            period = parameters.get('period', 20)
            threshold = parameters.get('threshold', 0.02)
            
            # Simple moving average strategy simulation
            sma = market_data['close'].rolling(window=period).mean()
            signals = (market_data['close'] > sma).astype(int)
            signals = signals.shift(1).fillna(0)
            
            # Generate strategy returns
            strategy_returns = signals * returns
            
            return strategy_returns.dropna()
            
        except Exception as e:
            self.logger.error(f"Error simulating strategy performance: {str(e)}")
            return None
    
    def _choose_optimization_method(self, 
                                  triggers: List[OptimizationTrigger],
                                  constraints: List[ParameterConstraint]) -> str:
        """Choose optimization method based on triggers and constraints."""
        try:
            # Use genetic optimization for complex constraints or multiple triggers
            if len(triggers) > 1 or len(constraints) > 5:
                return 'genetic'
            
            # Use Bayesian optimization for continuous parameters
            continuous_params = [c for c in constraints if c.parameter_type == 'continuous']
            if len(continuous_params) > len(constraints) / 2:
                return 'bayesian'
            
            # Default to genetic optimization
            return 'genetic'
            
        except Exception as e:
            self.logger.error(f"Error choosing optimization method: {str(e)}")
            return 'genetic'
    
    async def _run_bayesian_optimization(self, 
                                       objective: callable,
                                       current_parameters: Dict[str, Any],
                                       constraints: List[ParameterConstraint]) -> Dict[str, Any]:
        """Run Bayesian optimization."""
        try:
            # Define parameter space
            param_space = {}
            for constraint in constraints:
                if constraint.parameter_type == 'continuous':
                    param_space[constraint.parameter_name] = [constraint.min_value, constraint.max_value]
                else:
                    # Discrete parameters
                    values = np.arange(constraint.min_value, constraint.max_value + 1, 
                                     constraint.step_size or 1)
                    param_space[constraint.parameter_name] = values.tolist()
            
            # Run optimization
            with ThreadPoolExecutor() as executor:
                best_params = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    self.bayesian_optimizer.optimize,
                    objective,
                    param_space,
                    n_trials=20
                )
            
            return best_params or current_parameters
            
        except Exception as e:
            self.logger.error(f"Error in Bayesian optimization: {str(e)}")
            return current_parameters
    
    async def _run_genetic_optimization(self, 
                                      objective: callable,
                                      current_parameters: Dict[str, Any],
                                      constraints: List[ParameterConstraint]) -> Dict[str, Any]:
        """Run genetic optimization."""
        try:
            # Define parameter space
            param_space = {}
            for constraint in constraints:
                if constraint.parameter_type == 'continuous':
                    param_space[constraint.parameter_name] = [constraint.min_value, constraint.max_value]
                else:
                    # Discrete parameters
                    values = np.arange(constraint.min_value, constraint.max_value + 1, 
                                     constraint.step_size or 1)
                    param_space[constraint.parameter_name] = values.tolist()
            
            # Run optimization
            with ThreadPoolExecutor() as executor:
                best_params = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    self.genetic_optimizer.optimize,
                    objective,
                    param_space,
                    population_size=30,
                    generations=15
                )
            
            return best_params or current_parameters
            
        except Exception as e:
            self.logger.error(f"Error in genetic optimization: {str(e)}")
            return current_parameters
    
    def _evaluate_improvement(self, 
                            strategy_name: str,
                            old_parameters: Dict[str, Any],
                            new_parameters: Dict[str, Any],
                            strategy_performance: pd.Series) -> float:
        """Evaluate performance improvement from parameter change."""
        try:
            # Simulate performance with old and new parameters
            # This is a simplified evaluation
            old_performance = self._simulate_strategy_performance(
                strategy_name, old_parameters, pd.DataFrame()  # Would need market data
            )
            new_performance = self._simulate_strategy_performance(
                strategy_name, new_parameters, pd.DataFrame()  # Would need market data
            )
            
            if old_performance is not None and new_performance is not None:
                old_sharpe = calculate_sharpe_ratio(old_performance)
                new_sharpe = calculate_sharpe_ratio(new_performance)
                
                if old_sharpe != 0:
                    improvement = (new_sharpe - old_sharpe) / abs(old_sharpe)
                    return improvement
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error evaluating improvement: {str(e)}")
            return 0.0
    
    def _calculate_optimization_confidence(self, 
                                         performance_improvement: float,
                                         triggers: List[OptimizationTrigger],
                                         market_data: pd.DataFrame) -> float:
        """Calculate confidence in optimization result."""
        try:
            confidence = 0.5  # Base confidence
            
            # Performance improvement contribution
            if performance_improvement > 0:
                confidence += min(0.3, performance_improvement)
            
            # Trigger contribution
            if OptimizationTrigger.PERFORMANCE_DECLINE in triggers:
                confidence += 0.1
            if OptimizationTrigger.MARKET_REGIME_CHANGE in triggers:
                confidence += 0.1
            
            # Market data quality contribution
            if not market_data.empty and len(market_data) > 100:
                confidence += 0.1
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating optimization confidence: {str(e)}")
            return 0.5
    
    def _store_optimization_result(self, result: OptimizationResult):
        """Store optimization result in memory."""
        try:
            self.memory.store('optimization_results', {
                'result': result.__dict__,
                'timestamp': datetime.now()
            })
        except Exception as e:
            self.logger.error(f"Error storing optimization result: {str(e)}")
    
    def _load_optimization_data(self):
        """Load optimization data from memory."""
        try:
            # Load optimization history
            history_data = self.memory.get('optimization_history')
            if history_data:
                self.optimization_history = [OptimizationResult(**r) for r in history_data.get('results', [])]
            
            # Load parameter constraints
            constraints_data = self.memory.get('parameter_constraints')
            if constraints_data:
                for strategy_name, constraints in constraints_data.items():
                    self.parameter_constraints[strategy_name] = [
                        ParameterConstraint(**c) for c in constraints
                    ]
                    
        except Exception as e:
            self.logger.error(f"Error loading optimization data: {str(e)}")
    
    def get_optimization_summary(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of optimization activities."""
        try:
            if strategy_name:
                filtered_results = [r for r in self.optimization_history if r.strategy_name == strategy_name]
            else:
                filtered_results = self.optimization_history
            
            if not filtered_results:
                return {}
            
            # Calculate summary statistics
            improvements = [r.performance_improvement for r in filtered_results]
            confidences = [r.confidence for r in filtered_results]
            optimization_times = [r.optimization_time for r in filtered_results]
            
            return {
                'total_optimizations': len(filtered_results),
                'avg_improvement': np.mean(improvements),
                'avg_confidence': np.mean(confidences),
                'avg_optimization_time': np.mean(optimization_times),
                'recent_optimizations': [
                    {
                        'strategy': r.strategy_name,
                        'trigger': r.trigger.value,
                        'improvement': r.performance_improvement,
                        'confidence': r.confidence
                    }
                    for r in filtered_results[-5:]  # Last 5 optimizations
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting optimization summary: {str(e)}")
            return {}
    
    def set_parameter_constraints(self, 
                                strategy_name: str,
                                constraints: List[ParameterConstraint]):
        """Set parameter constraints for a strategy."""
        try:
            self.parameter_constraints[strategy_name] = constraints
            
            # Store in memory
            self.memory.store('parameter_constraints', {
                strategy_name: [c.__dict__ for c in constraints],
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            self.logger.error(f"Error setting parameter constraints: {str(e)}")