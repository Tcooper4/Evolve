"""Enhanced Prompt Agent for Trading System.

This module provides an intelligent agent that can route user prompts through
the complete trading pipeline: Forecast → Strategy → Backtest → Report → Trade.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re
from dataclasses import dataclass
from pathlib import Path

from trading.models.forecast_router import ForecastRouter
from trading.backtesting.backtester import Backtester
from trading.strategies.gatekeeper import StrategyGatekeeper
from trading.execution.trade_executor import TradeExecutor
from trading.optimization.self_tuning_optimizer import SelfTuningOptimizer
from trading.data.providers.fallback_provider import FallbackDataProvider

logger = logging.getLogger(__name__)

@dataclass
class AgentResponse:
    """Response from the prompt agent."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    visualizations: Optional[List[Any]] = None
    recommendations: Optional[List[str]] = None
    next_actions: Optional[List[str]] = None

class PromptAgent:
    """Enhanced prompt agent with full trading pipeline routing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize prompt agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.forecast_router = ForecastRouter()
        self.strategy_gatekeeper = StrategyGatekeeper()
        self.trade_executor = TradeExecutor()
        self.optimizer = SelfTuningOptimizer()
        self.data_provider = FallbackDataProvider()
        
        # Strategy registry
        self.strategy_registry = {
            'rsi': 'RSI Mean Reversion',
            'bollinger': 'Bollinger Bands',
            'macd': 'MACD Strategy',
            'sma': 'Moving Average Crossover',
            'garch': 'GARCH Volatility',
            'ridge': 'Ridge Regression',
            'informer': 'Informer Model',
            'transformer': 'Transformer',
            'autoformer': 'Autoformer',
            'lstm': 'LSTM Strategy',
            'xgboost': 'XGBoost Strategy',
            'ensemble': 'Ensemble Strategy'
        }
        
        # Model registry
        self.model_registry = {
            'arima': 'ARIMA',
            'lstm': 'LSTM',
            'xgboost': 'XGBoost',
            'prophet': 'Prophet',
            'autoformer': 'Autoformer',
            'transformer': 'Transformer',
            'informer': 'Informer',
            'garch': 'GARCH',
            'ridge': 'Ridge Regression'
        }
        
        logger.info("Enhanced Prompt Agent initialized with full pipeline routing")
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def process_prompt(self, prompt: str) -> AgentResponse:
        """Process user prompt and route through trading pipeline.
        
        Args:
            prompt: User prompt
            
        Returns:
            Agent response with results and recommendations
        """
        try:
            logger.info(f"Processing prompt: {prompt}")
            
            # Parse prompt to extract intent and parameters
            intent, params = self._parse_prompt(prompt)
            
            if intent == 'forecast':
                return self._handle_forecast_request(params)
            elif intent == 'strategy':
                return self._handle_strategy_request(params)
            elif intent == 'backtest':
                return self._handle_backtest_request(params)
            elif intent == 'trade':
                return self._handle_trade_request(params)
            elif intent == 'optimize':
                return self._handle_optimization_request(params)
            elif intent == 'analyze':
                return self._handle_analysis_request(params)
            else:
                return {'success': True, 'result': self._handle_general_request(prompt, params), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
                
        except Exception as e:
            logger.error(f"Error processing prompt: {e}")
            return AgentResponse(
                success=False,
                message=f"Error processing request: {str(e)}",
                recommendations=["Please try rephrasing your request"]
            )
    
    def _parse_prompt(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Parse prompt to extract intent and parameters.
        
        Args:
            prompt: User prompt
            
        Returns:
            Tuple of (intent, parameters)
        """
        prompt_lower = prompt.lower()
        
        # Extract symbol
        symbol_match = re.search(r'\b([A-Z]{1,5})\b', prompt.upper())
        symbol = symbol_match.group(1) if symbol_match else 'AAPL'
        
        # Extract timeframe
        timeframe = '15d'
        if 'next' in prompt_lower and 'day' in prompt_lower:
            timeframe_match = re.search(r'next (\d+)d?', prompt_lower)
            if timeframe_match:
                timeframe = f"{timeframe_match.group(1)}d"
        elif 'next' in prompt_lower and 'week' in prompt_lower:
            timeframe_match = re.search(r'next (\d+)w?', prompt_lower)
            if timeframe_match:
                timeframe = f"{int(timeframe_match.group(1)) * 7}d"
        elif 'next' in prompt_lower and 'month' in prompt_lower:
            timeframe_match = re.search(r'next (\d+)m?', prompt_lower)
            if timeframe_match:
                timeframe = f"{int(timeframe_match.group(1)) * 30}d"
        
        # Determine intent
        if any(word in prompt_lower for word in ['forecast', 'predict', 'price']):
            intent = 'forecast'
        elif any(word in prompt_lower for word in ['strategy', 'strategy', 'signal']):
            intent = 'strategy'
        elif any(word in prompt_lower for word in ['backtest', 'test', 'simulate']):
            intent = 'backtest'
        elif any(word in prompt_lower for word in ['trade', 'buy', 'sell', 'execute']):
            intent = 'trade'
        elif any(word in prompt_lower for word in ['optimize', 'improve', 'tune']):
            intent = 'optimize'
        elif any(word in prompt_lower for word in ['analyze', 'analysis', 'report']):
            intent = 'analyze'
        else:
            intent = 'general'
        
        # Extract strategy preference
        strategy = None
        for strategy_key, strategy_name in self.strategy_registry.items():
            if strategy_key in prompt_lower:
                strategy = strategy_name
                break
        
        # Extract model preference
        model = None
        for model_key, model_name in self.model_registry.items():
            if model_key in prompt_lower:
                model = model_name
                break
        
        # Auto-select best strategy if none specified
        if not strategy:
            strategy = self._select_best_strategy(symbol)
        
        # Auto-select best model if none specified
        if not model:
            model = self._select_best_model(symbol, timeframe)
        
        params = {
            'symbol': symbol,
            'timeframe': timeframe,
            'strategy': strategy,
            'model': model,
            'prompt': prompt
        }
        
        logger.info(f"Parsed prompt - Intent: {intent}, Params: {params}")
        
        return {'success': True, 'result': intent, params, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _select_best_strategy(self, symbol: str) -> str:
        """Select best strategy based on symbol characteristics.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Best strategy name
        """
        # Simple heuristic based on symbol type
        if symbol in ['BTC', 'ETH', 'ADA', 'DOT']:
            return 'RSI Mean Reversion'  # Good for crypto
        elif symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
            return 'Moving Average Crossover'  # Good for tech stocks
        elif symbol in ['SPY', 'QQQ', 'IWM']:
            return 'Bollinger Bands'  # Good for ETFs
        else:
            return 'Ensemble Strategy'  # Default to ensemble
    
    def _select_best_model(self, symbol: str, timeframe: str) -> str:
        """Select best model based on symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Forecast timeframe
            
        Returns:
            Best model name
        """
        # Parse timeframe
        days = int(timeframe.replace('d', ''))
        
        if days <= 7:
            return 'ARIMA'  # Good for short-term
        elif days <= 30:
            return 'LSTM'  # Good for medium-term
        else:
            return 'Transformer'  # Good for long-term
    
    def _handle_forecast_request(self, params: Dict[str, Any]) -> AgentResponse:
        """Handle forecast request.
        
        Args:
            params: Request parameters
            
        Returns:
            Agent response
        """
        try:
            symbol = params['symbol']
            timeframe = params['timeframe']
            model = params['model']
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 year of data
            
            data = self.data_provider.get_historical_data(
                symbol, start_date, end_date, '1d'
            )
            
            if data is None or data.empty:
                return AgentResponse(
                    success=False,
                    message=f"Unable to get data for {symbol}",
                    recommendations=["Try a different symbol or check data availability"]
                )
            
            # Generate forecast
            horizon = int(timeframe.replace('d', ''))
            forecast_result = self.forecast_router.get_forecast(
                data=data,
                horizon=horizon,
                model_type=model.lower()
            )
            
            # Create response
            message = f"Forecast for {symbol} using {model} model:\n"
            message += f"Horizon: {timeframe}\n"
            message += f"Confidence: {forecast_result['confidence']:.2%}\n"
            
            if 'warnings' in forecast_result:
                message += f"Warnings: {', '.join(forecast_result['warnings'])}\n"
            
            recommendations = [
                f"Consider using {forecast_result['model']} for future forecasts",
                "Monitor forecast accuracy and adjust model selection",
                "Use multiple models for ensemble forecasting"
            ]
            
            next_actions = [
                f"Run backtest with {params['strategy']} strategy",
                "Generate trading signals",
                "Execute paper trade"
            ]
            
            return AgentResponse(
                success=True,
                message=message,
                data=forecast_result,
                recommendations=recommendations,
                next_actions=next_actions
            )
            
        except Exception as e:
            logger.error(f"Error in forecast request: {e}")
            return AgentResponse(
                success=False,
                message=f"Forecast failed: {str(e)}",
                recommendations=["Try a different model or timeframe"]
            )
    
    def _handle_strategy_request(self, params: Dict[str, Any]) -> AgentResponse:
        """Handle strategy request.
        
        Args:
            params: Request parameters
            
        Returns:
            Agent response
        """
        try:
            symbol = params['symbol']
            strategy = params['strategy']
            
            # Get strategy analysis
            strategy_analysis = self.strategy_gatekeeper.analyze_strategy(
                strategy, symbol
            )
            
            message = f"Strategy Analysis for {strategy} on {symbol}:\n"
            message += f"Health Score: {strategy_analysis.get('health_score', 'N/A')}\n"
            message += f"Risk Level: {strategy_analysis.get('risk_level', 'N/A')}\n"
            
            recommendations = strategy_analysis.get('recommendations', [])
            
            next_actions = [
                "Run backtest to validate strategy",
                "Optimize strategy parameters",
                "Execute strategy in paper trading"
            ]
            
            return AgentResponse(
                success=True,
                message=message,
                data=strategy_analysis,
                recommendations=recommendations,
                next_actions=next_actions
            )
            
        except Exception as e:
            logger.error(f"Error in strategy request: {e}")
            return AgentResponse(
                success=False,
                message=f"Strategy analysis failed: {str(e)}",
                recommendations=["Try a different strategy or check symbol availability"]
            )
    
    def _handle_backtest_request(self, params: Dict[str, Any]) -> AgentResponse:
        """Handle backtest request.
        
        Args:
            params: Request parameters
            
        Returns:
            Agent response
        """
        try:
            symbol = params['symbol']
            strategy = params['strategy']
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            data = self.data_provider.get_historical_data(
                symbol, start_date, end_date, '1d'
            )
            
            if data is None or data.empty:
                return AgentResponse(
                    success=False,
                    message=f"Unable to get data for {symbol}",
                    recommendations=["Try a different symbol or check data availability"]
                )
            
            # Run backtest
            backtester = Backtester(data)
            equity_curve, trade_log, metrics = backtester.run_backtest([strategy])
            
            # Analyze results
            sharpe = metrics.get('sharpe_ratio', 0)
            total_return = metrics.get('total_return', 0)
            max_dd = metrics.get('max_drawdown', 0)
            
            message = f"Backtest Results for {strategy} on {symbol}:\n"
            message += f"Sharpe Ratio: {sharpe:.2f}\n"
            message += f"Total Return: {total_return:.2%}\n"
            message += f"Max Drawdown: {max_dd:.2%}\n"
            message += f"Win Rate: {metrics.get('win_rate', 0):.2%}\n"
            
            recommendations = []
            if sharpe < 1.0:
                recommendations.append("Sharpe ratio below 1.0 - consider strategy optimization")
            if max_dd > 0.2:
                recommendations.append("High drawdown - implement better risk management")
            if total_return < 0.05:
                recommendations.append("Low returns - consider alternative strategies")
            
            if not recommendations:
                recommendations.append("Strategy performing well - consider live trading")
            
            next_actions = []
            if sharpe >= 1.0 and total_return > 0.1:
                next_actions.append("Execute paper trade")
                next_actions.append("Optimize strategy parameters")
            else:
                next_actions.append("Try different strategy")
                next_actions.append("Adjust risk parameters")
            
            return AgentResponse(
                success=True,
                message=message,
                data={
                    'metrics': metrics,
                    'equity_curve': equity_curve,
                    'trade_log': trade_log
                },
                recommendations=recommendations,
                next_actions=next_actions
            )
            
        except Exception as e:
            logger.error(f"Error in backtest request: {e}")
            return AgentResponse(
                success=False,
                message=f"Backtest failed: {str(e)}",
                recommendations=["Check data availability and strategy parameters"]
            )
    
    def _handle_trade_request(self, params: Dict[str, Any]) -> AgentResponse:
        """Handle trade request.
        
        Args:
            params: Request parameters
            
        Returns:
            Agent response
        """
        try:
            symbol = params['symbol']
            
            # Get current market price
            current_price = self.data_provider.get_live_price(symbol)
            
            if current_price is None:
                return AgentResponse(
                    success=False,
                    message=f"Unable to get current price for {symbol}",
                    recommendations=["Check symbol validity and market hours"]
                )
            
            # Simulate trade execution
            quantity = 100  # Default quantity
            side = 'buy'  # Default side
            
            execution_result = self.trade_executor.simulate_trade(
                symbol=symbol,
                side=side,
                quantity=quantity,
                market_price=current_price
            )
            
            if execution_result.success:
                message = f"Trade Simulation for {symbol}:\n"
                message += f"Side: {side.upper()}\n"
                message += f"Quantity: {quantity}\n"
                message += f"Execution Price: ${execution_result.execution_price:.2f}\n"
                message += f"Commission: ${execution_result.commission:.2f}\n"
                message += f"Slippage: {execution_result.slippage:.4f}\n"
                message += f"Total Cost: ${execution_result.total_cost:.2f}\n"
                
                recommendations = [
                    "Monitor trade performance",
                    "Set stop-loss and take-profit levels",
                    "Review execution quality"
                ]
                
                next_actions = [
                    "Monitor position",
                    "Set up alerts",
                    "Plan exit strategy"
                ]
            else:
                message = f"Trade execution failed: {execution_result.error_message}"
                recommendations = ["Check market conditions", "Verify order parameters"]
                next_actions = ["Retry trade", "Adjust parameters"]
            
            return AgentResponse(
                success=execution_result.success,
                message=message,
                data={
                    'execution_result': execution_result,
                    'current_price': current_price
                },
                recommendations=recommendations,
                next_actions=next_actions
            )
            
        except Exception as e:
            logger.error(f"Error in trade request: {e}")
            return AgentResponse(
                success=False,
                message=f"Trade execution failed: {str(e)}",
                recommendations=["Check market hours and symbol validity"]
            )
    
    def _handle_optimization_request(self, params: Dict[str, Any]) -> AgentResponse:
        """Handle optimization request.
        
        Args:
            params: Request parameters
            
        Returns:
            Agent response
        """
        try:
            strategy = params['strategy']
            
            # Get current performance metrics
            current_metrics = {
                'sharpe_ratio': 0.8,  # Mock metrics
                'total_return': 0.15,
                'max_drawdown': 0.12,
                'win_rate': 0.55
            }
            
            # Run optimization
            optimization_result = self.optimizer.optimize_strategy(
                strategy=strategy,
                current_parameters={'period': 14, 'threshold': 0.05},
                current_metrics=current_metrics
            )
            
            if optimization_result:
                message = f"Optimization Results for {strategy}:\n"
                message += f"Confidence: {optimization_result.confidence:.2%}\n"
                message += f"Improvements: {optimization_result.improvement}\n"
                
                recommendations = optimization_result.recommendations
                next_actions = [
                    "Apply new parameters",
                    "Run backtest with optimized parameters",
                    "Monitor performance improvement"
                ]
            else:
                message = f"No optimization needed for {strategy}"
                recommendations = ["Continue monitoring performance"]
                next_actions = ["Run periodic optimization checks"]
            
            return AgentResponse(
                success=True,
                message=message,
                data={'optimization_result': optimization_result},
                recommendations=recommendations,
                next_actions=next_actions
            )
            
        except Exception as e:
            logger.error(f"Error in optimization request: {e}")
            return AgentResponse(
                success=False,
                message=f"Optimization failed: {str(e)}",
                recommendations=["Check strategy configuration and historical data"]
            )
    
    def _handle_analysis_request(self, params: Dict[str, Any]) -> AgentResponse:
        """Handle analysis request.
        
        Args:
            params: Request parameters
            
        Returns:
            Agent response
        """
        try:
            symbol = params['symbol']
            
            # Get market data
            market_data = self.data_provider.get_market_data([symbol])
            
            if symbol not in market_data:
                return AgentResponse(
                    success=False,
                    message=f"Unable to get market data for {symbol}",
                    recommendations=["Check symbol validity and market hours"]
                )
            
            data = market_data[symbol]
            
            message = f"Market Analysis for {symbol}:\n"
            message += f"Current Price: ${data['price']:.2f}\n"
            message += f"Change: {data['change']:+.2f} ({data['change_pct']:+.2f}%)\n"
            message += f"Volume: {data['volume']:,.0f}\n"
            
            recommendations = [
                "Monitor key support/resistance levels",
                "Check for news catalysts",
                "Review technical indicators"
            ]
            
            next_actions = [
                "Generate price forecast",
                "Run technical analysis",
                "Check fundamental data"
            ]
            
            return AgentResponse(
                success=True,
                message=message,
                data=market_data,
                recommendations=recommendations,
                next_actions=next_actions
            )
            
        except Exception as e:
            logger.error(f"Error in analysis request: {e}")
            return AgentResponse(
                success=False,
                message=f"Analysis failed: {str(e)}",
                recommendations=["Check data availability and symbol validity"]
            )
    
    def _handle_general_request(self, prompt: str, params: Dict[str, Any]) -> AgentResponse:
        """Handle general request with full pipeline.
        
        Args:
            prompt: Original prompt
            params: Parsed parameters
            
        Returns:
            Agent response
        """
        try:
            # Run full pipeline: Forecast → Strategy → Backtest → Report
            results = {}
            
            # 1. Generate forecast
            forecast_response = self._handle_forecast_request(params)
            if forecast_response.success:
                results['forecast'] = forecast_response.data
            
            # 2. Analyze strategy
            strategy_response = self._handle_strategy_request(params)
            if strategy_response.success:
                results['strategy'] = strategy_response.data
            
            # 3. Run backtest
            backtest_response = self._handle_backtest_request(params)
            if backtest_response.success:
                results['backtest'] = backtest_response.data
            
            # 4. Generate comprehensive report
            message = f"Complete Analysis for {params['symbol']}:\n\n"
            
            if 'forecast' in results:
                message += "📈 FORECAST:\n"
                message += f"Model: {results['forecast']['model']}\n"
                message += f"Confidence: {results['forecast']['confidence']:.2%}\n\n"
            
            if 'strategy' in results:
                message += "🎯 STRATEGY:\n"
                message += f"Health Score: {results['strategy'].get('health_score', 'N/A')}\n"
                message += f"Risk Level: {results['strategy'].get('risk_level', 'N/A')}\n\n"
            
            if 'backtest' in results:
                metrics = results['backtest']['metrics']
                message += "📊 BACKTEST RESULTS:\n"
                message += f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
                message += f"Total Return: {metrics.get('total_return', 0):.2%}\n"
                message += f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
                message += f"Win Rate: {metrics.get('win_rate', 0):.2%}\n\n"
            
            # Generate recommendations
            recommendations = []
            if 'backtest' in results:
                metrics = results['backtest']['metrics']
                sharpe = metrics.get('sharpe_ratio', 0)
                
                if sharpe >= 1.0:
                    recommendations.append("✅ Strategy performing well - consider live trading")
                elif sharpe >= 0.5:
                    recommendations.append("⚠️ Strategy needs optimization - run parameter tuning")
                else:
                    recommendations.append("❌ Strategy underperforming - try alternative approach")
            
            recommendations.extend([
                "Monitor performance regularly",
                "Set up automated alerts",
                "Review and adjust parameters monthly"
            ])
            
            # Suggest next actions
            next_actions = [
                "Execute paper trade",
                "Set up performance monitoring",
                "Schedule regular reviews"
            ]
            
            return AgentResponse(
                success=True,
                message=message,
                data=results,
                recommendations=recommendations,
                next_actions=next_actions
            )
            
        except Exception as e:
            logger.error(f"Error in general request: {e}")
            return AgentResponse(
                success=False,
                message=f"Analysis failed: {str(e)}",
                recommendations=["Try breaking down the request into smaller parts"]
            )

# Global prompt agent instance
prompt_agent = PromptAgent()

def get_prompt_agent() -> PromptAgent:
    """Get the global prompt agent instance."""
    return prompt_agent