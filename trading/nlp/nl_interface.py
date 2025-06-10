import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path

from .prompt_processor import PromptProcessor, ProcessedPrompt
from .response_formatter import ResponseFormatter, ResponseData
from ..models.base_model import BaseModel
from ..strategies.strategy_manager import StrategyManager
from ..risk.risk_manager import RiskManager
from ..portfolio.portfolio_manager import PortfolioManager
from ..market.market_analyzer import MarketAnalyzer

@dataclass
class NLResponse:
    """Data class to hold natural language response information."""
    text: str
    visualization: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None

class NLInterface:
    """Class to handle natural language interactions with the trading system."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the natural language interface.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent / "config"
        
        # Initialize components
        self.prompt_processor = PromptProcessor(self.config_dir)
        self.response_formatter = ResponseFormatter(self.config_dir)
        
        # Initialize trading components
        self.model = BaseModel()
        self.strategy_manager = StrategyManager()
        self.risk_manager = RiskManager()
        self.portfolio_manager = PortfolioManager()
        self.market_analyzer = MarketAnalyzer()
        
    def process_query(self, query: str) -> NLResponse:
        """Process a natural language query.
        
        Args:
            query: The input query to process
            
        Returns:
            NLResponse object containing formatted response and visualization
        """
        try:
            # Process the prompt
            processed_prompt = self.prompt_processor.process_prompt(query)
            
            # Validate the prompt
            is_valid, missing_entities = self.prompt_processor.validate_prompt(processed_prompt)
            if not is_valid:
                return self._create_error_response(f"Missing required information: {', '.join(missing_entities)}")
                
            # Generate response based on intent
            response_data = self._generate_response(processed_prompt)
            
            # Format the response
            formatted_text = self.response_formatter.format_response(response_data)
            
            # Create visualization
            visualization = self.response_formatter.create_visualization(response_data)
            
            return NLResponse(
                text=formatted_text,
                visualization=visualization,
                metadata=response_data.metadata
            )
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return self._create_error_response(str(e))
            
    def _generate_response(self, prompt: ProcessedPrompt) -> ResponseData:
        """Generate response based on processed prompt.
        
        Args:
            prompt: ProcessedPrompt object containing intent and entities
            
        Returns:
            ResponseData object containing response content and metadata
        """
        try:
            if prompt.intent == "forecast":
                return self._generate_forecast(prompt)
            elif prompt.intent == "analyze":
                return self._generate_analysis(prompt)
            elif prompt.intent == "recommend":
                return self._generate_recommendation(prompt)
            elif prompt.intent == "explain":
                return self._generate_explanation(prompt)
            elif prompt.intent == "compare":
                return self._generate_comparison(prompt)
            elif prompt.intent == "optimize":
                return self._generate_optimization(prompt)
            elif prompt.intent == "validate":
                return self._generate_validation(prompt)
            elif prompt.intent == "monitor":
                return self._generate_monitoring(prompt)
            else:
                return self._create_error_response("Unknown intent")
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return self._create_error_response(str(e))
            
    def _generate_forecast(self, prompt: ProcessedPrompt) -> ResponseData:
        """Generate forecast response."""
        try:
            # Get required entities
            timeframe = self.prompt_processor.get_entity_by_type(prompt, "timeframe")
            asset = self.prompt_processor.get_entity_by_type(prompt, "asset")
            
            if not timeframe or not asset:
                return self._create_error_response("Missing required information for forecast")
                
            # Get historical data
            historical_data = self.market_analyzer.get_historical_data(
                asset.value,
                timeframe.value
            )
            
            # Generate forecast
            forecast = self.model.predict(historical_data)
            
            # Calculate confidence intervals
            confidence_intervals = self.model.get_confidence_intervals(forecast)
            
            return ResponseData(
                content={
                    "timeframe": timeframe.value,
                    "asset": asset.value,
                    "prediction": forecast.mean(),
                    "confidence": prompt.confidence * 100,
                    "factors": self.model.get_feature_importance(),
                    "historical_dates": historical_data.index,
                    "historical_values": historical_data.values,
                    "forecast_dates": forecast.index,
                    "forecast_values": forecast.values,
                    "confidence_intervals": confidence_intervals
                },
                type="forecast",
                confidence=prompt.confidence,
                metadata={
                    "timeframe": timeframe.value,
                    "asset": asset.value,
                    "model": self.model.__class__.__name__
                }
            )
        except Exception as e:
            self.logger.error(f"Error generating forecast: {e}")
            return self._create_error_response(str(e))
            
    def _generate_analysis(self, prompt: ProcessedPrompt) -> ResponseData:
        """Generate analysis response."""
        try:
            # Get required entities
            asset = self.prompt_processor.get_entity_by_type(prompt, "asset")
            
            if not asset:
                return self._create_error_response("Missing required information for analysis")
                
            # Get market data
            market_data = self.market_analyzer.get_market_data(asset.value)
            
            # Perform technical analysis
            technical_analysis = self.market_analyzer.analyze_technical(market_data)
            
            # Get market state
            market_state = self.market_analyzer.get_market_state(market_data)
            
            return ResponseData(
                content={
                    "asset": asset.value,
                    "timeframe": market_data.index.freq,
                    "state": market_state,
                    "indicators": technical_analysis,
                    "confidence": prompt.confidence * 100,
                    "dates": market_data.index,
                    "open": market_data["open"],
                    "high": market_data["high"],
                    "low": market_data["low"],
                    "close": market_data["close"],
                    "volume": market_data["volume"]
                },
                type="analysis",
                confidence=prompt.confidence,
                metadata={
                    "asset": asset.value,
                    "timeframe": market_data.index.freq,
                    "indicators": list(technical_analysis.keys())
                }
            )
        except Exception as e:
            self.logger.error(f"Error generating analysis: {e}")
            return self._create_error_response(str(e))
            
    def _generate_recommendation(self, prompt: ProcessedPrompt) -> ResponseData:
        """Generate recommendation response."""
        try:
            # Get required entities
            asset = self.prompt_processor.get_entity_by_type(prompt, "asset")
            action = self.prompt_processor.get_entity_by_type(prompt, "action")
            
            if not asset or not action:
                return self._create_error_response("Missing required information for recommendation")
                
            # Get market data
            market_data = self.market_analyzer.get_market_data(asset.value)
            
            # Generate trading signals
            signals = self.strategy_manager.generate_signals(market_data)
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                market_data,
                signals,
                self.portfolio_manager.get_portfolio_value()
            )
            
            # Calculate entry, stop loss, and take profit levels
            entry = self.strategy_manager.get_entry_level(market_data, signals)
            stop_loss = self.risk_manager.calculate_stop_loss(market_data, entry)
            take_profit = self.risk_manager.calculate_take_profit(market_data, entry)
            
            return ResponseData(
                content={
                    "asset": asset.value,
                    "action": action.value,
                    "entry": entry,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "confidence": prompt.confidence * 100,
                    "rationale": self.strategy_manager.get_signal_rationale(signals),
                    "dates": market_data.index,
                    "prices": market_data["close"],
                    "entry_date": market_data.index[-1],
                    "entry_price": entry
                },
                type="recommendation",
                confidence=prompt.confidence,
                metadata={
                    "asset": asset.value,
                    "action": action.value,
                    "strategy": self.strategy_manager.get_active_strategy()
                }
            )
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {e}")
            return self._create_error_response(str(e))
            
    def _generate_explanation(self, prompt: ProcessedPrompt) -> ResponseData:
        """Generate explanation response."""
        try:
            # Get required entities
            topic = self.prompt_processor.get_entity_by_type(prompt, "topic")
            
            if not topic:
                return self._create_error_response("Missing required information for explanation")
                
            # Get topic explanation
            explanation = self.market_analyzer.explain_topic(topic.value)
            
            # Get key points
            key_points = self.market_analyzer.get_key_points(topic.value)
            
            return ResponseData(
                content={
                    "topic": topic.value,
                    "analysis": explanation,
                    "points": key_points,
                    "confidence": prompt.confidence * 100
                },
                type="explanation",
                confidence=prompt.confidence,
                metadata={
                    "topic": topic.value,
                    "source": self.market_analyzer.get_explanation_source()
                }
            )
        except Exception as e:
            self.logger.error(f"Error generating explanation: {e}")
            return self._create_error_response(str(e))
            
    def _generate_comparison(self, prompt: ProcessedPrompt) -> ResponseData:
        """Generate comparison response."""
        try:
            # Get required entities
            assets = self.prompt_processor.get_entity_values(prompt, "asset")
            
            if not assets:
                return self._create_error_response("Missing required information for comparison")
                
            # Get market data for all assets
            market_data = {
                asset: self.market_analyzer.get_market_data(asset)
                for asset in assets
            }
            
            # Calculate correlation matrix
            correlation_matrix = self.market_analyzer.calculate_correlation(market_data)
            
            # Get comparative analysis
            comparison = self.market_analyzer.compare_assets(market_data)
            
            return ResponseData(
                content={
                    "assets": assets,
                    "timeframe": list(market_data.values())[0].index.freq,
                    "comparison": comparison,
                    "differences": self.market_analyzer.get_key_differences(market_data),
                    "confidence": prompt.confidence * 100,
                    "correlation_matrix": correlation_matrix.values
                },
                type="compare",
                confidence=prompt.confidence,
                metadata={
                    "assets": assets,
                    "timeframe": list(market_data.values())[0].index.freq,
                    "metrics": list(correlation_matrix.columns)
                }
            )
        except Exception as e:
            self.logger.error(f"Error generating comparison: {e}")
            return self._create_error_response(str(e))
            
    def _generate_optimization(self, prompt: ProcessedPrompt) -> ResponseData:
        """Generate optimization response."""
        try:
            # Get required entities
            strategy = self.prompt_processor.get_entity_by_type(prompt, "strategy")
            
            if not strategy:
                return self._create_error_response("Missing required information for optimization")
                
            # Get strategy parameters
            parameters = self.strategy_manager.get_strategy_parameters(strategy.value)
            
            # Optimize strategy
            optimization_result = self.strategy_manager.optimize_strategy(
                strategy.value,
                parameters
            )
            
            return ResponseData(
                content={
                    "strategy": strategy.value,
                    "parameters": optimization_result.best_params,
                    "performance": optimization_result.best_score,
                    "improvements": self.strategy_manager.get_improvements(optimization_result),
                    "confidence": prompt.confidence * 100,
                    "x_values": optimization_result.all_scores,
                    "y_values": optimization_result.convergence_history,
                    "best_x": optimization_result.best_score,
                    "best_y": optimization_result.best_params
                },
                type="optimize",
                confidence=prompt.confidence,
                metadata={
                    "strategy": strategy.value,
                    "optimization_time": optimization_result.optimization_time,
                    "method": optimization_result.method
                }
            )
        except Exception as e:
            self.logger.error(f"Error generating optimization: {e}")
            return self._create_error_response(str(e))
            
    def _generate_validation(self, prompt: ProcessedPrompt) -> ResponseData:
        """Generate validation response."""
        try:
            # Get required entities
            test = self.prompt_processor.get_entity_by_type(prompt, "test")
            
            if not test:
                return self._create_error_response("Missing required information for validation")
                
            # Run validation test
            test_results = self.strategy_manager.validate_strategy(test.value)
            
            return ResponseData(
                content={
                    "test": test.value,
                    "results": test_results.results,
                    "accuracy": test_results.accuracy,
                    "confidence": prompt.confidence * 100,
                    "metrics": test_results.metrics,
                    "values": test_results.values
                },
                type="validate",
                confidence=prompt.confidence,
                metadata={
                    "test": test.value,
                    "validation_time": test_results.validation_time,
                    "method": test_results.method
                }
            )
        except Exception as e:
            self.logger.error(f"Error generating validation: {e}")
            return self._create_error_response(str(e))
            
    def _generate_monitoring(self, prompt: ProcessedPrompt) -> ResponseData:
        """Generate monitoring response."""
        try:
            # Get required entities
            asset = self.prompt_processor.get_entity_by_type(prompt, "asset")
            
            if not asset:
                return self._create_error_response("Missing required information for monitoring")
                
            # Get portfolio status
            portfolio_status = self.portfolio_manager.get_portfolio_status()
            
            # Get performance metrics
            performance = self.portfolio_manager.get_performance_metrics()
            
            # Get alerts
            alerts = self.portfolio_manager.get_alerts()
            
            return ResponseData(
                content={
                    "asset": asset.value,
                    "status": portfolio_status,
                    "performance": performance,
                    "alerts": alerts,
                    "confidence": prompt.confidence * 100,
                    "dates": performance.index,
                    "thresholds": self.portfolio_manager.get_thresholds()
                },
                type="monitor",
                confidence=prompt.confidence,
                metadata={
                    "asset": asset.value,
                    "monitoring_time": pd.Timestamp.now(),
                    "alerts_count": len(alerts)
                }
            )
        except Exception as e:
            self.logger.error(f"Error generating monitoring: {e}")
            return self._create_error_response(str(e))
            
    def _create_error_response(self, error_message: str) -> ResponseData:
        """Create error response."""
        return ResponseData(
            content={
                "error": error_message
            },
            type="error",
            confidence=0.0,
            metadata={
                "error_type": "processing_error",
                "timestamp": pd.Timestamp.now()
            }
        ) 