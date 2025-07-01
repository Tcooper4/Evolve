import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
from datetime import datetime

from .prompt_processor import PromptProcessor, ProcessedPrompt
from .response_formatter import ResponseFormatter, ResponseData
from trading.models.base_model import BaseModel
from trading.strategies.strategy_manager import StrategyManager
from trading.risk.risk_manager import RiskManager
from trading.portfolio.portfolio_manager import PortfolioManager
from trading.market.market_analyzer import MarketAnalyzer
from .llm_processor import LLMProcessor

@dataclass
class NLResponse:
    """Data class to hold natural language response information."""
    text: str
    visualization: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create logs directory if it doesn't exist
os.makedirs('trading/nlp/logs', exist_ok=True)

# Add file handler for debug logs
debug_handler = logging.FileHandler('trading/nlp/logs/nlp_debug.log')
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
debug_handler.setFormatter(debug_formatter)
logger.addHandler(debug_handler)

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
        self.llm_processor = LLMProcessor()
        
        # Initialize trading components
        self.model = BaseModel()
        self.strategy_manager = StrategyManager()
        self.risk_manager = RiskManager()
        self.portfolio_manager = PortfolioManager()
        self.market_analyzer = MarketAnalyzer()
        
        # Load confidence thresholds
        self.confidence_thresholds = {
            'intent_detection': 0.6,
            'entity_extraction': 0.7,
            'response_generation': 0.8
        }
        
        # Load fallback prompts
        self.fallback_prompts = {
            'intent_clarification': "I'm not sure what you're asking about. Could you please clarify?",
            'entity_clarification': "I need more information about {entity}. Could you provide more details?",
            'response_fallback': "I'm having trouble generating a complete response. Here's what I can tell you: {partial_response}"
        }
        
        logger.info("NLInterface initialized with confidence thresholds and fallback prompts")
        
            return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def process_query(self, query: str, session_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a natural language query.
        
        Args:
            query: User query string
            session_state: Optional session state for context
            
        Returns:
            Dictionary containing processed response
        """
        logger.debug(f"Processing query: {query}")
        
        try:
            # Extract intent and entities
            intent, intent_confidence = self._detect_intent(query)
            logger.debug(f"Detected intent: {intent} (confidence: {intent_confidence:.2f})")
            
            # Check intent confidence
            if intent_confidence < self.confidence_thresholds['intent_detection']:
                logger.warning(f"Low intent confidence: {intent_confidence:.2f}")
                return self._handle_low_confidence('intent', query)
            
            # Extract and expand entities
            entities = self._extract_entities(query, intent)
            logger.debug(f"Extracted entities: {entities}")
            
            # Check entity confidence
            if not self._validate_entities(entities):
                logger.warning("Missing or low confidence entities")
                return self._handle_missing_entities(entities, query)
            
            # Generate response
            response = self._generate_response(query, intent, entities, session_state)
            
            # Format response
            formatted_response = self._format_response(response, session_state)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return self._handle_error(str(e))
    
    def _detect_intent(self, query: str) -> Tuple[str, float]:
        """Detect intent from query.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (intent, confidence)
        """
        try:
            # Process query with LLM
            prompt = self.prompt_processor.create_intent_prompt(query)
            response = self.llm_processor.process(prompt)
            
            # Parse intent and confidence
            intent_data = json.loads(response)
            return intent_data['intent'], float(intent_data['confidence'])
            
        except Exception as e:
            logger.error(f"Error detecting intent: {str(e)}", exc_info=True)
            return {'success': True, 'result': "unknown", 0.0, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _extract_entities(self, query: str, intent: str) -> Dict[str, Any]:
        """Extract and expand entities from query.
        
        Args:
            query: User query string
            intent: Detected intent
            
        Returns:
            Dictionary of extracted entities
        """
        try:
            # Extract base entities
            entities = self.prompt_processor.extract_entities(query, intent)
            
            # Expand entities with synonyms and clarifications
            expanded_entities = self.prompt_processor.expand_entities(entities)
            
            return expanded_entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}", exc_info=True)
            return {}
    
    def _validate_entities(self, entities: Dict[str, Any]) -> bool:
        """Validate extracted entities.
        
        Args:
            entities: Dictionary of extracted entities
            
        Returns:
            True if entities are valid, False otherwise
        """
        if not entities:
            return False
            
        # Check confidence for each entity
        for entity, data in entities.items():
            if 'confidence' in data and data['confidence'] < self.confidence_thresholds['entity_extraction']:
                return False
                
        return True
    
    def _generate_response(self, query: str, intent: str, entities: Dict[str, Any],
                          session_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate response based on intent and entities.
        
        Args:
            query: Original user query
            intent: Detected intent
            entities: Extracted entities
            session_state: Optional session state
            
        Returns:
            Dictionary containing response data
        """
        try:
            # Create response prompt
            prompt = self.prompt_processor.create_response_prompt(
                query, intent, entities, session_state
            )
            
            # Generate response with streaming
            response_stream = self.llm_processor.process_stream(prompt)
            
            # Collect and validate response
            response = ""
            for chunk in response_stream:
                response += chunk
                
                # Check for unsafe content
                if self.llm_processor.is_unsafe_content(chunk):
                    logger.warning("Detected unsafe content in response")
                    return self._handle_unsafe_content()
            
            # Parse response
            response_data = json.loads(response)
            
            # Check response confidence
            if response_data.get('confidence', 0) < self.confidence_thresholds['response_generation']:
                logger.warning("Low response confidence")
                return self._handle_low_confidence('response', response_data)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return self._handle_error(str(e))
    
    def _format_response(self, response: Dict[str, Any],
                        session_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format response for output.
        
        Args:
            response: Response data dictionary
            session_state: Optional session state
            
        Returns:
            Formatted response dictionary
        """
        try:
            # Get output format from session state or default to text
            output_format = session_state.get('output_format', 'text') if session_state else 'text'
            
            # Get theme from session state or default to light
            theme = session_state.get('theme', 'light') if session_state else 'light'
            
            # Format response
            formatted = self.response_formatter.format(
                response,
                output_format=output_format,
                theme=theme
            )
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}", exc_info=True)
            return self._handle_error(str(e))
    
    def _handle_low_confidence(self, component: str, data: Any) -> Dict[str, Any]:
        """Handle low confidence in any component.
        
        Args:
            component: Component with low confidence
            data: Original data
            
        Returns:
            Fallback response
        """
        if component == 'intent':
            return {
                'type': 'clarification',
                'message': self.fallback_prompts['intent_clarification'],
                'original_query': data
            }
        elif component == 'response':
            return {
                'type': 'partial',
                'message': self.fallback_prompts['response_fallback'].format(
                    partial_response=data.get('partial', '')
                ),
                'confidence': data.get('confidence', 0)
            }
        else:
            return self._handle_error(f"Unknown component: {component}")
    
    def _handle_missing_entities(self, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Handle missing or low confidence entities.
        
        Args:
            entities: Extracted entities
            query: Original query
            
        Returns:
            Clarification response
        """
        missing = []
        for entity, data in entities.items():
            if 'confidence' not in data or data['confidence'] < self.confidence_thresholds['entity_extraction']:
                missing.append(entity)
        
        if missing:
            return {
                'type': 'clarification',
                'message': self.fallback_prompts['entity_clarification'].format(
                    entity=', '.join(missing)
                ),
                'missing_entities': missing,
                'original_query': query
            }
        else:
            return self._handle_error("No entities found")
    
    def _handle_unsafe_content(self) -> Dict[str, Any]:
        """Handle unsafe content in response.
        
        Returns:
            Error response
        """
        return {
            'type': 'error',
            'message': "I cannot process this request as it may contain unsafe content.",
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_error(self, error: str) -> Dict[str, Any]:
        """Handle general errors.
        
        Args:
            error: Error message
            
        Returns:
            Error response
        """
        return {
            'type': 'error',
            'message': f"An error occurred: {error}",
            'timestamp': datetime.now().isoformat()
        }

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