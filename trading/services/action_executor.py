"""
Action Executor Module

Handles execution of different trading actions based on parsed query intent.
Provides model recommendations, trading signals, and market analysis.
"""

import logging
from typing import Dict, Any, List, Optional
from trading.services.service_client import ServiceClient

logger = logging.getLogger(__name__)

class ActionExecutor:
    """
    Executes different trading actions based on parsed query intent.
    
    Handles model recommendations, trading signals, market analysis,
    and general analysis requests.
    """
    
    def __init__(self, service_client: ServiceClient):
        """
        Initialize the action executor.
        
        Args:
            service_client: Service client for interacting with trading services
        """
        self.client = service_client
    
    def execute_action(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the appropriate action based on parsed intent.
        
        Args:
            parsed: Parsed query parameters
            
        Returns:
            Results from the executed action
        """
        intent = parsed.get('intent')
        symbol = parsed.get('symbol')
        timeframe = parsed.get('timeframe', '1h')
        period = parsed.get('period', '30d')
        
        if intent == 'model_recommendation':
            return self._get_model_recommendation(symbol, timeframe, period)
        elif intent == 'trading_signal':
            return self._get_trading_signal(symbol, timeframe, period)
        elif intent == 'market_analysis':
            return self._get_market_analysis(symbol, timeframe, period)
        else:
            return self._get_general_analysis(symbol, timeframe, period)
    
    def _get_model_recommendation(self, symbol: str, timeframe: str, period: str) -> Dict[str, Any]:
        """Get the best model recommendation for a symbol."""
        try:
            # Build multiple models
            models = []
            for model_type in ['lstm', 'xgboost', 'ensemble']:
                result = self.client.build_model(model_type, symbol, timeframe)
                if result and result.get('status') == 'success':
                    models.append({
                        'model_type': model_type,
                        'model_info': result.get('model_info', {})
                    })
            
            # Evaluate all models
            evaluations = []
            for model in models:
                model_id = model['model_info'].get('model_id')
                if model_id:
                    eval_result = self.client.evaluate_model(model_id, symbol, timeframe, period)
                    if eval_result and eval_result.get('status') == 'success':
                        evaluations.append({
                            'model_type': model['model_type'],
                            'model_id': model_id,
                            'evaluation': eval_result.get('evaluation', {})
                        })
            
            # Find best model
            best_model = None
            best_score = -1
            
            for eval_data in evaluations:
                score = eval_data['evaluation'].get('overall_score', 0)
                if score > best_score:
                    best_score = score
                    best_model = eval_data
            
            return {
                'action': 'model_recommendation',
                'symbol': symbol,
                'timeframe': timeframe,
                'period': period,
                'models_built': len(models),
                'models_evaluated': len(evaluations),
                'best_model': best_model,
                'all_evaluations': evaluations
            }
            
        except Exception as e:
            logger.error(f"Error getting model recommendation: {e}")
            return {
                'action': 'model_recommendation',
                'error': str(e),
                'symbol': symbol,
                'timeframe': timeframe,
                'period': period
            }
    
    def _get_trading_signal(self, symbol: str, timeframe: str, period: str) -> Dict[str, Any]:
        """Get trading signal for a symbol."""
        try:
            # First get the best model
            model_result = self._get_model_recommendation(symbol, timeframe, period)
            
            if 'error' in model_result:
                return model_result
            
            best_model = model_result.get('best_model')
            if not best_model:
                return {
                    'action': 'trading_signal',
                    'error': 'No suitable model found',
                    'symbol': symbol
                }
            
            # Generate forecast using the best model
            model_id = best_model['model_id']
            forecast_result = self._generate_forecast(model_id, symbol, timeframe)
            
            # Generate trading signal based on forecast
            signal = self._generate_signal(forecast_result, best_model['evaluation'])
            
            return {
                'action': 'trading_signal',
                'symbol': symbol,
                'timeframe': timeframe,
                'period': period,
                'best_model': best_model,
                'forecast': forecast_result,
                'signal': signal
            }
            
        except Exception as e:
            logger.error(f"Error getting trading signal: {e}")
            return {
                'action': 'trading_signal',
                'error': str(e),
                'symbol': symbol
            }
    
    def _get_market_analysis(self, symbol: str, timeframe: str, period: str) -> Dict[str, Any]:
        """Get comprehensive market analysis."""
        try:
            # Get model recommendation
            model_result = self._get_model_recommendation(symbol, timeframe, period)
            
            # Get market data analysis
            market_data = self._get_market_data(symbol, timeframe, period)
            
            # Generate plots
            plots = []
            for plot_type in ['price_chart', 'volume_analysis', 'technical_indicators']:
                plot_result = self.client.generate_plot(plot_type, f"{symbol}_{timeframe}")
                if plot_result and plot_result.get('status') == 'success':
                    plots.append({
                        'type': plot_type,
                        'path': plot_result.get('result', {}).get('save_path')
                    })
            
            return {
                'action': 'market_analysis',
                'symbol': symbol,
                'timeframe': timeframe,
                'period': period,
                'model_analysis': model_result,
                'market_data': market_data,
                'plots': plots
            }
            
        except Exception as e:
            logger.error(f"Error getting market analysis: {e}")
            return {
                'action': 'market_analysis',
                'error': str(e),
                'symbol': symbol
            }
    
    def _get_general_analysis(self, symbol: str, timeframe: str, period: str) -> Dict[str, Any]:
        """Get general analysis for any query."""
        try:
            # Combine multiple analyses
            model_result = self._get_model_recommendation(symbol, timeframe, period)
            market_result = self._get_market_analysis(symbol, timeframe, period)
            
            return {
                'action': 'general_analysis',
                'symbol': symbol,
                'timeframe': timeframe,
                'period': period,
                'model_analysis': model_result,
                'market_analysis': market_result
            }
            
        except Exception as e:
            logger.error(f"Error getting general analysis: {e}")
            return {
                'action': 'general_analysis',
                'error': str(e),
                'symbol': symbol
            }
    
    def _generate_forecast(self, model_id: str, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Generate forecast using a specific model."""
        # This would integrate with your existing forecasting system
        # For now, return a mock forecast
        return {
            'model_id': model_id,
            'symbol': symbol,
            'timeframe': timeframe,
            'prediction': 'bullish',
            'confidence': 0.75,
            'price_target': 150.0,
            'time_horizon': '1d'
        }
    
    def _generate_signal(self, forecast: Dict[str, Any], evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal based on forecast and model evaluation."""
        prediction = forecast.get('prediction', 'neutral')
        confidence = forecast.get('confidence', 0.5)
        model_score = evaluation.get('overall_score', 0.5)
        
        # Simple signal generation logic
        if prediction == 'bullish' and confidence > 0.6 and model_score > 0.6:
            signal = 'BUY'
            strength = 'strong' if confidence > 0.8 else 'moderate'
        elif prediction == 'bearish' and confidence > 0.6 and model_score > 0.6:
            signal = 'SELL'
            strength = 'strong' if confidence > 0.8 else 'moderate'
        else:
            signal = 'HOLD'
            strength = 'weak'
        
        return {
            'signal': signal,
            'strength': strength,
            'confidence': confidence,
            'model_score': model_score,
            'reasoning': f"Model predicts {prediction} with {confidence:.1%} confidence"
        }
    
    def _get_market_data(self, symbol: str, timeframe: str, period: str) -> Dict[str, Any]:
        """Get market data analysis."""
        # This would integrate with your data providers
        # For now, return mock data
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'period': period,
            'current_price': 145.0,
            'price_change': 2.5,
            'volume': 1000000,
            'volatility': 0.15,
            'trend': 'uptrend'
        } 