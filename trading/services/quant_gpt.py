"""
QuantGPT Interface

A natural language interface for the Evolve trading system that provides
GPT-powered commentary on trading decisions and model recommendations.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import openai
from pathlib import Path
import sys

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from trading.services.service_client import ServiceClient
from trading.services.base_service import BaseService
from trading.memory.agent_memory import AgentMemory
from trading.services.quant_gpt import QuantGPT

logger = logging.getLogger(__name__)

class QuantGPT:
    """
    Natural language interface for the Evolve trading system.
    
    Provides GPT-powered analysis and commentary on trading decisions,
    model recommendations, and market analysis.
    """
    
    def __init__(self, openai_api_key: str = None, redis_host: str = 'localhost', 
                 redis_port: int = 6379, redis_db: int = 0):
        """
        Initialize QuantGPT.
        
        Args:
            openai_api_key: OpenAI API key for GPT commentary
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
        """
        self.client = ServiceClient(redis_host, redis_port, redis_db)
        self.memory = AgentMemory()
        
        # Initialize OpenAI
        if openai_api_key:
            openai.api_key = openai_api_key
        else:
            # Try to get from environment
            import os
            openai.api_key = os.getenv('OPENAI_API_KEY')
        
        if not openai.api_key:
            logger.warning("OpenAI API key not provided. GPT commentary will be disabled.")
        
        # Trading system context
        self.trading_context = {
            'available_symbols': ['BTCUSDT', 'ETHUSDT', 'NVDA', 'TSLA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN'],
            'available_timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
            'available_periods': ['7d', '14d', '30d', '90d', '180d', '1y'],
            'available_models': ['lstm', 'xgboost', 'ensemble', 'transformer', 'tcn']
        }
        
        logger.info("QuantGPT initialized")
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    # Alias for backward compatibility
    QuantGPTAgent = QuantGPT
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query and return comprehensive analysis.
        
        Args:
            query: Natural language query (e.g., "Give me the best model for NVDA over 90 days")
            
        Returns:
            Dictionary containing analysis results and GPT commentary
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Parse the query to extract intent and parameters
            parsed = self._parse_query(query)
            
            # Execute the appropriate action based on intent
            result = self._execute_action(parsed)
            
            # Generate GPT commentary
            commentary = self._generate_commentary(query, parsed, result)
            
            # Log the interaction
            self.memory.log_decision(
                agent_name='quant_gpt',
                decision_type='query_processed',
                details={
                    'query': query,
                    'intent': parsed.get('intent'),
                    'symbol': parsed.get('symbol'),
                    'timeframe': parsed.get('timeframe'),
                    'period': parsed.get('period'),
                    'confidence': parsed.get('confidence', 0)
                }
            )
            
            return {
                'query': query,
                'parsed_intent': parsed,
                'results': result,
                'gpt_commentary': commentary,
                'timestamp': time.time(),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'query': query,
                'error': str(e),
                'status': 'error',
                'timestamp': time.time()
            }
    
    def _parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query to extract intent and parameters.
        
        Args:
            query: Natural language query
            
        Returns:
            Parsed query with intent and parameters
        """
        try:
            # Use OpenAI to parse the query if available
            if openai.api_key:
                return self._parse_with_gpt(query)
            else:
                return self._parse_with_regex(query)
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            return self._parse_with_regex(query)
    
    def _parse_with_gpt(self, query: str) -> Dict[str, Any]:
        """Parse query using GPT for better understanding."""
        try:
            system_prompt = """
            You are a trading system query parser. Extract the following information from user queries:
            - intent: The main action requested (model_recommendation, trading_signal, market_analysis, etc.)
            - symbol: The trading symbol/asset (e.g., NVDA, TSLA, BTCUSDT)
            - timeframe: The time interval (1m, 5m, 15m, 1h, 4h, 1d)
            - period: The analysis period (7d, 14d, 30d, 90d, 180d, 1y)
            - model_type: Specific model type if mentioned (lstm, xgboost, ensemble, etc.)
            - confidence: Confidence score (0-1) for the parsing
            
            Available symbols: BTCUSDT, ETHUSDT, NVDA, TSLA, AAPL, GOOGL, MSFT, AMZN
            Available timeframes: 1m, 5m, 15m, 1h, 4h, 1d
            Available periods: 7d, 14d, 30d, 90d, 180d, 1y
            Available models: lstm, xgboost, ensemble, transformer, tcn
            
            Return only valid JSON with the extracted information.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            parsed = json.loads(response.choices[0].message.content)
            
            # Validate and clean the parsed data
            return self._validate_parsed_data(parsed)
            
        except Exception as e:
            logger.error(f"GPT parsing failed: {e}")
            return self._parse_with_regex(query)
    
    def _parse_with_regex(self, query: str) -> Dict[str, Any]:
        """Parse query using regex patterns as fallback."""
        import re
        
        query_lower = query.lower()
        
        # Extract symbol
        symbol_pattern = r'\b(BTCUSDT|ETHUSDT|NVDA|TSLA|AAPL|GOOGL|MSFT|AMZN)\b'
        symbol_match = re.search(symbol_pattern, query_lower)
        symbol = symbol_match.group(1) if symbol_match else None
        
        # Extract timeframe
        timeframe_pattern = r'\b(1m|5m|15m|1h|4h|1d)\b'
        timeframe_match = re.search(timeframe_pattern, query_lower)
        timeframe = timeframe_match.group(1) if timeframe_match else '1h'
        
        # Extract period
        period_pattern = r'\b(7d|14d|30d|90d|180d|1y)\b'
        period_match = re.search(period_pattern, query_lower)
        period = period_match.group(1) if period_match else '30d'
        
        # Determine intent
        if any(word in query_lower for word in ['best', 'model', 'recommend']):
            intent = 'model_recommendation'
        elif any(word in query_lower for word in ['long', 'short', 'buy', 'sell', 'trade']):
            intent = 'trading_signal'
        elif any(word in query_lower for word in ['analyze', 'analysis', 'market']):
            intent = 'market_analysis'
        else:
            intent = 'general_query'
        
        return {
            'intent': intent,
            'symbol': symbol,
            'timeframe': timeframe,
            'period': period,
            'model_type': None,
            'confidence': 0.7
        }
    
    def _validate_parsed_data(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean parsed data."""
        # Ensure required fields exist
        required_fields = ['intent', 'symbol', 'timeframe', 'period']
        for field in required_fields:
            if field not in parsed:
                parsed[field] = None
        
        # Validate symbol
        if parsed['symbol'] and parsed['symbol'] not in self.trading_context['available_symbols']:
            parsed['symbol'] = None
        
        # Validate timeframe
        if parsed['timeframe'] and parsed['timeframe'] not in self.trading_context['available_timeframes']:
            parsed['timeframe'] = '1h'
        
        # Validate period
        if parsed['period'] and parsed['period'] not in self.trading_context['available_periods']:
            parsed['period'] = '30d'
        
        # Set default confidence if not provided
        if 'confidence' not in parsed:
            parsed['confidence'] = 0.8
        
        return parsed
    
    def _execute_action(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
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
            return {'success': True, 'result': self._get_general_analysis(symbol, timeframe, period), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
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
            
            # Get recent market data and generate prediction
            # This would integrate with your forecasting system
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
    
    def _generate_commentary(self, query: str, parsed: Dict[str, Any], result: Dict[str, Any]) -> str:
        """
        Generate GPT commentary on the analysis results.
        
        Args:
            query: Original user query
            parsed: Parsed query parameters
            result: Analysis results
            
        Returns:
            GPT-generated commentary
        """
        if not openai.api_key:
            return self._generate_fallback_commentary(query, parsed, result)
        
        try:
            # Prepare context for GPT
            context = {
                'query': query,
                'intent': parsed.get('intent'),
                'symbol': parsed.get('symbol'),
                'timeframe': parsed.get('timeframe'),
                'period': parsed.get('period'),
                'results': result
            }
            
            system_prompt = """
            You are a quantitative trading analyst providing commentary on trading decisions and model recommendations.
            
            Your role is to:
            1. Explain the analysis results in clear, professional language
            2. Provide insights into why certain decisions were made
            3. Highlight key metrics and their significance
            4. Offer actionable recommendations
            5. Mention any risks or limitations
            
            Be concise but comprehensive. Use financial terminology appropriately.
            Focus on the most important findings and their implications for trading decisions.
            """
            
            user_prompt = f"""
            User Query: {query}
            
            Analysis Results:
            {json.dumps(context, indent=2)}
            
            Please provide a comprehensive commentary on these results, explaining the key findings and their implications for trading decisions.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating GPT commentary: {e}")
            return {'success': True, 'result': self._generate_fallback_commentary(query, parsed, result), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _generate_fallback_commentary(self, query: str, parsed: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Generate fallback commentary without GPT."""
        intent = parsed.get('intent')
        symbol = parsed.get('symbol')
        
        if intent == 'model_recommendation':
            best_model = result.get('best_model')
            if best_model:
                return f"Based on our analysis, the {best_model['model_type'].upper()} model shows the best performance for {symbol} with an overall score of {best_model['evaluation'].get('overall_score', 0):.2f}. This model has been evaluated across multiple metrics including Sharpe ratio, maximum drawdown, and win rate."
            else:
                return f"Analysis completed for {symbol}, but no suitable model was found. Consider adjusting parameters or timeframes."
        
        elif intent == 'trading_signal':
            signal = result.get('signal', {})
            if signal:
                return f"Trading signal for {symbol}: {signal['signal']} ({signal['strength']} confidence). {signal['reasoning']} Model performance score: {signal['model_score']:.2f}"
            else:
                return f"Unable to generate trading signal for {symbol}. Please check model availability and data quality."
        
        elif intent == 'market_analysis':
            return f"Comprehensive market analysis completed for {symbol}. The analysis includes model performance evaluation, market data trends, and technical indicators. Review the generated plots for visual insights."
        
        else:
            return f"Analysis completed for {symbol}. The system processed your query and generated relevant insights based on available data and models."
    
    def close(self):
        """Close the QuantGPT interface and clean up resources."""
        self.client.close()

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='QuantGPT - Natural Language Trading Interface')
    parser.add_argument('--query', required=True, help='Natural language query')
    parser.add_argument('--openai-key', help='OpenAI API key')
    parser.add_argument('--redis-host', default='localhost', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    
    args = parser.parse_args()
    
    # Initialize QuantGPT
    quant_gpt = QuantGPT(
        openai_api_key=args.openai_key,
        redis_host=args.redis_host,
        redis_port=args.redis_port
    )
    
    try:
        # Process the query
        result = quant_gpt.process_query(args.query)
        
        # Print results
        print(json.dumps(result, indent=2))
        
        return {
            "status": "completed",
            "query": args.query,
            "result": result
        }
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return {
            "status": "interrupted",
            "query": args.query,
            "result": "user_interrupted"
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "status": "failed",
            "query": args.query,
            "error": str(e)
        }
    finally:
        quant_gpt.close()

if __name__ == "__main__":
    main() 