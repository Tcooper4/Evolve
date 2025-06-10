import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from collections import deque
import re
import os
import random

@dataclass
class PromptContext:
    history: deque
    max_history: int = 10
    current_intent: Optional[str] = None
    entities: Dict[str, Any] = None
    confidence: float = 0.0

class PromptProcessor:
    def __init__(self):
        self.intent_patterns = {
            'forecast': r'(forecast|predict|outlook|projection)',
            'analyze': r'(analyze|analysis|examine|study)',
            'recommend': r'(recommend|suggest|advise|propose)',
            'explain': r'(explain|describe|detail|elaborate)'
        }
        self.entity_patterns = {
            'timeframe': r'(daily|weekly|monthly|quarterly|yearly|\d+\s*(day|week|month|year)s?)',
            'metric': r'(price|volume|return|volatility|trend)',
            'asset': r'(stock|bond|commodity|crypto|forex)',
            'action': r'(buy|sell|hold|trade|invest)'
        }
        
    def extract_intent(self, text: str) -> str:
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, text.lower()):
                return intent
        return 'unknown'
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        entities = {}
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text.lower())
            entities[entity_type] = [m.group() for m in matches]
        return entities

class LLMInterface:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the LLM interface."""
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.is_configured = False
        
        # Try to import openai, but don't fail if not available
        try:
            import openai
            if self.api_key:
                openai.api_key = self.api_key
                self.is_configured = True
                self.logger.info("OpenAI API configured successfully")
        except ImportError:
            self.logger.warning("OpenAI package not installed. LLM features will be disabled.")
        
        # Initialize prompt processor
        self.prompt_processor = PromptProcessor()
        self.context = PromptContext(history=deque(maxlen=10))
        
        # System prompt that defines the AI's capabilities
        self.system_prompt = """You are an advanced trading AI assistant. You can:
1. Analyze market data and provide insights
2. Generate trading strategies based on market conditions
3. Assess portfolio risk and provide recommendations
4. Explain technical indicators and their implications
5. Provide backtesting analysis and results
6. Optimize portfolio allocation
7. Monitor and analyze trading performance

When responding:
- Be concise but informative
- Include relevant data and metrics when available
- Suggest specific actions when appropriate
- Explain your reasoning
- Consider risk management in all recommendations
- Adapt to the user's experience level
- Use technical terms appropriately but explain them when needed"""
        
        # Initialize prompts dictionary
        self.prompts = {
            "market_analysis": "Analyze the following market data:\nPrice: {price}\nChange: {change}\nVolume: {volume}\nIndicators: {indicators}",
            "trading_strategy": "Generate a trading strategy based on the following market conditions:\n{conditions}",
            "risk_analysis": "Analyze the risk profile of the following portfolio:\n{portfolio_data}"
        }
        
        self.logger.info("LLMInterface initialized successfully")
    
    def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """Process a natural language prompt and determine the appropriate action."""
        try:
            if not self.is_configured:
                self.logger.info("Using mock response as API is not configured")
                return self._get_mock_response(prompt)
            
            self.logger.info(f"Processing prompt: {prompt}")
            
            # Process the prompt to extract intent and entities
            processed = self.prompt_processor.extract_intent(prompt)
            entities = self.prompt_processor.extract_entities(prompt)
            
            self.logger.debug(f"Extracted intent: {processed}")
            self.logger.debug(f"Extracted entities: {entities}")
            
            # Prepare the context-aware prompt
            context_prompt = self._prepare_context_prompt(prompt, processed)
            
            # Get response from OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context_prompt}
                ]
            )
            
            # Extract the response content
            content = response.choices[0].message.content
            self.logger.debug(f"Received response: {content}")
            
            # Determine the type of response and any required actions
            response_type = self._determine_response_type(content, processed)
            actions = self._generate_actions(content, response_type)
            
            self.logger.info(f"Generated {len(actions)} actions for response type: {response_type}")
            
            # Update context
            self.context.history.append({
                'prompt': prompt,
                'response': content,
                'type': response_type,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'content': content,
                'type': response_type,
                'actions': actions,
                'confidence': self._calculate_confidence(processed, entities)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing prompt: {str(e)}", exc_info=True)
            return {
                'error': f"Error processing prompt: {str(e)}",
                'type': 'error',
                'actions': []
            }
    
    def _determine_response_type(self, content: str, processed: Dict) -> str:
        """Determine the type of response based on content and processed intent."""
        if 'analyze' in processed.get('intent', '').lower():
            return 'analysis'
        elif 'strategy' in processed.get('intent', '').lower():
            return 'strategy'
        elif 'risk' in processed.get('intent', '').lower():
            return 'risk'
        elif 'explain' in processed.get('intent', '').lower():
            return 'explanation'
        else:
            return 'general'
    
    def _generate_actions(self, content: str, response_type: str) -> List[Dict]:
        """Generate actions based on the response type and content."""
        actions = []
        
        # Extract entities and parameters from content
        entities = self.prompt_processor.extract_entities(content)
        
        if response_type == 'analysis':
            # Market analysis actions
            actions.append({
                'type': 'update_chart',
                'chart_type': 'market_analysis',
                'data': self._extract_chart_data(content)
            })
            actions.append({
                'type': 'update_metrics',
                'metrics': {
                    'market': self._extract_metrics(content)
                }
            })
            actions.append({
                'type': 'show_analysis',
                'analysis_type': 'market',
                'data': self._extract_analysis_data(content)
            })
            
        elif response_type == 'strategy':
            # Trading strategy actions
            actions.append({
                'type': 'configure_strategy',
                'strategy_params': self._extract_strategy_params(content)
            })
            actions.append({
                'type': 'update_metrics',
                'metrics': {
                    'strategy': self._extract_metrics(content)
                }
            })
            actions.append({
                'type': 'run_backtest',
                'backtest_params': self._extract_backtest_params(content)
            })
            
        elif response_type == 'risk':
            # Risk analysis actions
            actions.append({
                'type': 'update_chart',
                'chart_type': 'risk',
                'data': self._extract_risk_data(content)
            })
            actions.append({
                'type': 'update_metrics',
                'metrics': {
                    'risk': self._extract_risk_metrics(content)
                }
            })
            actions.append({
                'type': 'show_analysis',
                'analysis_type': 'risk',
                'data': self._extract_risk_analysis(content)
            })
            
        elif response_type == 'portfolio':
            # Portfolio management actions
            actions.append({
                'type': 'update_portfolio',
                'allocation': self._extract_allocation(content)
            })
            actions.append({
                'type': 'optimize_portfolio',
                'optimization_params': self._extract_optimization_params(content)
            })
            actions.append({
                'type': 'update_chart',
                'chart_type': 'portfolio',
                'data': self._extract_portfolio_data(content)
            })
            
        elif response_type == 'ml':
            # ML model actions
            actions.append({
                'type': 'update_models',
                'model_params': self._extract_model_params(content)
            })
            actions.append({
                'type': 'generate_features',
                'feature_params': self._extract_feature_params(content)
            })
            actions.append({
                'type': 'show_analysis',
                'analysis_type': 'ml',
                'data': self._extract_ml_analysis(content)
            })
        
        return actions
    
    def _extract_strategy_params(self, content: str) -> Dict:
        """Extract strategy parameters from content."""
        try:
            # Extract strategy type, timeframe, and other parameters
            params = {
                'type': 'momentum',  # Default strategy type
                'timeframe': '1d',   # Default timeframe
                'parameters': {}
            }
            
            # Look for strategy type indicators
            if 'momentum' in content.lower():
                params['type'] = 'momentum'
            elif 'mean reversion' in content.lower():
                params['type'] = 'mean_reversion'
            elif 'trend following' in content.lower():
                params['type'] = 'trend_following'
            
            # Look for timeframe indicators
            timeframe_patterns = {
                '1m': r'1\s*min|1\s*minute',
                '5m': r'5\s*min|5\s*minutes',
                '15m': r'15\s*min|15\s*minutes',
                '1h': r'1\s*hour|1\s*h',
                '4h': r'4\s*hours|4\s*h',
                '1d': r'1\s*day|1\s*d|daily'
            }
            
            for tf, pattern in timeframe_patterns.items():
                if re.search(pattern, content.lower()):
                    params['timeframe'] = tf
                    break
            
            return params
        except Exception as e:
            self.logger.error(f"Error extracting strategy parameters: {str(e)}")
            return {}
    
    def _extract_backtest_params(self, content: str) -> Dict:
        """Extract backtest parameters from content."""
        try:
            params = {
                'start_date': None,
                'end_date': None,
                'assets': [],
                'initial_capital': 100000
            }
            
            # Extract date ranges
            date_pattern = r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}'
            dates = re.findall(date_pattern, content)
            if len(dates) >= 2:
                params['start_date'] = dates[0]
                params['end_date'] = dates[1]
            
            # Extract assets
            asset_pattern = r'\b[A-Z]{1,5}\b'  # Simple pattern for stock symbols
            params['assets'] = re.findall(asset_pattern, content)
            
            # Extract initial capital if mentioned
            capital_pattern = r'\$?(\d+(?:,\d+)*(?:\.\d+)?)'
            capital_match = re.search(capital_pattern, content)
            if capital_match:
                params['initial_capital'] = float(capital_match.group(1).replace(',', ''))
            
            return params
        except Exception as e:
            self.logger.error(f"Error extracting backtest parameters: {str(e)}")
            return {}
    
    def _extract_optimization_params(self, content: str) -> Dict:
        """Extract optimization parameters from content."""
        try:
            params = {
                'objective': 'min_risk',  # Default objective
                'constraints': {},
                'assets': []
            }
            
            # Determine optimization objective
            if 'maximize' in content.lower() or 'max' in content.lower():
                if 'return' in content.lower():
                    params['objective'] = 'max_return'
                elif 'sharpe' in content.lower():
                    params['objective'] = 'max_sharpe'
            
            # Extract constraints
            if 'risk' in content.lower():
                params['constraints']['max_risk'] = 0.2  # Default 20% max risk
            
            # Extract assets
            asset_pattern = r'\b[A-Z]{1,5}\b'
            params['assets'] = re.findall(asset_pattern, content)
            
            return params
        except Exception as e:
            self.logger.error(f"Error extracting optimization parameters: {str(e)}")
            return {}
    
    def _extract_model_params(self, content: str) -> Dict:
        """Extract model parameters from content."""
        try:
            params = {
                'model_type': 'lstm',  # Default model type
                'features': [],
                'target': 'price',
                'parameters': {}
            }
            
            # Determine model type
            if 'lstm' in content.lower():
                params['model_type'] = 'lstm'
            elif 'random forest' in content.lower():
                params['model_type'] = 'random_forest'
            elif 'xgboost' in content.lower():
                params['model_type'] = 'xgboost'
            
            # Extract features
            feature_patterns = {
                'price': r'price|close',
                'volume': r'volume',
                'rsi': r'rsi',
                'macd': r'macd',
                'bollinger': r'bollinger|bb',
                'stochastic': r'stochastic',
                'atr': r'atr|average true range'
            }
            
            for feature, pattern in feature_patterns.items():
                if re.search(pattern, content.lower()):
                    params['features'].append(feature)
            
            return params
        except Exception as e:
            self.logger.error(f"Error extracting model parameters: {str(e)}")
            return {}
    
    def _extract_feature_params(self, content: str) -> Dict:
        """Extract feature parameters from content."""
        try:
            params = {
                'features': [],
                'timeframes': ['1d'],
                'indicators': []
            }
            
            # Extract technical indicators
            indicator_patterns = {
                'rsi': r'rsi|relative strength',
                'macd': r'macd',
                'bollinger': r'bollinger|bb',
                'stochastic': r'stochastic',
                'atr': r'atr|average true range'
            }
            
            for indicator, pattern in indicator_patterns.items():
                if re.search(pattern, content.lower()):
                    params['indicators'].append(indicator)
            
            # Extract timeframes
            timeframe_patterns = {
                '1m': r'1\s*min|1\s*minute',
                '5m': r'5\s*min|5\s*minutes',
                '15m': r'15\s*min|15\s*minutes',
                '1h': r'1\s*hour|1\s*h',
                '4h': r'4\s*hours|4\s*h',
                '1d': r'1\s*day|1\s*d|daily'
            }
            
            for tf, pattern in timeframe_patterns.items():
                if re.search(pattern, content.lower()):
                    params['timeframes'].append(tf)
            
            return params
        except Exception as e:
            self.logger.error(f"Error extracting feature parameters: {str(e)}")
            return {}
    
    def _extract_ml_analysis(self, content: str) -> Dict:
        """Extract ML analysis data from content."""
        # This would be implemented to parse the content and extract ML analysis data
        return {}
    
    def _extract_allocation(self, content: str) -> Dict:
        """Extract portfolio allocation from content."""
        # This would be implemented to parse the content and extract portfolio allocation
        return {}
    
    def _extract_portfolio_data(self, content: str) -> Dict:
        """Extract portfolio data from content."""
        # This would be implemented to parse the content and extract portfolio data
        return {}
    
    def _extract_analysis_data(self, content: str) -> Dict:
        """Extract analysis data from content."""
        # This would be implemented to parse the content and extract analysis data
        return {}
    
    def _extract_risk_metrics(self, content: str) -> Dict:
        """Extract risk metrics from content."""
        # This would be implemented to parse the content and extract risk metrics
        return {}
    
    def _extract_risk_analysis(self, content: str) -> Dict:
        """Extract risk analysis data from content."""
        # This would be implemented to parse the content and extract risk analysis data
        return {}
    
    def _extract_risk_data(self, content: str) -> Dict:
        """Extract risk data from the response content."""
        try:
            risk_data = {
                'var': None,
                'beta': None,
                'sharpe': None,
                'volatility': None,
                'drawdown': None
            }
            
            # Extract Value at Risk (VaR)
            var_pattern = r'VaR[:\s]+(\d+\.?\d*)%'
            var_match = re.search(var_pattern, content)
            if var_match:
                risk_data['var'] = float(var_match.group(1))
            
            # Extract Beta
            beta_pattern = r'beta[:\s]+(\d+\.?\d*)'
            beta_match = re.search(beta_pattern, content.lower())
            if beta_match:
                risk_data['beta'] = float(beta_match.group(1))
            
            # Extract Sharpe Ratio
            sharpe_pattern = r'sharpe[:\s]+(\d+\.?\d*)'
            sharpe_match = re.search(sharpe_pattern, content.lower())
            if sharpe_match:
                risk_data['sharpe'] = float(sharpe_match.group(1))
            
            # Extract Volatility
            vol_pattern = r'volatility[:\s]+(\d+\.?\d*)%'
            vol_match = re.search(vol_pattern, content.lower())
            if vol_match:
                risk_data['volatility'] = float(vol_match.group(1))
            
            # Extract Maximum Drawdown
            drawdown_pattern = r'drawdown[:\s]+(\d+\.?\d*)%'
            drawdown_match = re.search(drawdown_pattern, content.lower())
            if drawdown_match:
                risk_data['drawdown'] = float(drawdown_match.group(1))
            
            return risk_data
        except Exception as e:
            self.logger.error(f"Error extracting risk data: {str(e)}")
            return {}
    
    def _get_mock_response(self, prompt: str) -> Dict[str, Any]:
        """Generate a mock response when API is not available."""
        mock_responses = {
            'analysis': "Based on the current market data, I observe a bullish trend with increasing volume. The RSI indicates overbought conditions, suggesting a potential pullback. Consider taking profits on long positions.",
            'strategy': "Given the current market conditions, I recommend a mean reversion strategy with tight stop losses. Look for opportunities to enter long positions on pullbacks to the 20-day moving average.",
            'risk': "Your portfolio shows moderate risk exposure. The current beta is 1.2, and the Sharpe ratio is 1.8. Consider reducing position sizes in high-volatility assets to maintain risk targets.",
            'explanation': "The RSI (Relative Strength Index) is a momentum oscillator that measures the speed and change of price movements. It ranges from 0 to 100, with readings above 70 indicating overbought conditions and below 30 indicating oversold conditions.",
            'general': "I understand you're interested in trading. Could you please provide more specific information about what you'd like to know? I can help with market analysis, strategy development, risk assessment, or portfolio optimization."
        }
        
        # Determine the most appropriate mock response
        intent = self.prompt_processor.extract_intent(prompt)
        response_type = self._determine_response_type("", {'intent': intent})
        
        return {
            'content': mock_responses.get(response_type, mock_responses['general']),
            'type': response_type,
            'actions': [],
            'confidence': 0.5
        }
    
    def _extract_chart_data(self, content: str) -> Dict:
        """Extract chart data from the response content."""
        # This would be implemented to parse the content and extract relevant data
        return {}
    
    def _extract_metrics(self, content: str) -> Dict:
        """Extract metrics from the response content."""
        # This would be implemented to parse the content and extract relevant metrics
        return {}

    def set_prompts(self, prompts: Dict[str, str]):
        """Set custom prompts."""
        self.prompts = prompts
    
    def _get_mock_analysis(self, data_type: str) -> str:
        """Generate mock analysis when API is not available."""
        mock_responses = {
            "market_analysis": [
                "Market shows strong bullish momentum with increasing volume.",
                "Technical indicators suggest a potential reversal.",
                "Market sentiment is mixed with conflicting signals.",
                "Current trend appears to be consolidating."
            ],
            "trading_strategy": [
                "Consider long positions with tight stop losses.",
                "Short-term mean reversion opportunities present.",
                "Range-bound trading recommended with clear support/resistance levels.",
                "Wait for clearer trend confirmation before entering positions."
            ],
            "risk_analysis": [
                "Portfolio shows moderate risk exposure with good diversification.",
                "High concentration in tech sector requires monitoring.",
                "Risk metrics indicate stable portfolio health.",
                "Consider reducing position sizes in volatile assets."
            ]
        }
        return random.choice(mock_responses.get(data_type, ["No analysis available."]))
    
    def analyze_market(self, market_data: Dict) -> Dict:
        """Analyze market data using LLM."""
        if not self.is_configured:
            return {
                'analysis': self._get_mock_analysis("market_analysis"),
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            import openai
            
            prompt = self.prompts["market_analysis"].format(
                price=market_data.get('price', 'N/A'),
                change=market_data.get('change', 'N/A'),
                volume=market_data.get('volume', 'N/A'),
                indicators=market_data.get('indicators', 'N/A')
            )
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional market analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return {
                'analysis': response.choices[0].message.content,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_trading_strategy(self, market_conditions: Dict) -> Dict:
        """Generate trading strategy based on market conditions."""
        if not self.is_configured:
            return {
                'strategy': self._get_mock_analysis("trading_strategy"),
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            import openai
            
            prompt = self.prompts["trading_strategy"].format(
                trend=market_conditions.get('trend', 'N/A'),
                volatility=market_conditions.get('volatility', 'N/A'),
                support=market_conditions.get('support', 'N/A'),
                resistance=market_conditions.get('resistance', 'N/A')
            )
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional trading strategist."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return {
                'strategy': response.choices[0].message.content,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def analyze_risk(self, portfolio_data: Dict) -> Dict:
        """Analyze portfolio risk using LLM."""
        if not self.is_configured:
            return {
                'analysis': self._get_mock_analysis("risk_analysis"),
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            import openai
            
            prompt = self.prompts["risk_analysis"].format(
                total_value=portfolio_data.get('total_value', 'N/A'),
                positions=portfolio_data.get('positions', 'N/A'),
                exposure=portfolio_data.get('exposure', 'N/A'),
                risk_metrics=portfolio_data.get('risk_metrics', 'N/A')
            )
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional risk analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return {
                'analysis': response.choices[0].message.content,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_llm_metrics(self) -> Dict[str, str]:
        """Get LLM interface metrics."""
        return {
            'api_key_configured': 'Yes' if self.is_configured else 'No',
            'model': 'gpt-3.5-turbo' if self.is_configured else 'Mock Data',
            'status': 'Active' if self.is_configured else 'Using Mock Data'
        }

    def _calculate_confidence(self, intent: str, entities: Dict) -> float:
        """Calculate confidence score for the processed prompt."""
        confidence = 0.0
        if intent != 'unknown':
            confidence += 0.4
        if entities:
            confidence += min(0.6, len(entities) * 0.1)
        return confidence

    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """Generate response with context awareness."""
        try:
            # Process prompt
            processed = self.process_prompt(prompt)
            
            # Check cache
            cache_key = f"{prompt}_{processed['intent']}"
            if cache_key in self.response_cache:
                return self.response_cache[cache_key]
            
            # Prepare context-aware prompt
            context_prompt = self._prepare_context_prompt(prompt, processed)
            
            # Generate response
            inputs = self.tokenizer(context_prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Cache response
            self.response_cache[cache_key] = response
            
            return response
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return ""

    def _prepare_context_prompt(self, prompt: str, processed: Dict) -> str:
        """Prepare context-aware prompt with relevant information."""
        context_parts = []
        
        # Add relevant knowledge base entries
        if processed['entities']:
            for entity_type, entities in processed['entities'].items():
                for entity in entities:
                    if entity in self.knowledge_base:
                        context_parts.append(self.knowledge_base[entity])
        
        # Add recent history context
        if self.context.history:
            recent_context = self.context.history[-1]
            context_parts.append(f"Previous context: {recent_context['prompt']}")
        
        # Combine context with current prompt
        context = " ".join(context_parts)
        return f"{context}\nCurrent prompt: {prompt}"

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment with confidence scoring."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
            outputs = self.sentiment_model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)[0]
            
            confidence = float(torch.max(scores))
            label = 'POSITIVE' if scores[1] > scores[0] else 'NEGATIVE'
            
            return {
                'label': label,
                'score': float(scores[1] if label == 'POSITIVE' else scores[0]),
                'confidence': confidence
            }
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return {'label': 'ERROR', 'score': 0.0, 'confidence': 0.0}

    def generate_market_analysis(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive market analysis with confidence scoring."""
        try:
            # Calculate statistics
            stats = {
                'mean': market_data.mean().to_dict(),
                'std': market_data.std().to_dict(),
                'trend': 'up' if market_data.iloc[-1].mean() > market_data.iloc[0].mean() else 'down',
                'volatility': market_data.std().mean() / market_data.mean().mean()
            }
            
            # Generate analysis
            analysis_prompt = f"""
            Market Analysis Report:
            Current trend: {stats['trend']}
            Average values: {stats['mean']}
            Volatility: {stats['std']}
            Market volatility: {stats['volatility']}
            
            Provide a detailed market analysis with confidence levels:
            """
            
            analysis = self.generate_response(analysis_prompt)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_analysis_confidence(market_data)
            
            return {
                'analysis': analysis,
                'statistics': stats,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error generating market analysis: {str(e)}")
            return {
                'analysis': "Error generating analysis",
                'statistics': {},
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }

    def _calculate_analysis_confidence(self, data: pd.DataFrame) -> float:
        """Calculate confidence score for market analysis."""
        confidence = 0.0
        
        # Data quality checks
        if not data.empty:
            confidence += 0.2
        if len(data) > 100:  # Sufficient data points
            confidence += 0.2
        if not data.isnull().any().any():  # No missing values
            confidence += 0.2
        if data.std().mean() > 0:  # Non-zero variance
            confidence += 0.2
        if len(data.columns) > 1:  # Multiple features
            confidence += 0.2
            
        return confidence

    def generate_trading_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals with confidence scoring."""
        try:
            signals = {}
            confidence_scores = {}
            
            for column in market_data.columns:
                if 'price' in column.lower() or 'close' in column.lower():
                    # Technical analysis
                    sma_20 = market_data[column].rolling(window=20).mean()
                    sma_50 = market_data[column].rolling(window=50).mean()
                    rsi = self._calculate_rsi(market_data[column])
                    
                    # Generate signal
                    signal, confidence = self._generate_signal(
                        sma_20.iloc[-1],
                        sma_50.iloc[-1],
                        rsi.iloc[-1]
                    )
                    
                    signals[column] = signal
                    confidence_scores[column] = confidence
            
            return {
                'signals': signals,
                'confidence_scores': confidence_scores,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {str(e)}")
            return {
                'signals': {},
                'confidence_scores': {},
                'timestamp': datetime.now().isoformat()
            }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _generate_signal(self, sma_20: float, sma_50: float, rsi: float) -> tuple:
        """Generate trading signal with confidence score."""
        signal = 'HOLD'
        confidence = 0.5  # Base confidence
        
        # Trend analysis
        if sma_20 > sma_50:
            signal = 'BUY'
            confidence += 0.2
        elif sma_20 < sma_50:
            signal = 'SELL'
            confidence += 0.2
            
        # RSI analysis
        if rsi > 70:
            if signal == 'BUY':
                confidence -= 0.1
        elif rsi < 30:
            if signal == 'SELL':
                confidence -= 0.1
                
        return signal, min(max(confidence, 0.0), 1.0) 