"""LLM Agent for Trading Decisions."""

import logging
import warnings
warnings.filterwarnings('ignore')

# Try to import LLM libraries with fallbacks
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pipeline = AutoTokenizer = AutoModelForCausalLM = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Import from trading package
try:
    from trading.utils.common import get_logger
    from trading.config.configuration import TradingConfig
    from trading.memory.agent_memory import AgentMemory
    from trading.market.market_data import MarketData
    from trading.evaluation.metrics import calculate_metrics
except ImportError:
    # Fallback imports
    def get_logger(name):
        return logging.getLogger(name)
    
    class TradingConfig:
        def __init__(self):
            self.llm_model = "gpt-3.5-turbo"
            self.llm_api_key = None
    
    class AgentMemory:
        def __init__(self):
            self.memory = []
    
    class MarketData:
        def __init__(self):
            self.data = {}
    
    def calculate_metrics(returns):
        return {'sharpe_ratio': 0.0, 'total_return': 0.0}

logger = get_logger(__name__)

class LLMAgent:
    """LLM-powered trading agent."""
    
    def __init__(self, config: TradingConfig = None):
        """Initialize LLM agent."""
        self.config = config or TradingConfig()
        self.memory = AgentMemory()
        self.market_data = MarketData()
        self.performance_history = []
        
        # Initialize LLM components
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM components."""
        if OPENAI_AVAILABLE and self.config.llm_api_key:
            openai.api_key = self.config.llm_api_key
            self.llm_type = "openai"
        elif TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            self.llm_type = "local"
            self._load_local_model()
        else:
            self.llm_type = "dummy"
            logger.warning("No LLM available, using dummy responses")
    
    def _load_local_model(self):
        """Load local transformer model."""
        try:
            model_name = "microsoft/DialoGPT-medium"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            self.llm_type = "dummy"
    
    def analyze_market(self, market_data: dict) -> dict:
        """Analyze market data using LLM."""
        try:
            # Prepare market context
            context = self._prepare_market_context(market_data)
            
            # Generate analysis
            if self.llm_type == "openai":
                analysis = self._openai_analysis(context)
            elif self.llm_type == "local":
                analysis = self._local_analysis(context)
            else:
                analysis = self._dummy_analysis(context)
            
            # Store in memory
            self.memory.memory.append({
                'timestamp': 'now',
                'type': 'market_analysis',
                'data': analysis
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return self._dummy_analysis({})
    
    def generate_trading_signal(self, market_data: dict, 
                               portfolio_state: dict) -> dict:
        """Generate trading signal using LLM."""
        try:
            # Prepare trading context
            context = self._prepare_trading_context(market_data, portfolio_state)
            
            # Generate signal
            if self.llm_type == "openai":
                signal = self._openai_signal(context)
            elif self.llm_type == "local":
                signal = self._local_signal(context)
            else:
                signal = self._dummy_signal(context)
            
            # Validate signal
            signal = self._validate_signal(signal)
            
            # Store in memory
            self.memory.memory.append({
                'timestamp': 'now',
                'type': 'trading_signal',
                'data': signal
            })
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return self._dummy_signal({})
    
    def _prepare_market_context(self, market_data: dict) -> str:
        """Prepare market context for LLM."""
        context = f"""
        Market Data Analysis:
        - Symbol: {market_data.get('symbol', 'Unknown')}
        - Current Price: ${market_data.get('price', 0):.2f}
        - Volume: {market_data.get('volume', 0):,}
        - 24h Change: {market_data.get('change_24h', 0):.2f}%
        - Market Cap: ${market_data.get('market_cap', 0):,.0f}
        
        Technical Indicators:
        - RSI: {market_data.get('rsi', 0):.2f}
        - MACD: {market_data.get('macd', 0):.4f}
        - Moving Average: ${market_data.get('ma_20', 0):.2f}
        - Bollinger Bands: Upper=${market_data.get('bb_upper', 0):.2f}, Lower=${market_data.get('bb_lower', 0):.2f}
        
        Market Sentiment:
        - Fear & Greed Index: {market_data.get('fear_greed', 0)}
        - News Sentiment: {market_data.get('news_sentiment', 'neutral')}
        """
        return context
    
    def _prepare_trading_context(self, market_data: dict, 
                                portfolio_state: dict) -> str:
        """Prepare trading context for LLM."""
        context = f"""
        Trading Context:
        
        Market Data:
        - Symbol: {market_data.get('symbol', 'Unknown')}
        - Current Price: ${market_data.get('price', 0):.2f}
        - Price Change: {market_data.get('price_change', 0):.2f}%
        
        Portfolio State:
        - Current Position: {portfolio_state.get('position', 0)} shares
        - Available Cash: ${portfolio_state.get('cash', 0):,.2f}
        - Total Value: ${portfolio_state.get('total_value', 0):,.2f}
        - P&L: ${portfolio_state.get('pnl', 0):+,.2f}
        
        Risk Parameters:
        - Max Position Size: {portfolio_state.get('max_position', 0)}%
        - Stop Loss: {portfolio_state.get('stop_loss', 0)}%
        - Take Profit: {portfolio_state.get('take_profit', 0)}%
        
        Recent Performance:
        - Win Rate: {portfolio_state.get('win_rate', 0):.1f}%
        - Avg Return: {portfolio_state.get('avg_return', 0):.2f}%
        - Max Drawdown: {portfolio_state.get('max_drawdown', 0):.2f}%
        """
        return context
    
    def _openai_analysis(self, context: str) -> dict:
        """Generate analysis using OpenAI."""
        try:
            response = openai.ChatCompletion.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": "You are a professional market analyst. Provide concise, actionable market analysis."},
                    {"role": "user", "content": f"Analyze this market data and provide insights:\n{context}"}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content
            
            return {
                'sentiment': self._extract_sentiment(analysis_text),
                'confidence': self._extract_confidence(analysis_text),
                'key_factors': self._extract_factors(analysis_text),
                'recommendation': analysis_text
            }
            
        except Exception as e:
            logger.error(f"OpenAI analysis error: {e}")
            return self._dummy_analysis({})
    
    def _local_analysis(self, context: str) -> dict:
        """Generate analysis using local model."""
        try:
            inputs = self.tokenizer.encode(context, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            analysis_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                'sentiment': 'neutral',
                'confidence': 0.6,
                'key_factors': ['technical_indicators', 'market_sentiment'],
                'recommendation': analysis_text
            }
            
        except Exception as e:
            logger.error(f"Local analysis error: {e}")
            return self._dummy_analysis({})
    
    def _openai_signal(self, context: str) -> dict:
        """Generate trading signal using OpenAI."""
        try:
            response = openai.ChatCompletion.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": "You are a trading signal generator. Provide clear buy/sell/hold signals with confidence levels."},
                    {"role": "user", "content": f"Generate a trading signal based on this context:\n{context}"}
                ],
                max_tokens=200,
                temperature=0.2
            )
            
            signal_text = response.choices[0].message.content
            
            return {
                'action': self._extract_action(signal_text),
                'confidence': self._extract_confidence(signal_text),
                'reasoning': signal_text,
                'price_target': self._extract_price_target(signal_text),
                'stop_loss': self._extract_stop_loss(signal_text)
            }
            
        except Exception as e:
            logger.error(f"OpenAI signal error: {e}")
            return self._dummy_signal({})
    
    def _local_signal(self, context: str) -> dict:
        """Generate trading signal using local model."""
        try:
            inputs = self.tokenizer.encode(context, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.5,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            signal_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                'action': 'hold',
                'confidence': 0.5,
                'reasoning': signal_text,
                'price_target': None,
                'stop_loss': None
            }
            
        except Exception as e:
            logger.error(f"Local signal error: {e}")
            return self._dummy_signal({})
    
    def _dummy_analysis(self, context: dict) -> dict:
        """Generate dummy analysis."""
        return {
            'sentiment': 'neutral',
            'confidence': 0.5,
            'key_factors': ['market_volatility', 'technical_indicators'],
            'recommendation': 'Market conditions are neutral. Monitor key support/resistance levels.'
        }
    
    def _dummy_signal(self, context: dict) -> dict:
        """Generate dummy trading signal."""
        return {
            'action': 'hold',
            'confidence': 0.5,
            'reasoning': 'Insufficient data for confident signal generation.',
            'price_target': None,
            'stop_loss': None
        }
    
    def _extract_sentiment(self, text: str) -> str:
        """Extract sentiment from text."""
        text_lower = text.lower()
        if any(word in text_lower for word in ['bullish', 'positive', 'buy', 'up']):
            return 'bullish'
        elif any(word in text_lower for word in ['bearish', 'negative', 'sell', 'down']):
            return 'bearish'
        else:
            return 'neutral'
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence level from text."""
        # Simple confidence extraction
        if 'high confidence' in text.lower():
            return 0.8
        elif 'medium confidence' in text.lower():
            return 0.6
        elif 'low confidence' in text.lower():
            return 0.4
        else:
            return 0.5
    
    def _extract_factors(self, text: str) -> list:
        """Extract key factors from text."""
        factors = []
        if 'technical' in text.lower():
            factors.append('technical_indicators')
        if 'fundamental' in text.lower():
            factors.append('fundamental_analysis')
        if 'sentiment' in text.lower():
            factors.append('market_sentiment')
        if 'volatility' in text.lower():
            factors.append('volatility')
        
        return factors if factors else ['general_analysis']
    
    def _extract_action(self, text: str) -> str:
        """Extract trading action from text."""
        text_lower = text.lower()
        if 'buy' in text_lower:
            return 'buy'
        elif 'sell' in text_lower:
            return 'sell'
        else:
            return 'hold'
    
    def _extract_price_target(self, text: str) -> float:
        """Extract price target from text."""
        # Simple price target extraction
        import re
        price_match = re.search(r'\$(\d+\.?\d*)', text)
        if price_match:
            return float(price_match.group(1))
        return None
    
    def _extract_stop_loss(self, text: str) -> float:
        """Extract stop loss from text."""
        # Simple stop loss extraction
        import re
        stop_match = re.search(r'stop.*?\$(\d+\.?\d*)', text.lower())
        if stop_match:
            return float(stop_match.group(1))
        return None
    
    def _validate_signal(self, signal: dict) -> dict:
        """Validate trading signal."""
        # Ensure required fields
        required_fields = ['action', 'confidence', 'reasoning']
        for field in required_fields:
            if field not in signal:
                signal[field] = 'hold' if field == 'action' else 0.5 if field == 'confidence' else 'No reasoning provided'
        
        # Validate action
        if signal['action'] not in ['buy', 'sell', 'hold']:
            signal['action'] = 'hold'
        
        # Validate confidence
        if not isinstance(signal['confidence'], (int, float)) or signal['confidence'] < 0 or signal['confidence'] > 1:
            signal['confidence'] = 0.5
        
        return signal
    
    def update_performance(self, trade_result: dict):
        """Update agent performance."""
        self.performance_history.append(trade_result)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def get_performance_summary(self) -> dict:
        """Get performance summary."""
        if not self.performance_history:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'total_return': 0.0
            }
        
        wins = sum(1 for trade in self.performance_history if trade.get('pnl', 0) > 0)
        total_trades = len(self.performance_history)
        total_pnl = sum(trade.get('pnl', 0) for trade in self.performance_history)
        
        return {
            'total_trades': total_trades,
            'win_rate': wins / total_trades if total_trades > 0 else 0.0,
            'avg_return': total_pnl / total_trades if total_trades > 0 else 0.0,
            'total_return': total_pnl
        }

# Global LLM agent instance
llm_agent = LLMAgent()

def get_llm_agent() -> LLMAgent:
    """Get the global LLM agent instance."""
    return llm_agent 