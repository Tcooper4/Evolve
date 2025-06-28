"""
Unified AgentHub for routing and managing all trading system agents.

This module provides a centralized interface for all agent interactions,
including PromptAgent, ForecastRouter, LLMHandler, and QuantGPTAgent.
"""

import logging
import streamlit as st
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)


class AgentHub:
    """Centralized hub for managing and routing all trading system agents."""
    
    def __init__(self):
        """Initialize the AgentHub with all available agents."""
        self.agents = {}
        self.routing_rules = {}
        self.fallback_agent = None
        self._initialize_agents()
        self._setup_routing_rules()
        
    def _initialize_agents(self):
        """Initialize all available agents."""
        try:
            # Initialize PromptAgent
            from trading.meta_agents.agents.prompt_agent import PromptAgent
            self.agents['prompt'] = PromptAgent()
            logger.info("PromptAgent initialized")
        except Exception as e:
            logger.warning(f"PromptAgent initialization failed: {e}")
            st.session_state["status"] = "fallback activated"
            
        try:
            # Initialize ForecastRouter
            from models.forecast_router import ForecastRouter
            self.agents['forecast'] = ForecastRouter()
            logger.info("ForecastRouter initialized")
        except Exception as e:
            logger.warning(f"ForecastRouter initialization failed: {e}")
            st.session_state["status"] = "fallback activated"
            
        try:
            # Initialize LLMHandler
            from trading.llm.llm_interface import LLMHandler
            self.agents['llm'] = LLMHandler()
            logger.info("LLMHandler initialized")
        except Exception as e:
            logger.warning(f"LLMHandler initialization failed: {e}")
            st.session_state["status"] = "fallback activated"
            
        try:
            # Initialize QuantGPTAgent
            from trading.services.quant_gpt import QuantGPTAgent
            self.agents['quant_gpt'] = QuantGPTAgent()
            logger.info("QuantGPTAgent initialized")
        except Exception as e:
            logger.warning(f"QuantGPTAgent initialization failed: {e}")
            st.session_state["status"] = "fallback activated"
            
        # Set fallback agent
        self.fallback_agent = self.agents.get('prompt') or self.agents.get('llm')
        
    def _setup_routing_rules(self):
        """Setup routing rules for different types of prompts."""
        self.routing_rules = {
            'forecast': [
                'forecast', 'predict', 'prediction', 'price', 'market', 'trend',
                'technical', 'analysis', 'chart', 'indicator'
            ],
            'trading': [
                'trade', 'buy', 'sell', 'position', 'portfolio', 'strategy',
                'signal', 'execution', 'order'
            ],
            'analysis': [
                'analyze', 'analysis', 'report', 'metrics', 'performance',
                'backtest', 'evaluation'
            ],
            'llm': [
                'explain', 'describe', 'what is', 'how to', 'why', 'when',
                'question', 'help', 'assist'
            ],
            'quant_gpt': [
                'quantitative', 'math', 'statistics', 'model', 'algorithm',
                'optimization', 'risk', 'probability'
            ]
        }
        
    def route(self, prompt: str) -> Dict[str, Any]:
        """
        Route a prompt to the appropriate agent.
        
        Args:
            prompt: User input prompt
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Determine the best agent for this prompt
            agent_type = self._determine_agent_type(prompt.lower())
            
            # Get the appropriate agent
            agent = self.agents.get(agent_type)
            
            if not agent:
                logger.warning(f"Agent type '{agent_type}' not available, using fallback")
                agent = self.fallback_agent
                st.session_state["status"] = "fallback activated"
                
            if not agent:
                return self._fallback_response(prompt)
                
            # Route to the agent
            logger.info(f"Routing to {agent_type} agent")
            response = self._call_agent(agent, agent_type, prompt)
            
            # Log the interaction
            self._log_interaction(prompt, agent_type, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in AgentHub routing: {e}")
            return self._fallback_response(prompt)
            
    def _determine_agent_type(self, prompt: str) -> str:
        """Determine the best agent type for a given prompt."""
        scores = {}
        
        for agent_type, keywords in self.routing_rules.items():
            score = sum(1 for keyword in keywords if keyword in prompt)
            if score > 0:
                scores[agent_type] = score
                
        if not scores:
            return 'llm'  # Default to LLM for general queries
            
        # Return the agent type with the highest score
        return max(scores, key=scores.get)
        
    def _call_agent(self, agent, agent_type: str, prompt: str) -> Dict[str, Any]:
        """Call the appropriate agent method."""
        try:
            if agent_type == 'forecast':
                return self._call_forecast_agent(agent, prompt)
            elif agent_type == 'trading':
                return self._call_trading_agent(agent, prompt)
            elif agent_type == 'analysis':
                return self._call_analysis_agent(agent, prompt)
            elif agent_type == 'quant_gpt':
                return self._call_quant_gpt_agent(agent, prompt)
            else:
                return self._call_llm_agent(agent, prompt)
                
        except Exception as e:
            logger.error(f"Error calling {agent_type} agent: {e}")
            return self._fallback_response(prompt)
            
    def _call_forecast_agent(self, agent, prompt: str) -> Dict[str, Any]:
        """Call the forecast agent."""
        try:
            # Extract ticker and timeframe from prompt
            ticker = self._extract_ticker(prompt)
            timeframe = self._extract_timeframe(prompt)
            
            if hasattr(agent, 'generate_forecast'):
                forecast = agent.generate_forecast(ticker, timeframe)
                return {
                    'type': 'forecast',
                    'content': forecast,
                    'agent': 'forecast',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return self._fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"Error in forecast agent: {e}")
            return self._fallback_response(prompt)
            
    def _call_trading_agent(self, agent, prompt: str) -> Dict[str, Any]:
        """Call the trading agent."""
        try:
            if hasattr(agent, 'process_trading_request'):
                response = agent.process_trading_request(prompt)
                return {
                    'type': 'trading',
                    'content': response,
                    'agent': 'trading',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return self._fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"Error in trading agent: {e}")
            return self._fallback_response(prompt)
            
    def _call_analysis_agent(self, agent, prompt: str) -> Dict[str, Any]:
        """Call the analysis agent."""
        try:
            if hasattr(agent, 'analyze'):
                analysis = agent.analyze(prompt)
                return {
                    'type': 'analysis',
                    'content': analysis,
                    'agent': 'analysis',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return self._fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"Error in analysis agent: {e}")
            return self._fallback_response(prompt)
            
    def _call_quant_gpt_agent(self, agent, prompt: str) -> Dict[str, Any]:
        """Call the QuantGPT agent."""
        try:
            if hasattr(agent, 'process_query'):
                response = agent.process_query(prompt)
                return {
                    'type': 'quantitative',
                    'content': response,
                    'agent': 'quant_gpt',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return self._fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"Error in QuantGPT agent: {e}")
            return self._fallback_response(prompt)
            
    def _call_llm_agent(self, agent, prompt: str) -> Dict[str, Any]:
        """Call the LLM agent."""
        try:
            if hasattr(agent, 'generate_response'):
                response = agent.generate_response(prompt)
                return {
                    'type': 'llm',
                    'content': response,
                    'agent': 'llm',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return self._fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"Error in LLM agent: {e}")
            return self._fallback_response(prompt)
            
    def _extract_ticker(self, prompt: str) -> str:
        """Extract ticker symbol from prompt."""
        # Simple ticker extraction - can be enhanced
        import re
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        matches = re.findall(ticker_pattern, prompt)
        
        # Common tickers to look for
        common_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META']
        
        for match in matches:
            if match in common_tickers:
                return match
                
        return 'AAPL'  # Default ticker
        
    def _extract_timeframe(self, prompt: str) -> str:
        """Extract timeframe from prompt."""
        timeframes = {
            '1d': ['daily', 'day', '1d', '1 day'],
            '1w': ['weekly', 'week', '1w', '1 week'],
            '1m': ['monthly', 'month', '1m', '1 month'],
            '1h': ['hourly', 'hour', '1h', '1 hour'],
            '15m': ['15 minute', '15m', '15min']
        }
        
        prompt_lower = prompt.lower()
        for timeframe, keywords in timeframes.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return timeframe
                
        return '1d'  # Default timeframe
        
    def _fallback_response(self, prompt: str) -> Dict[str, Any]:
        """Generate a fallback response when agents are unavailable."""
        logger.warning(f"Using fallback response for: {prompt}")
        st.session_state["status"] = "fallback activated"
        
        return {
            'type': 'fallback',
            'content': f"I understand your query: '{prompt}'. Please use the navigation menu to access specific features:\n"
                      f"- Use 'Forecast & Trade' for market predictions\n"
                      f"- Use 'Portfolio Dashboard' for position management\n"
                      f"- Use 'Performance Tracker' for historical analysis",
            'agent': 'fallback',
            'timestamp': datetime.now().isoformat()
        }
        
    def _log_interaction(self, prompt: str, agent_type: str, response: Dict[str, Any]):
        """Log the agent interaction."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'agent_type': agent_type,
            'response_type': response.get('type', 'unknown'),
            'success': response.get('type') != 'fallback'
        }
        
        # Log to session state for UI display
        if 'agent_interactions' not in st.session_state:
            st.session_state['agent_interactions'] = []
            
        st.session_state['agent_interactions'].append(log_entry)
        
        # Keep only last 10 interactions
        if len(st.session_state['agent_interactions']) > 10:
            st.session_state['agent_interactions'] = st.session_state['agent_interactions'][-10:]
            
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        status = {}
        for agent_type, agent in self.agents.items():
            status[agent_type] = {
                'available': agent is not None,
                'type': type(agent).__name__ if agent else None
            }
        return status
        
    def get_recent_interactions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent agent interactions."""
        return st.session_state.get('agent_interactions', [])[-limit:]
