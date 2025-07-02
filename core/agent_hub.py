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
        self.agents: Dict[str, Any] = {}
        self.routing_rules: Dict[str, List[str]] = {}
        self.fallback_agent: Optional[Any] = None
        self.interaction_history: List[Dict[str, Any]] = []
        self._initialize_agents()
        self._setup_routing_rules()

    def _initialize_agents(self) -> None:
        """Initialize all available agents with proper error handling."""
        try:
            # Initialize PromptAgent
            from trading.agents.prompt_router_agent import PromptRouterAgent
            self.agents['prompt'] = PromptRouterAgent()
            logger.info("PromptRouterAgent initialized")
        except ImportError as e:
            logger.warning(f"PromptRouterAgent not available: {e}")
        except Exception as e:
            logger.warning(f"PromptRouterAgent initialization failed: {e}")
            st.session_state["status"] = "fallback activated"
            
        try:
            # Initialize ForecastRouter
            from models.forecast_router import ForecastRouter
            self.agents['forecast'] = ForecastRouter()
            logger.info("ForecastRouter initialized")
        except ImportError as e:
            logger.warning(f"ForecastRouter not available: {e}")
        except Exception as e:
            logger.warning(f"ForecastRouter initialization failed: {e}")
            st.session_state["status"] = "fallback activated"
            
        try:
            # Initialize LLMHandler
            from trading.llm.llm_interface import LLMHandler
            self.agents['llm'] = LLMHandler()
            logger.info("LLMHandler initialized")
        except ImportError as e:
            logger.warning(f"LLMHandler not available: {e}")
        except Exception as e:
            logger.warning(f"LLMHandler initialization failed: {e}")
            st.session_state["status"] = "fallback activated"
            
        try:
            # Initialize QuantGPTAgent
            from trading.services.quant_gpt import QuantGPT
            self.agents['quant_gpt'] = QuantGPT()
            logger.info("QuantGPT initialized")
        except ImportError as e:
            logger.warning(f"QuantGPT not available: {e}")
        except Exception as e:
            logger.warning(f"QuantGPT initialization failed: {e}")
            st.session_state["status"] = "fallback activated"
            
        # Set fallback agent
        self.fallback_agent = self.agents.get('prompt') or self.agents.get('llm')
        if self.fallback_agent:
            logger.info(f"Fallback agent set to: {type(self.fallback_agent).__name__}")
        else:
            logger.warning("No fallback agent available")

    def _setup_routing_rules(self) -> None:
        """Setup routing rules for different types of prompts."""
        self.routing_rules = {
            'forecast': [
                'forecast', 'predict', 'prediction', 'price', 'market', 'trend',
                'technical', 'analysis', 'chart', 'indicator', 'future', 'outlook'
            ],
            'trading': [
                'trade', 'buy', 'sell', 'position', 'portfolio', 'strategy',
                'signal', 'execution', 'order', 'entry', 'exit', 'stop loss'
            ],
            'analysis': [
                'analyze', 'analysis', 'report', 'metrics', 'performance',
                'backtest', 'evaluation', 'review', 'assessment'
            ],
            'llm': [
                'explain', 'describe', 'what is', 'how to', 'why', 'when',
                'question', 'help', 'assist', 'tell me', 'show me'
            ],
            'quant_gpt': [
                'quantitative', 'math', 'statistics', 'model', 'algorithm',
                'optimization', 'risk', 'probability', 'correlation', 'regression'
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
        """
        Determine the best agent type for a given prompt.
        
        Args:
            prompt: Lowercase prompt text
            
        Returns:
            String indicating the best agent type
        """
        scores = {}
        
        for agent_type, keywords in self.routing_rules.items():
            score = sum(1 for keyword in keywords if keyword in prompt)
            if score > 0:
                scores[agent_type] = score
                
        if not scores:
            return 'llm'  # Default to LLM for general queries
            
        # Return the agent type with the highest score
        return max(scores, key=scores.get)
        
    def _call_agent(self, agent: Any, agent_type: str, prompt: str) -> Dict[str, Any]:
        """
        Call the appropriate agent method.
        
        Args:
            agent: The agent instance to call
            agent_type: Type of agent being called
            prompt: User prompt
            
        Returns:
            Dictionary containing agent response
        """
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
            
    def _call_forecast_agent(self, agent: Any, prompt: str) -> Dict[str, Any]:
        """
        Call the forecast agent.
        
        Args:
            agent: Forecast agent instance
            prompt: User prompt
            
        Returns:
            Dictionary containing forecast response
        """
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
                    'timestamp': datetime.now().isoformat(),
                    'confidence': 0.85,
                    'metadata': {
                        'ticker': ticker,
                        'timeframe': timeframe,
                        'model_used': getattr(agent, 'current_model', 'unknown'),
                        'strategy_applied': 'forecast_router',
                        'agent_confidence': 0.85
                    }
                }
            else:
                return self._fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"Error in forecast agent: {e}")
            return self._fallback_response(prompt)
            
    def _call_trading_agent(self, agent: Any, prompt: str) -> Dict[str, Any]:
        """
        Call the trading agent.
        
        Args:
            agent: Trading agent instance
            prompt: User prompt
            
        Returns:
            Dictionary containing trading response
        """
        try:
            if hasattr(agent, 'process_trading_request'):
                response = agent.process_trading_request(prompt)
                return {
                    'type': 'trading',
                    'content': response,
                    'agent': 'trading',
                    'timestamp': datetime.now().isoformat(),
                    'confidence': 0.80,
                    'metadata': {
                        'strategy_applied': 'trading_agent',
                        'agent_confidence': 0.80,
                        'execution_status': 'pending'
                    }
                }
            else:
                return self._fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"Error in trading agent: {e}")
            return self._fallback_response(prompt)
            
    def _call_analysis_agent(self, agent: Any, prompt: str) -> Dict[str, Any]:
        """
        Call the analysis agent.
        
        Args:
            agent: Analysis agent instance
            prompt: User prompt
            
        Returns:
            Dictionary containing analysis response
        """
        try:
            if hasattr(agent, 'analyze'):
                analysis = agent.analyze(prompt)
                return {
                    'type': 'analysis',
                    'content': analysis,
                    'agent': 'analysis',
                    'timestamp': datetime.now().isoformat(),
                    'confidence': 0.90,
                    'metadata': {
                        'analysis_type': 'market_analysis',
                        'agent_confidence': 0.90,
                        'data_sources': ['market_data', 'technical_indicators']
                    }
                }
            else:
                return self._fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"Error in analysis agent: {e}")
            return self._fallback_response(prompt)
            
    def _call_quant_gpt_agent(self, agent: Any, prompt: str) -> Dict[str, Any]:
        """
        Call the QuantGPT agent.
        
        Args:
            agent: QuantGPT agent instance
            prompt: User prompt
            
        Returns:
            Dictionary containing QuantGPT response
        """
        try:
            if hasattr(agent, 'process_query'):
                response = agent.process_query(prompt)
                return {
                    'type': 'quant_gpt',
                    'content': response,
                    'agent': 'quant_gpt',
                    'timestamp': datetime.now().isoformat(),
                    'confidence': 0.95,
                    'metadata': {
                        'model_type': 'quantitative_analysis',
                        'agent_confidence': 0.95,
                        'reasoning_chain': 'quantitative'
                    }
                }
            else:
                return self._fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"Error in QuantGPT agent: {e}")
            return self._fallback_response(prompt)
            
    def _call_llm_agent(self, agent: Any, prompt: str) -> Dict[str, Any]:
        """
        Call the LLM agent.
        
        Args:
            agent: LLM agent instance
            prompt: User prompt
            
        Returns:
            Dictionary containing LLM response
        """
        try:
            if hasattr(agent, 'generate_response'):
                response = agent.generate_response(prompt)
                return {
                    'type': 'llm',
                    'content': response,
                    'agent': 'llm',
                    'timestamp': datetime.now().isoformat(),
                    'confidence': 0.75,
                    'metadata': {
                        'model_type': 'language_model',
                        'agent_confidence': 0.75,
                        'response_type': 'general'
                    }
                }
            else:
                return self._fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"Error in LLM agent: {e}")
            return self._fallback_response(prompt)
            
    def _extract_ticker(self, prompt: str) -> str:
        """
        Extract ticker symbol from prompt.
        
        Args:
            prompt: User prompt text
            
        Returns:
            Extracted ticker symbol or default
        """
        import re
        # Look for common ticker patterns
        ticker_pattern = r'\b([A-Z]{1,5})\b'
        matches = re.findall(ticker_pattern, prompt)
        
        # Filter out common words that might match ticker pattern
        common_words = {'THE', 'AND', 'FOR', 'WITH', 'FROM', 'THIS', 'THAT', 'WHAT', 'WHEN', 'WHERE'}
        for match in matches:
            if match not in common_words:
                return match
                
        return 'AAPL'  # Default fallback
        
    def _extract_timeframe(self, prompt: str) -> str:
        """
        Extract timeframe from prompt.
        
        Args:
            prompt: User prompt text
            
        Returns:
            Extracted timeframe or default
        """
        import re
        timeframe_pattern = r'\b(daily|weekly|monthly|yearly|1d|1w|1m|1y|30d|60d|90d)\b'
        match = re.search(timeframe_pattern, prompt.lower())
        return match.group(1) if match else '30d'
        
    def _fallback_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a fallback response when no agent is available.
        
        Args:
            prompt: Original user prompt
            
        Returns:
            Dictionary containing fallback response
        """
        return {
            'type': 'fallback',
            'content': f"I'm sorry, I couldn't process your request: '{prompt}'. Please try rephrasing or contact support.",
            'agent': 'fallback',
            'timestamp': datetime.now().isoformat(),
            'confidence': 0.0,
            'metadata': {
                'error': 'No suitable agent available',
                'fallback_used': True
            }
        }
        
    def _log_interaction(self, prompt: str, agent_type: str, response: Dict[str, Any]) -> None:
        """
        Log the interaction for monitoring and debugging.
        
        Args:
            prompt: User prompt
            agent_type: Type of agent used
            response: Agent response
        """
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'agent_type': agent_type,
            'response_type': response.get('type', 'unknown'),
            'confidence': response.get('confidence', 0.0)
        }
        self.interaction_history.append(interaction)
        
        # Keep only recent interactions
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]

    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get status of all agents.
        
        Returns:
            Dictionary containing status of all agents
        """
        status = {}
        for name, agent in self.agents.items():
            status[name] = {
                'available': agent is not None,
                'health': self._check_agent_health(agent),
                'last_used': None  # Could be enhanced with actual usage tracking
            }
        return status
        
    def _check_agent_health(self, agent: Any) -> str:
        """
        Check the health of an agent.
        
        Args:
            agent: Agent instance to check
            
        Returns:
            Health status string
        """
        if agent is None:
            return 'unavailable'
        try:
            # Simple health check - could be enhanced
            if hasattr(agent, 'health_check'):
                return agent.health_check()
            else:
                return 'unknown'
        except Exception:
            return 'error'
            
    def get_recent_interactions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent interactions for monitoring.
        
        Args:
            limit: Maximum number of interactions to return
            
        Returns:
            List of recent interactions
        """
        return self.interaction_history[-limit:]
        
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health.
        
        Returns:
            Dictionary containing system health information
        """
        agent_status = self.get_agent_status()
        available_agents = sum(1 for status in agent_status.values() if status['available'])
        total_agents = len(agent_status)
        
        return {
            'total_agents': total_agents,
            'available_agents': available_agents,
            'health_score': available_agents / total_agents if total_agents > 0 else 0,
            'fallback_agent_available': self.fallback_agent is not None,
            'recent_interactions': len(self.interaction_history),
            'timestamp': datetime.now().isoformat()
        }
        
    def reset_agents(self) -> bool:
        """
        Reset all agents.
        
        Returns:
            True if reset successful, False otherwise
        """
        try:
            self._initialize_agents()
            self.interaction_history = []
            logger.info("All agents reset successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reset agents: {e}")
            return False
