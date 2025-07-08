"""
Commentary Agent

This agent provides LLM-based commentary for trading decisions and analysis.
It wraps the Commentary Engine and follows the BaseAgent interface for
seamless integration with the agent system.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from .base_agent_interface import BaseAgent, AgentConfig, AgentResult
from trading.commentary import CommentaryEngine, CommentaryType, CommentaryRequest, create_commentary_engine
from trading.memory.agent_memory import AgentMemory
from trading.utils.reasoning_logger import ReasoningLogger, DecisionType, ConfidenceLevel

logger = logging.getLogger(__name__)

@dataclass
class CommentaryAgentRequest:
    """Request for commentary agent."""
    commentary_type: str
    symbol: str
    trade_data: Optional[Dict[str, Any]] = None
    performance_data: Optional[Dict[str, Any]] = None
    market_data: Optional[Any] = None
    portfolio_data: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None

class CommentaryAgent(BaseAgent):
    """
    Commentary Agent that provides LLM-based explanations and insights.
    
    This agent wraps the Commentary Engine and provides a standardized interface
    for generating commentary across the trading system.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the commentary agent."""
        if config is None:
            config = AgentConfig(
                name="CommentaryAgent",
                enabled=True,
                priority=2,
                max_concurrent_runs=5,
                timeout_seconds=60,
                retry_attempts=2,
                custom_config={}
            )
        super().__init__(config)
        
        self.logger = logging.getLogger(__name__)
        self.memory = AgentMemory()
        self.reasoning_logger = ReasoningLogger()
        
        # Initialize commentary engine
        engine_config = self.config.custom_config.get('engine_config', {})
        self.commentary_engine = create_commentary_engine(engine_config)
        
        # Commentary type mapping
        self.commentary_type_mapping = {
            'trade': CommentaryType.TRADE_EXPLANATION,
            'performance': CommentaryType.PERFORMANCE_ANALYSIS,
            'regime': CommentaryType.MARKET_REGIME,
            'risk': CommentaryType.RISK_ASSESSMENT,
            'strategy': CommentaryType.STRATEGY_RECOMMENDATION,
            'counterfactual': CommentaryType.COUNTERFACTUAL_ANALYSIS,
            'daily': CommentaryType.DAILY_SUMMARY,
            'portfolio': CommentaryType.PORTFOLIO_OVERVIEW
        }
        
        self.logger.info("CommentaryAgent initialized successfully")
    
    async def generate_commentary(self, request: CommentaryAgentRequest) -> AgentResult:
        """
        Generate commentary based on the request.
        
        Args:
            request: Commentary request
            
        Returns:
            AgentResult with commentary data
        """
        try:
            self.logger.info(f"Generating {request.commentary_type} commentary for {request.symbol}")
            
            # Map commentary type
            commentary_type = self.commentary_type_mapping.get(
                request.commentary_type.lower(), 
                CommentaryType.TRADE_EXPLANATION
            )
            
            # Create commentary request
            commentary_request = CommentaryRequest(
                commentary_type=commentary_type,
                symbol=request.symbol,
                timestamp=datetime.now(),
                trade_data=request.trade_data,
                performance_data=request.performance_data,
                market_data=request.market_data,
                portfolio_data=request.portfolio_data,
                context=request.context
            )
            
            # Generate commentary
            response = await self.commentary_engine.generate_commentary(commentary_request)
            
            # Log decision
            self._log_commentary_decision(response)
            
            # Store in memory
            self._store_commentary(response)
            
            return AgentResult(
                success=True,
                message=f"Generated {request.commentary_type} commentary",
                data={
                    'commentary': response,
                    'type': request.commentary_type,
                    'symbol': request.symbol,
                    'confidence': response.confidence_score
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating commentary: {str(e)}")
            return AgentResult(
                success=False,
                message=f"Failed to generate commentary: {str(e)}",
                data={'error': str(e)}
            )
    
    def explain_trade(self, symbol: str, trade_data: Dict[str, Any], 
                     market_data: Optional[Any] = None) -> AgentResult:
        """
        Explain a trading decision.
        
        Args:
            symbol: Trading symbol
            trade_data: Trade details
            market_data: Market data
            
        Returns:
            AgentResult with trade explanation
        """
        request = CommentaryAgentRequest(
            commentary_type='trade',
            symbol=symbol,
            trade_data=trade_data,
            market_data=market_data
        )
        
        # Run async function in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new task
                task = asyncio.create_task(self.generate_commentary(request))
                # This is a simplified approach - in practice you'd handle this differently
                return AgentResult(
                    success=False,
                    message="Async commentary generation not supported in sync context",
                    data={'suggestion': 'Use generate_commentary_async for async operation'}
                )
            else:
                return loop.run_until_complete(self.generate_commentary(request))
        except RuntimeError:
            # No event loop running
            return AgentResult(
                success=False,
                message="No event loop available for async commentary generation",
                data={'suggestion': 'Use generate_commentary_async for async operation'}
            )
    
    def analyze_performance(self, symbol: str, performance_data: Dict[str, Any]) -> AgentResult:
        """
        Analyze trading performance.
        
        Args:
            symbol: Trading symbol
            performance_data: Performance metrics
            
        Returns:
            AgentResult with performance analysis
        """
        request = CommentaryAgentRequest(
            commentary_type='performance',
            symbol=symbol,
            performance_data=performance_data
        )
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return AgentResult(
                    success=False,
                    message="Async commentary generation not supported in sync context",
                    data={'suggestion': 'Use generate_commentary_async for async operation'}
                )
            else:
                return loop.run_until_complete(self.generate_commentary(request))
        except RuntimeError:
            return AgentResult(
                success=False,
                message="No event loop available for async commentary generation",
                data={'suggestion': 'Use generate_commentary_async for async operation'}
            )
    
    def assess_risk(self, symbol: str, trade_data: Optional[Dict[str, Any]] = None,
                   portfolio_data: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Assess trading risks.
        
        Args:
            symbol: Trading symbol
            trade_data: Trade details (optional)
            portfolio_data: Portfolio data (optional)
            
        Returns:
            AgentResult with risk assessment
        """
        request = CommentaryAgentRequest(
            commentary_type='risk',
            symbol=symbol,
            trade_data=trade_data,
            portfolio_data=portfolio_data
        )
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return AgentResult(
                    success=False,
                    message="Async commentary generation not supported in sync context",
                    data={'suggestion': 'Use generate_commentary_async for async operation'}
                )
            else:
                return loop.run_until_complete(self.generate_commentary(request))
        except RuntimeError:
            return AgentResult(
                success=False,
                message="No event loop available for async commentary generation",
                data={'suggestion': 'Use generate_commentary_async for async operation'}
            )
    
    def _log_commentary_decision(self, response):
        """Log commentary decision."""
        self.reasoning_logger.log_decision(
            agent_name='CommentaryAgent',
            decision_type=DecisionType.COMMENTARY_GENERATION,
            action_taken=f"Generated {response.commentary_type.value} commentary",
            context={
                'commentary_type': response.commentary_type.value,
                'symbol': response.metadata.get('symbol', 'Unknown'),
                'confidence': response.confidence_score,
                'insights_count': len(response.key_insights),
                'recommendations_count': len(response.recommendations)
            },
            reasoning={
                'primary_reason': f"Generated {response.commentary_type.value} commentary",
                'supporting_factors': response.key_insights,
                'recommendations': response.recommendations,
                'risk_warnings': response.risk_warnings
            },
            confidence_level=ConfidenceLevel.HIGH if response.confidence_score > 0.8 else ConfidenceLevel.MEDIUM,
            metadata=response.metadata
        )
    
    def _store_commentary(self, response):
        """Store commentary in memory."""
        try:
            self.memory.store('commentary_history', {
                'response': response.__dict__,
                'timestamp': datetime.now(),
                'agent': 'CommentaryAgent'
            })
        except Exception as e:
            self.logger.error(f"Error storing commentary: {str(e)}")
    
    def get_commentary_statistics(self) -> Dict[str, Any]:
        """Get commentary generation statistics."""
        return self.commentary_engine.get_commentary_statistics()

# Convenience function for creating commentary agent
def create_commentary_agent(config: Optional[AgentConfig] = None) -> CommentaryAgent:
    """Create a configured commentary agent."""
    return CommentaryAgent(config) 