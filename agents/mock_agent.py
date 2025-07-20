"""
Mock Agent for Fallback

This module provides a mock agent that can be used as a fallback when no agents
are registered or when agent registration fails.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MockAgentResult:
    """Result from mock agent execution."""
    success: bool
    data: Dict[str, Any]
    message: str
    agent_name: str
    execution_time: float


class MockAgent:
    """
    Mock agent that provides basic functionality when no real agents are available.
    
    This agent can handle basic queries and provide informative responses
    about the system status and available capabilities.
    """
    
    def __init__(self, name: str = "MockAgent", capabilities: Optional[list] = None):
        """
        Initialize the mock agent.
        
        Args:
            name: Name of the agent
            capabilities: List of capabilities this agent can handle
        """
        self.name = name
        self.capabilities = capabilities or [
            "general_query",
            "system_status",
            "help",
            "fallback_response"
        ]
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.created_at = datetime.now()
        
        self.logger.info(f"Mock agent '{name}' initialized with capabilities: {self.capabilities}")
    
    async def execute(self, prompt: str, **kwargs) -> MockAgentResult:
        """
        Execute a mock response based on the prompt.
        
        Args:
            prompt: User prompt
            **kwargs: Additional parameters
            
        Returns:
            MockAgentResult: Mock execution result
        """
        start_time = datetime.now()
        
        try:
            # Analyze the prompt to determine response type
            response_type = self._analyze_prompt(prompt)
            
            # Generate appropriate response
            if response_type == "system_status":
                result = self._generate_system_status_response()
            elif response_type == "help":
                result = self._generate_help_response()
            elif response_type == "forecast":
                result = self._generate_forecast_response(prompt)
            elif response_type == "strategy":
                result = self._generate_strategy_response(prompt)
            elif response_type == "analysis":
                result = self._generate_analysis_response(prompt)
            else:
                result = self._generate_general_response(prompt)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return MockAgentResult(
                success=True,
                data=result,
                message=f"Mock agent '{self.name}' processed your request",
                agent_name=self.name,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in mock agent execution: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return MockAgentResult(
                success=False,
                data={"error": str(e)},
                message=f"Mock agent encountered an error: {str(e)}",
                agent_name=self.name,
                execution_time=execution_time
            )
    
    def _analyze_prompt(self, prompt: str) -> str:
        """Analyze prompt to determine response type."""
        prompt_lower = prompt.lower()
        
        # Check for system status requests
        if any(word in prompt_lower for word in ["status", "health", "system", "running"]):
            return "system_status"
        
        # Check for help requests
        if any(word in prompt_lower for word in ["help", "what can you do", "capabilities", "available"]):
            return "help"
        
        # Check for forecast requests
        if any(word in prompt_lower for word in ["forecast", "predict", "price", "stock"]):
            return "forecast"
        
        # Check for strategy requests
        if any(word in prompt_lower for word in ["strategy", "trade", "approach", "method"]):
            return "strategy"
        
        # Check for analysis requests
        if any(word in prompt_lower for word in ["analyze", "analysis", "market", "trend"]):
            return "analysis"
        
        return "general"
    
    def _generate_system_status_response(self) -> Dict[str, Any]:
        """Generate system status response."""
        return {
            "type": "system_status",
            "status": "operational",
            "message": "System is running with mock agent fallback",
            "timestamp": datetime.now().isoformat(),
            "agent_info": {
                "name": self.name,
                "type": "mock",
                "capabilities": self.capabilities,
                "created_at": self.created_at.isoformat()
            },
            "system_info": {
                "mode": "fallback",
                "real_agents_available": False,
                "recommendation": "Check agent registration and configuration"
            }
        }
    
    def _generate_help_response(self) -> Dict[str, Any]:
        """Generate help response."""
        return {
            "type": "help",
            "message": "I'm a mock agent providing fallback functionality",
            "capabilities": self.capabilities,
            "available_actions": [
                "System status queries",
                "Basic help and information",
                "Mock forecast responses",
                "Mock strategy suggestions",
                "Mock market analysis"
            ],
            "note": "This is a fallback agent. Real agents may not be properly registered.",
            "suggestions": [
                "Check agent configuration files",
                "Verify agent registration process",
                "Review system logs for errors",
                "Restart the system to reload agents"
            ]
        }
    
    def _generate_forecast_response(self, prompt: str) -> Dict[str, Any]:
        """Generate mock forecast response."""
        return {
            "type": "forecast",
            "message": "Mock forecast response (no real agents available)",
            "original_prompt": prompt,
            "forecast_data": {
                "prediction": "Mock prediction",
                "confidence": 0.5,
                "timeframe": "1 day",
                "note": "This is a placeholder response from mock agent"
            },
            "agent_note": "Real forecasting agents are not available. Check agent registration."
        }
    
    def _generate_strategy_response(self, prompt: str) -> Dict[str, Any]:
        """Generate mock strategy response."""
        return {
            "type": "strategy",
            "message": "Mock strategy response (no real agents available)",
            "original_prompt": prompt,
            "strategy_data": {
                "strategy_type": "mock_strategy",
                "description": "Placeholder strategy from mock agent",
                "risk_level": "medium",
                "note": "This is a placeholder response from mock agent"
            },
            "agent_note": "Real strategy agents are not available. Check agent registration."
        }
    
    def _generate_analysis_response(self, prompt: str) -> Dict[str, Any]:
        """Generate mock analysis response."""
        return {
            "type": "analysis",
            "message": "Mock analysis response (no real agents available)",
            "original_prompt": prompt,
            "analysis_data": {
                "analysis_type": "mock_analysis",
                "summary": "Placeholder analysis from mock agent",
                "key_points": ["Mock point 1", "Mock point 2"],
                "note": "This is a placeholder response from mock agent"
            },
            "agent_note": "Real analysis agents are not available. Check agent registration."
        }
    
    def _generate_general_response(self, prompt: str) -> Dict[str, Any]:
        """Generate general response."""
        return {
            "type": "general",
            "message": "Mock agent response (no real agents available)",
            "original_prompt": prompt,
            "response": "I'm a mock agent providing fallback functionality. Real agents are not currently available.",
            "suggestions": [
                "Check if agents are properly registered",
                "Review system configuration",
                "Check logs for agent initialization errors",
                "Restart the system to reload agents"
            ]
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "name": self.name,
            "type": "mock",
            "status": "active",
            "capabilities": self.capabilities,
            "created_at": self.created_at.isoformat(),
            "uptime_seconds": (datetime.now() - self.created_at).total_seconds()
        }


def create_mock_agent(name: str = "MockAgent", capabilities: Optional[list] = None) -> MockAgent:
    """
    Create a mock agent instance.
    
    Args:
        name: Name for the mock agent
        capabilities: List of capabilities
        
    Returns:
        MockAgent: Mock agent instance
    """
    return MockAgent(name=name, capabilities=capabilities)


# Default mock agent instance
default_mock_agent = MockAgent("DefaultMockAgent")
