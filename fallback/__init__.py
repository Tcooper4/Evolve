"""
Fallback Module for Evolve Trading Platform

This module provides fallback implementations for all core components
when primary services are unavailable or fail. These fallbacks ensure
the system remains functional even in degraded conditions.

Components:
- AgentHub: Fallback agent routing and management
- DataFeed: Mock data generation and basic data operations
- PromptRouter: Basic prompt interpretation and routing
- ModelMonitor: Simple model performance tracking
- StrategyLogger: Basic decision logging
- PortfolioManager: Mock portfolio operations
- StrategySelector: Basic strategy selection
- MarketRegimeAgent: Simple regime classification
- HybridEngine: Basic strategy execution
- QuantGPT: Mock AI commentary generation
- ReportExporter: Basic report generation
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging

from .agent_hub import FallbackAgentHub
from .data_feed import FallbackDataFeed
from .prompt_router import FallbackPromptRouter
from .model_monitor import FallbackModelMonitor
from .strategy_logger import FallbackStrategyLogger
from .portfolio_manager import FallbackPortfolioManager
from .strategy_selector import FallbackStrategySelector
from .market_regime_agent import FallbackMarketRegimeAgent
from .hybrid_engine import FallbackHybridEngine
from .quant_gpt import FallbackQuantGPT
from .report_exporter import FallbackReportExporter

__all__ = [
    'FallbackAgentHub',
    'FallbackDataFeed', 
    'FallbackPromptRouter',
    'FallbackModelMonitor',
    'FallbackStrategyLogger',
    'FallbackPortfolioManager',
    'FallbackStrategySelector',
    'FallbackMarketRegimeAgent',
    'FallbackHybridEngine',
    'FallbackQuantGPT',
    'FallbackReportExporter'
]

logger = logging.getLogger(__name__)

def create_fallback_components() -> Dict[str, Any]:
    """
    Create all fallback components and return them in a dictionary.
    
    Returns:
        Dict[str, Any]: Dictionary containing all fallback component instances
    """
    logger.info("Creating fallback components")
    
    try:
        components = {
            'agent_hub': FallbackAgentHub(),
            'data_feed': FallbackDataFeed(),
            'prompt_router': FallbackPromptRouter(),
            'model_monitor': FallbackModelMonitor(),
            'strategy_logger': FallbackStrategyLogger(),
            'portfolio_manager': FallbackPortfolioManager(),
            'strategy_selector': FallbackStrategySelector(),
            'market_regime_agent': FallbackMarketRegimeAgent(),
            'hybrid_engine': FallbackHybridEngine(),
            'quant_gpt': FallbackQuantGPT(),
            'report_exporter': FallbackReportExporter()
        }
        
        logger.info(f"Successfully created {len(components)} fallback components")
        return components
        
    except Exception as e:
        logger.error(f"Failed to create fallback components: {e}")
        raise 