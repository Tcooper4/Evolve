#!/usr/bin/env python3
"""
Fix import issues in trading/__init__.py
"""

import re

def fix_trading_init():
    """Fix the import statements in trading/__init__.py"""
    
    # Read the current file
    with open('trading/__init__.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix optimization imports
    content = re.sub(
        r'from \.optimization import \(\s*StrategyOptimizer, PortfolioOptimizer, OptunaOptimizer,\s*BaseOptimizer, OptimizationVisualizer\s*\)',
        'from .optimization import (\n        StrategyOptimizer, BaseOptimizer, OptimizationVisualizer\n    )',
        content
    )
    
    # Fix risk imports
    content = re.sub(
        r'from \.risk import \(\s*RiskManager, PositionSizingEngine, RiskAnalyzer,\s*RiskAdjustedStrategy, RiskMetrics\s*\)',
        'from .risk import (\n        RiskManager\n    )',
        content
    )
    
    # Fix portfolio imports
    content = re.sub(
        r'from \.portfolio import \(\s*PortfolioManager, PortfolioSimulator, PositionSizer,\s*LLMUtils\s*\)',
        'from .portfolio import (\n        PortfolioManager\n    )',
        content
    )
    
    # Fix agents imports
    content = re.sub(
        r'from \.agents import \(\s*PromptRouterAgent, ExecutionAgent, ModelBuilderAgent,\s*StrategySelectorAgent, MarketRegimeAgent, AgentRegistry\s*\)',
        'from .agents import (\n        PromptRouterAgent, ModelBuilderAgent\n    )',
        content
    )
    
    # Fix utils imports
    content = re.sub(
        r'from \.utils import \(\s*SafeExecutor, ReasoningLogger, PerformanceLogger,\s*ErrorHandler, ConfigUtils, DataUtils\s*\)',
        'from .utils import (\n        LogManager, ModelLogger, DataLogger, PerformanceLogger\n    )',
        content
    )
    
    # Write the fixed content back
    with open('trading/__init__.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed import statements in trading/__init__.py")

if __name__ == "__main__":
    fix_trading_init() 