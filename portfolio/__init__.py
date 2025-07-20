"""
Portfolio Management Module

This module provides comprehensive portfolio management capabilities including:
- Portfolio allocation strategies (MPT, Risk Parity, Kelly Criterion, etc.)
- Risk management and monitoring
- Portfolio simulation and backtesting
- Dynamic rebalancing
- Stress testing and scenario analysis

Main Components:
- allocator.py: Portfolio allocation strategies
- risk_manager.py: Risk management and monitoring
"""

# Import main classes from allocator
from .allocator import (
    PortfolioAllocator,
    AllocationStrategy,
    AssetMetrics,
    AllocationResult,
    create_allocator,
    allocate_portfolio
)

# Import main classes from risk_manager
from .risk_manager import (
    PortfolioRiskManager,
    RiskMetric,
    RiskLimits,
    PortfolioState,
    RiskViolation,
    RebalancingAction,
    create_risk_manager,
    calculate_portfolio_risk,
    check_risk_limits
)

# Version information
__version__ = "1.0.0"
__author__ = "Evolve Trading Platform"
__description__ = "Comprehensive portfolio management and risk control"

# Main exports
__all__ = [
    # Allocator exports
    'PortfolioAllocator',
    'AllocationStrategy', 
    'AssetMetrics',
    'AllocationResult',
    'create_allocator',
    'allocate_portfolio',
    
    # Risk manager exports
    'PortfolioRiskManager',
    'RiskMetric',
    'RiskLimits',
    'PortfolioState',
    'RiskViolation',
    'RebalancingAction',
    'create_risk_manager',
    'calculate_portfolio_risk',
    'check_risk_limits'
]

# Convenience function for quick portfolio analysis
def analyze_portfolio(assets, strategy='maximum_sharpe', config_path="config/app_config.yaml"):
    """
    Quick portfolio analysis function
    
    Args:
        assets: List of AssetMetrics objects
        strategy: Allocation strategy to use
        config_path: Path to configuration file
    
    Returns:
        dict: Analysis results including allocation, risk metrics, and recommendations
    """
    # Create allocator and risk manager
    allocator = create_allocator(config_path)
    risk_manager = create_risk_manager(config_path)
    
    # Allocate portfolio
    allocation_result = allocator.allocate_portfolio(assets, AllocationStrategy(strategy))
    
    # Simulate market data for risk analysis
    # This is a simplified version - in practice you'd use real market data
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    market_data = {}
    
    for asset in assets:
        returns = np.random.normal(asset.expected_return/252, asset.volatility/np.sqrt(252), len(dates))
        market_data[asset.ticker] = pd.DataFrame({'returns': returns}, index=dates)
    
    # Calculate risk metrics
    risk_metrics = risk_manager.calculate_portfolio_risk(allocation_result.weights, market_data)
    
    # Check risk limits
    violations = risk_manager.check_risk_limits(allocation_result.weights, risk_metrics)
    
    # Generate recommendations
    recommendations = []
    if violations:
        recommendations.append(f"Risk violations detected: {len(violations)}")
        for violation in violations:
            recommendations.append(f"- {violation.risk_metric.value}: {violation.action_required}")
    else:
        recommendations.append("No risk violations detected")
    
    recommendations.extend([
        f"Expected Return: {allocation_result.expected_return:.2%}",
        f"Expected Volatility: {allocation_result.expected_volatility:.2%}",
        f"Sharpe Ratio: {allocation_result.sharpe_ratio:.3f}",
        f"Diversification Ratio: {allocation_result.diversification_ratio:.3f}"
    ])
    
    return {
        'allocation': allocation_result,
        'risk_metrics': risk_metrics,
        'violations': violations,
        'recommendations': recommendations
    }


# Example usage function
def example_usage():
    """
    Example usage of the portfolio management module
    """
    print("Portfolio Management Module Example")
    print("="*50)
    
    # Create sample assets
    assets = [
        AssetMetrics(
            ticker="AAPL",
            expected_return=0.12,
            volatility=0.20,
            sharpe_ratio=0.5,
            beta=1.1,
            correlation={"TSLA": 0.3, "NVDA": 0.4},
            market_cap=2000000000000,
            sector="Technology",
            sentiment_score=0.6
        ),
        AssetMetrics(
            ticker="TSLA",
            expected_return=0.18,
            volatility=0.35,
            sharpe_ratio=0.46,
            beta=1.8,
            correlation={"AAPL": 0.3, "NVDA": 0.5},
            market_cap=800000000000,
            sector="Automotive",
            sentiment_score=0.7
        ),
        AssetMetrics(
            ticker="NVDA",
            expected_return=0.15,
            volatility=0.30,
            sharpe_ratio=0.43,
            beta=1.5,
            correlation={"AAPL": 0.4, "TSLA": 0.5},
            market_cap=1200000000000,
            sector="Technology",
            sentiment_score=0.8
        )
    ]
    
    # Quick analysis
    results = analyze_portfolio(assets, 'maximum_sharpe')
    
    print("Portfolio Analysis Results:")
    print(f"Strategy: {results['allocation'].strategy.value}")
    print(f"Weights: {results['allocation'].weights}")
    print(f"Expected Return: {results['allocation'].expected_return:.2%}")
    print(f"Expected Volatility: {results['allocation'].expected_volatility:.2%}")
    print(f"Sharpe Ratio: {results['allocation'].sharpe_ratio:.3f}")
    
    print("\nRisk Metrics:")
    for metric, value in results['risk_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nRecommendations:")
    for rec in results['recommendations']:
        print(f"  - {rec}")


if __name__ == "__main__":
    example_usage()
