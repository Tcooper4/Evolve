"""
Agent Leaderboard Demo

Demonstrates the AgentLeaderboard functionality with realistic trading performance data,
showing how to track agent performance, handle deprecation, and integrate with dashboards.
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd

from trading.agents.agent_leaderboard import AgentLeaderboard, AgentPerformance
from trading.agents.agent_manager import AgentManager, AgentManagerConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_realistic_performance_data(agent_name: str) -> Dict[str, float]:
    """Generate realistic trading performance data for demonstration."""
    
    # Base performance characteristics by agent type
    agent_performance_profiles = {
        "model_builder": {
            "sharpe_range": (0.8, 2.5),
            "drawdown_range": (0.05, 0.20),
            "win_rate_range": (0.55, 0.75),
            "return_range": (0.15, 0.45)
        },
        "performance_critic": {
            "sharpe_range": (1.2, 3.0),
            "drawdown_range": (0.03, 0.15),
            "win_rate_range": (0.60, 0.80),
            "return_range": (0.20, 0.50)
        },
        "updater": {
            "sharpe_range": (0.6, 2.0),
            "drawdown_range": (0.08, 0.25),
            "win_rate_range": (0.50, 0.70),
            "return_range": (0.12, 0.35)
        },
        "execution_agent": {
            "sharpe_range": (1.0, 2.8),
            "drawdown_range": (0.04, 0.18),
            "win_rate_range": (0.58, 0.78),
            "return_range": (0.18, 0.42)
        },
        "optimizer_agent": {
            "sharpe_range": (1.5, 3.2),
            "drawdown_range": (0.02, 0.12),
            "win_rate_range": (0.65, 0.85),
            "return_range": (0.25, 0.55)
        }
    }
    
    # Get profile based on agent name
    profile = agent_performance_profiles.get(agent_name, agent_performance_profiles["model_builder"])
    
    # Generate performance metrics with some randomness
    sharpe_ratio = random.uniform(*profile["sharpe_range"])
    max_drawdown = random.uniform(*profile["drawdown_range"])
    win_rate = random.uniform(*profile["win_rate_range"])
    total_return = random.uniform(*profile["return_range"])
    
    # Add some extra metrics
    extra_metrics = {
        "volatility": random.uniform(0.15, 0.35),
        "calmar_ratio": sharpe_ratio / max_drawdown if max_drawdown > 0 else 0,
        "profit_factor": random.uniform(1.2, 3.0),
        "max_consecutive_losses": random.randint(3, 8),
        "avg_trade_duration": random.uniform(2.5, 8.0),
        "total_trades": random.randint(50, 200)
    }
    
    return {
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_return": total_return,
        "extra_metrics": extra_metrics
    }

def demo_basic_leaderboard_usage():
    """Demonstrate basic leaderboard functionality."""
    print("\n" + "="*60)
    print("BASIC LEADERBOARD USAGE")
    print("="*60)
    
    # Initialize leaderboard
    leaderboard = AgentLeaderboard()
    
    # Add some agents with performance data
    agents = ["model_builder", "performance_critic", "updater", "execution_agent", "optimizer_agent"]
    
    for agent in agents:
        perf_data = generate_realistic_performance_data(agent)
        leaderboard.update_performance(
            agent_name=agent,
            sharpe_ratio=perf_data["sharpe_ratio"],
            max_drawdown=perf_data["max_drawdown"],
            win_rate=perf_data["win_rate"],
            total_return=perf_data["total_return"],
            extra_metrics=perf_data["extra_metrics"]
        )
    
    # Display leaderboard
    print("\nTop 5 Agents by Sharpe Ratio:")
    top_agents = leaderboard.get_leaderboard(top_n=5, sort_by="sharpe_ratio")
    for i, agent in enumerate(top_agents, 1):
        print(f"{i}. {agent['agent_name']}: Sharpe={agent['sharpe_ratio']:.2f}, "
              f"Drawdown={agent['max_drawdown']:.2%}, WinRate={agent['win_rate']:.2%}")
    
    # Show active vs deprecated agents
    print(f"\nActive Agents: {leaderboard.get_active_agents()}")
    print(f"Deprecated Agents: {leaderboard.get_deprecated_agents()}")
    
    # Show as DataFrame
    print("\nLeaderboard as DataFrame:")
    df = leaderboard.as_dataframe()
    print(df[['agent_name', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_return', 'status']].to_string(index=False))

def demo_deprecation_scenarios():
    """Demonstrate deprecation functionality with poor performers."""
    print("\n" + "="*60)
    print("DEPRECATION SCENARIOS")
    print("="*60)
    
    leaderboard = AgentLeaderboard()
    
    # Add some poor performing agents that should be deprecated
    poor_performers = [
        ("weak_model", 0.3, 0.35, 0.40, 0.05),  # Low Sharpe, high drawdown, low win rate
        ("risky_strategy", 0.4, 0.40, 0.42, 0.08),  # Very high drawdown
        ("unlucky_agent", 0.2, 0.20, 0.38, -0.05),  # Very low win rate
    ]
    
    for agent_name, sharpe, drawdown, win_rate, total_return in poor_performers:
        leaderboard.update_performance(agent_name, sharpe, drawdown, win_rate, total_return)
        print(f"Added {agent_name}: Sharpe={sharpe:.2f}, Drawdown={drawdown:.2%}, WinRate={win_rate:.2%}")
    
    print(f"\nDeprecated Agents: {leaderboard.get_deprecated_agents()}")
    print(f"Active Agents: {leaderboard.get_active_agents()}")

def demo_agent_manager_integration():
    """Demonstrate integration with AgentManager."""
    print("\n" + "="*60)
    print("AGENT MANAGER INTEGRATION")
    print("="*60)
    
    # Initialize agent manager
    config = AgentManagerConfig(
        config_file="trading/agents/agent_config.json",
        auto_start=False,
        enable_metrics=True
    )
    manager = AgentManager(config)
    
    # Simulate performance updates through the manager
    agents = ["model_builder", "performance_critic", "updater", "execution_agent"]
    
    for agent in agents:
        perf_data = generate_realistic_performance_data(agent)
        manager.log_agent_performance(
            agent_name=agent,
            sharpe_ratio=perf_data["sharpe_ratio"],
            max_drawdown=perf_data["max_drawdown"],
            win_rate=perf_data["win_rate"],
            total_return=perf_data["total_return"],
            extra_metrics=perf_data["extra_metrics"]
        )
    
    # Get leaderboard data through manager
    print("\nTop 3 Agents via AgentManager:")
    top_agents = manager.get_leaderboard(top_n=3, sort_by="sharpe_ratio")
    for i, agent in enumerate(top_agents, 1):
        print(f"{i}. {agent['agent_name']}: Sharpe={agent['sharpe_ratio']:.2f}, "
              f"Return={agent['total_return']:.2%}")
    
    # Get deprecated agents
    deprecated = manager.get_deprecated_agents()
    print(f"\nDeprecated Agents: {deprecated}")
    
    # Get active agents
    active = manager.get_active_agents()
    print(f"Active Agents: {active}")

def demo_leaderboard_analytics():
    """Demonstrate advanced leaderboard analytics."""
    print("\n" + "="*60)
    print("LEADERBOARD ANALYTICS")
    print("="*60)
    
    leaderboard = AgentLeaderboard()
    
    # Add diverse performance data
    agents_data = [
        ("conservative_model", 1.8, 0.08, 0.65, 0.25),
        ("aggressive_model", 2.5, 0.20, 0.58, 0.40),
        ("balanced_model", 2.1, 0.12, 0.62, 0.30),
        ("momentum_model", 1.9, 0.15, 0.60, 0.28),
        ("mean_reversion_model", 1.6, 0.10, 0.68, 0.22),
    ]
    
    for agent_name, sharpe, drawdown, win_rate, total_return in agents_data:
        leaderboard.update_performance(agent_name, sharpe, drawdown, win_rate, total_return)
    
    # Get DataFrame for analysis
    df = leaderboard.as_dataframe()
    
    print("\nPerformance Summary:")
    print(f"Total Agents: {len(df)}")
    print(f"Active Agents: {len(df[df['status'] == 'active'])}")
    print(f"Deprecated Agents: {len(df[df['status'] == 'deprecated'])}")
    
    print("\nPerformance Statistics:")
    print(f"Average Sharpe Ratio: {df['sharpe_ratio'].mean():.2f}")
    print(f"Average Max Drawdown: {df['max_drawdown'].mean():.2%}")
    print(f"Average Win Rate: {df['win_rate'].mean():.2%}")
    print(f"Average Total Return: {df['total_return'].mean():.2%}")
    
    # Show best performers by different metrics
    print("\nBest Performers by Metric:")
    best_sharpe = df.loc[df['sharpe_ratio'].idxmax()]
    best_return = df.loc[df['total_return'].idxmax()]
    best_win_rate = df.loc[df['win_rate'].idxmax()]
    
    print(f"Best Sharpe: {best_sharpe['agent_name']} ({best_sharpe['sharpe_ratio']:.2f})")
    print(f"Best Return: {best_return['agent_name']} ({best_return['total_return']:.2%})")
    print(f"Best Win Rate: {best_return['agent_name']} ({best_win_rate['win_rate']:.2%})")

def demo_reporting_integration():
    """Demonstrate how leaderboard data can be used in reports."""
    print("\n" + "="*60)
    print("REPORTING INTEGRATION")
    print("="*60)
    
    leaderboard = AgentLeaderboard()
    
    # Add performance data
    agents = ["model_builder", "performance_critic", "updater", "execution_agent"]
    for agent in agents:
        perf_data = generate_realistic_performance_data(agent)
        leaderboard.update_performance(
            agent_name=agent,
            sharpe_ratio=perf_data["sharpe_ratio"],
            max_drawdown=perf_data["max_drawdown"],
            win_rate=perf_data["win_rate"],
            total_return=perf_data["total_return"],
            extra_metrics=perf_data["extra_metrics"]
        )
    
    # Generate report data
    df = leaderboard.as_dataframe()
    
    # Example report sections
    report_data = {
        "summary": {
            "total_agents": len(df),
            "active_agents": len(df[df['status'] == 'active']),
            "deprecated_agents": len(df[df['status'] == 'deprecated']),
            "avg_sharpe": df['sharpe_ratio'].mean(),
            "avg_return": df['total_return'].mean()
        },
        "top_performers": leaderboard.get_leaderboard(top_n=3, sort_by="sharpe_ratio"),
        "deprecated_agents": leaderboard.get_deprecated_agents(),
        "performance_history": leaderboard.get_history(limit=10)
    }
    
    print("\nReport Data Structure:")
    for section, data in report_data.items():
        print(f"\n{section.upper()}:")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"  {key}: {value}")
        elif isinstance(data, list):
            print(f"  {len(data)} items")
        else:
            print(f"  {data}")

async def main():
    """Run all demo scenarios."""
    print("AGENT LEADERBOARD DEMO")
    print("="*60)
    
    # Run all demo scenarios
    demo_basic_leaderboard_usage()
    demo_deprecation_scenarios()
    demo_agent_manager_integration()
    demo_leaderboard_analytics()
    demo_reporting_integration()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("✓ Basic leaderboard functionality")
    print("✓ Performance tracking and updates")
    print("✓ Automatic deprecation of underperformers")
    print("✓ AgentManager integration")
    print("✓ Analytics and reporting capabilities")
    print("✓ DataFrame export for dashboards")
    print("✓ Performance history tracking")

if __name__ == "__main__":
    asyncio.run(main()) 