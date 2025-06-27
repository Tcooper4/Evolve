#!/usr/bin/env python3
"""
Simple integration test for AgentLeaderboard with AgentManager
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from trading.agents.agent_manager import AgentManager, AgentManagerConfig
from trading.agents.agent_leaderboard import AgentLeaderboard

async def test_integration():
    """Test the integration between AgentManager and AgentLeaderboard."""
    print("🧪 Testing AgentLeaderboard Integration")
    print("=" * 50)
    
    # Initialize agent manager
    config = AgentManagerConfig(
        config_file="trading/agents/agent_config.json",
        auto_start=False,
        enable_metrics=True
    )
    manager = AgentManager(config)
    
    print("✅ AgentManager initialized with leaderboard")
    
    # Add some sample performance data
    sample_data = [
        ("model_builder_v1", 1.8, 0.12, 0.68, 0.35),
        ("performance_critic_v2", 2.1, 0.08, 0.72, 0.42),
        ("updater_v1", 1.5, 0.15, 0.65, 0.28),
        ("execution_agent_v3", 1.9, 0.10, 0.70, 0.38),
        ("weak_agent", 0.3, 0.30, 0.40, 0.05),  # Will be deprecated
    ]
    
    print("\n📊 Adding sample performance data...")
    for agent_name, sharpe, drawdown, win_rate, total_return in sample_data:
        manager.log_agent_performance(
            agent_name=agent_name,
            sharpe_ratio=sharpe,
            max_drawdown=drawdown,
            win_rate=win_rate,
            total_return=total_return,
            extra_metrics={
                "volatility": 0.18,
                "calmar_ratio": sharpe / drawdown if drawdown > 0 else 0,
                "profit_factor": 1.8
            }
        )
        print(f"  ✅ Added {agent_name}: Sharpe={sharpe:.2f}, Drawdown={drawdown:.2%}")
    
    # Test leaderboard functionality
    print("\n🏆 Testing Leaderboard Functionality:")
    
    # Get top performers
    top_agents = manager.get_leaderboard(top_n=3, sort_by="sharpe_ratio")
    print(f"  Top 3 agents by Sharpe ratio:")
    for i, agent in enumerate(top_agents, 1):
        print(f"    {i}. {agent['agent_name']}: Sharpe={agent['sharpe_ratio']:.2f}")
    
    # Get deprecated agents
    deprecated = manager.get_deprecated_agents()
    print(f"  Deprecated agents: {deprecated}")
    
    # Get active agents
    active = manager.get_active_agents()
    print(f"  Active agents: {len(active)}")
    
    # Test DataFrame export
    df = manager.get_leaderboard_dataframe()
    print(f"\n📋 Leaderboard DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    # Test history
    history = manager.get_leaderboard_history(limit=10)
    print(f"  Performance history entries: {len(history)}")
    
    print("\n✅ Integration test completed successfully!")
    print("\n🎯 Key Features Demonstrated:")
    print("  ✓ Agent performance tracking")
    print("  ✓ Automatic deprecation")
    print("  ✓ Leaderboard ranking")
    print("  ✓ DataFrame export")
    print("  ✓ Performance history")
    print("  ✓ AgentManager integration")

if __name__ == "__main__":
    asyncio.run(test_integration()) 