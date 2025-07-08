#!/usr/bin/env python3
"""
Test Agent Integration

This script tests the integration between different agent components.
"""

import logging
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from trading.agents.agent_manager import AgentManager, AgentManagerConfig
from trading.agents.agent_leaderboard import AgentLeaderboard
from trading.core.agents import AgentConfig, AgentType

async def test_agent_integration():
    """Test agent integration functionality."""
    logger.info("🧪 Testing AgentLeaderboard Integration")
    logger.info("=" * 50)
    
    try:
        # Initialize components
        leaderboard = AgentLeaderboard()
        agent_manager = AgentManager()
        
        # Register leaderboard with agent manager
        agent_manager.register_callback(
            EventType.PERFORMANCE_IMPROVEMENT,
            leaderboard.update_agent_performance
        )
        
        logger.info("✅ AgentManager initialized with leaderboard")
        
        # Add sample agents
        sample_agents = [
            AgentConfig(
                agent_id="agent_1",
                name="RSI Strategy",
                agent_type=AgentType.TRADING
            ),
            AgentConfig(
                agent_id="agent_2", 
                name="MACD Strategy",
                agent_type=AgentType.TRADING
            ),
            AgentConfig(
                agent_id="agent_3",
                name="Bollinger Strategy", 
                agent_type=AgentType.TRADING
            )
        ]
        
        for agent_config in sample_agents:
            agent_manager.register_agent(agent_config)
        
        logger.info("\n📊 Adding sample performance data...")
        
        # Add sample performance data
        performance_data = [
            {"agent_name": "RSI Strategy", "sharpe_ratio": 1.2, "max_drawdown": 0.15, "win_rate": 0.65},
            {"agent_name": "MACD Strategy", "sharpe_ratio": 0.9, "max_drawdown": 0.20, "win_rate": 0.58},
            {"agent_name": "Bollinger Strategy", "sharpe_ratio": 1.5, "max_drawdown": 0.12, "win_rate": 0.72}
        ]
        
        for data in performance_data:
            leaderboard.add_agent_performance(
                agent_name=data["agent_name"],
                sharpe_ratio=data["sharpe_ratio"],
                max_drawdown=data["max_drawdown"],
                win_rate=data["win_rate"]
            )
            logger.info(f"  ✅ Added {data['agent_name']}: Sharpe={data['sharpe_ratio']:.2f}, Drawdown={data['max_drawdown']:.2%}")
        
        # Test leaderboard functionality
        logger.info("\n🏆 Testing Leaderboard Functionality:")
        
        # Get top agents
        top_agents = leaderboard.get_top_agents(limit=3, sort_by='sharpe_ratio')
        logger.info(f"  Top 3 agents by Sharpe ratio:")
        for i, agent in enumerate(top_agents, 1):
            logger.info(f"    {i}. {agent['agent_name']}: Sharpe={agent['sharpe_ratio']:.2f}")
        
        # Get deprecated agents
        deprecated = leaderboard.get_deprecated_agents()
        logger.info(f"  Deprecated agents: {deprecated}")
        
        # Get active agents
        active = leaderboard.get_active_agents()
        logger.info(f"  Active agents: {len(active)}")
        
        # Test DataFrame export
        df = leaderboard.get_leaderboard_dataframe()
        logger.info(f"\n📋 Leaderboard DataFrame shape: {df.shape}")
        logger.info(f"  Columns: {list(df.columns)}")
        
        # Test performance history
        history = leaderboard.get_performance_history("RSI Strategy")
        logger.info(f"  Performance history entries: {len(history)}")
        
        logger.info("\n✅ Integration test completed successfully!")
        logger.info("\n🎯 Key Features Demonstrated:")
        logger.info("  ✓ Agent performance tracking")
        logger.info("  ✓ Automatic deprecation")
        logger.info("  ✓ Leaderboard ranking")
        logger.info("  ✓ DataFrame export")
        logger.info("  ✓ Performance history")
        logger.info("  ✓ AgentManager integration")
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_agent_integration()) 