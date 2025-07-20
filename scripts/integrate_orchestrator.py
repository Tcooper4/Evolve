#!/usr/bin/env python3
"""
Task Orchestrator Integration Script

This script integrates the TaskOrchestrator with the existing Evolve platform
agents and components, ensuring seamless operation and proper initialization.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.task_orchestrator import TaskOrchestrator, TaskConfig, TaskType, TaskPriority


class EvolveOrchestratorIntegration:
    """
    Integration class for connecting TaskOrchestrator with Evolve platform
    """
    
    def __init__(self, config_path: str = "config/task_schedule.yaml"):
        self.config_path = config_path
        self.orchestrator = None
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def integrate_with_existing_agents(self):
        """Integrate orchestrator with existing Evolve agents"""
        self.logger.info("Starting Task Orchestrator integration...")
        
        try:
            # Create orchestrator
            self.orchestrator = TaskOrchestrator(self.config_path)
            
            # Discover and integrate existing agents
            await self._discover_agents()
            
            # Configure agent methods
            await self._configure_agent_methods()
            
            # Update task configurations
            await self._update_task_configs()
            
            # Initialize agent status
            await self._initialize_agent_status()
            
            self.logger.info("Integration completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Integration failed: {e}")
            return False
    
    async def _discover_agents(self):
        """Discover existing agents in the Evolve platform"""
        self.logger.info("Discovering existing agents...")
        
        discovered_agents = {}
        
        # Model Innovation Agent
        try:
            from agents.model_innovation_agent import ModelInnovationAgent
            discovered_agents['model_innovation'] = ModelInnovationAgent()
            self.logger.info("âœ… ModelInnovationAgent discovered")
        except ImportError:
            self.logger.warning("âš ï¸ ModelInnovationAgent not available")
        
        # Strategy Research Agent
        try:
            from agents.strategy_research_agent import StrategyResearchAgent
            discovered_agents['strategy_research'] = StrategyResearchAgent()
            self.logger.info("âœ… StrategyResearchAgent discovered")
        except ImportError:
            self.logger.warning("âš ï¸ StrategyResearchAgent not available")
        
        # Sentiment Analyzer
        try:
            from trading.nlp.sentiment_analyzer import SentimentAnalyzer
            discovered_agents['sentiment_fetch'] = SentimentAnalyzer()
            self.logger.info("âœ… SentimentAnalyzer discovered")
        except ImportError:
            self.logger.warning("âš ï¸ SentimentAnalyzer not available")
        
        # Risk Manager
        try:
            from trading.risk.risk_manager import RiskManager
            discovered_agents['risk_management'] = RiskManager()
            self.logger.info("âœ… RiskManager discovered")
        except ImportError:
            self.logger.warning("âš ï¸ RiskManager not available")
        
        # Execution Agent
        try:
            from execution.execution_agent import ExecutionAgent
            discovered_agents['execution'] = ExecutionAgent()
            self.logger.info("âœ… ExecutionAgent discovered")
        except ImportError:
            self.logger.warning("âš ï¸ ExecutionAgent not available")
        
        # Explainer Agent
        try:
            from reporting.explainer_agent import ExplainerAgent
            discovered_agents['explanation'] = ExplainerAgent()
            self.logger.info("âœ… ExplainerAgent discovered")
        except ImportError:
            self.logger.warning("âš ï¸ ExplainerAgent not available")
        
        # Meta Controller (if exists)
        try:
            from meta.meta_controller import MetaControllerAgent
            discovered_agents['meta_control'] = MetaControllerAgent()
            self.logger.info("âœ… MetaControllerAgent discovered")
        except ImportError:
            self.logger.warning("âš ï¸ MetaControllerAgent not available")
        
        # Data Sync components
        try:
            from trading.data.data_manager import DataManager
            discovered_agents['data_sync'] = DataManager()
            self.logger.info("âœ… DataManager discovered")
        except ImportError:
            self.logger.warning("âš ï¸ DataManager not available")
        
        # Performance Analysis
        try:
            from trading.evaluation.performance_analyzer import PerformanceAnalyzer
            discovered_agents['performance_analysis'] = PerformanceAnalyzer()
            self.logger.info("âœ… PerformanceAnalyzer discovered")
        except ImportError:
            self.logger.warning("âš ï¸ PerformanceAnalyzer not available")
        
        # System Health Monitor
        try:
            from system.health_monitor import SystemHealthMonitor
            discovered_agents['system_health'] = SystemHealthMonitor()
            self.logger.info("âœ… SystemHealthMonitor discovered")
        except ImportError:
            self.logger.warning("âš ï¸ SystemHealthMonitor not available")
        
        # Update orchestrator agents
        self.orchestrator.agents.update(discovered_agents)
        
        self.logger.info(f"Discovered {len(discovered_agents)} agents")
        return discovered_agents
    
    async def _configure_agent_methods(self):
        """Configure agent methods for orchestrator compatibility"""
        self.logger.info("Configuring agent methods...")
        
        method_mappings = {
            'model_innovation': ['innovate_models', 'improve_models', 'generate_models'],
            'strategy_research': ['research_strategies', 'analyze_strategies', 'develop_strategies'],
            'sentiment_fetch': ['fetch_sentiment', 'analyze_sentiment', 'get_sentiment'],
            'risk_management': ['manage_risk', 'assess_risk', 'monitor_risk'],
            'execution': ['execute_orders', 'process_orders', 'trade'],
            'explanation': ['generate_explanations', 'explain_decisions', 'create_explanations'],
            'meta_control': ['control_system', 'manage_system', 'coordinate'],
            'data_sync': ['sync_data', 'update_data', 'refresh_data'],
            'performance_analysis': ['analyze_performance', 'evaluate_performance', 'assess_performance'],
            'system_health': ['check_health', 'monitor_health', 'assess_health']
        }
        
        for agent_name, agent in self.orchestrator.agents.items():
            if agent_name in method_mappings:
                # Check if agent has the required methods
                available_methods = []
                for method_name in method_mappings[agent_name]:
                    if hasattr(agent, method_name):
                        available_methods.append(method_name)
                
                if available_methods:
                    self.logger.info(f"Agent {agent_name} has methods: {available_methods}")
                else:
                    self.logger.warning(f"Agent {agent_name} has no compatible methods")
    
    async def _update_task_configs(self):
        """Update task configurations based on discovered agents"""
        self.logger.info("Updating task configurations...")
        
        # Update task configurations based on available agents
        for task_name, task in self.orchestrator.tasks.items():
            if task_name not in self.orchestrator.agents:
                # Disable tasks for unavailable agents
                task.enabled = False
                self.logger.info(f"Disabled task {task_name} - agent not available")
            else:
                # Enable and configure tasks for available agents
                task.enabled = True
                self.logger.info(f"Enabled task {task_name}")
    
    async def _initialize_agent_status(self):
        """Initialize agent status for discovered agents"""
        self.logger.info("Initializing agent status...")
        
        for agent_name in self.orchestrator.agents.keys():
            if agent_name not in self.orchestrator.agent_status:
                self.orchestrator.agent_status[agent_name] = self.orchestrator.AgentStatus(
                    agent_name=agent_name
                )
                self.logger.info(f"Initialized status for {agent_name}")
    
    async def start_orchestrator(self):
        """Start the integrated orchestrator"""
        if not self.orchestrator:
            await self.integrate_with_existing_agents()
        
        if self.orchestrator:
            self.logger.info("Starting integrated Task Orchestrator...")
            await self.orchestrator.start()
            return True
        else:
            self.logger.error("Failed to create orchestrator")
            return False
    
    async def stop_orchestrator(self):
        """Stop the orchestrator"""
        if self.orchestrator and self.orchestrator.is_running:
            self.logger.info("Stopping Task Orchestrator...")
            await self.orchestrator.stop()
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status"""
        if not self.orchestrator:
            return {
                "status": "not_initialized",
                "message": "Orchestrator not initialized"
            }
        
        status = self.orchestrator.get_system_status()
        
        return {
            "status": "integrated",
            "total_agents": len(self.orchestrator.agents),
            "enabled_tasks": len([t for t in self.orchestrator.tasks.values() if t.enabled]),
            "overall_health": status['performance_metrics']['overall_health'],
            "agent_status": status['agent_status']
        }


async def main():
    """Main integration function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrate Task Orchestrator with Evolve Platform')
    parser.add_argument('--config', type=str, default='config/task_schedule.yaml',
                       help='Path to orchestrator configuration')
    parser.add_argument('--start', action='store_true', help='Start orchestrator after integration')
    parser.add_argument('--monitor', action='store_true', help='Enable monitoring mode')
    
    args = parser.parse_args()
    
    # Create integration instance
    integration = EvolveOrchestratorIntegration(args.config)
    
    try:
        # Perform integration
        success = await integration.integrate_with_existing_agents()
        
        if success:
            print("âœ… Integration completed successfully")
            
            # Show integration status
            status = integration.get_integration_status()
            print(f"\nIntegration Status:")
            print(f"  Total Agents: {status['total_agents']}")
            print(f"  Enabled Tasks: {status['enabled_tasks']}")
            print(f"  Overall Health: {status['overall_health']:.2f}")
            
            if args.start:
                print("\nðŸš€ Starting Task Orchestrator...")
                await integration.start_orchestrator()
                
                if args.monitor:
                    print("ðŸ“Š Monitoring mode enabled - Press Ctrl+C to stop")
                    try:
                        while True:
                            await asyncio.sleep(30)
                            status = integration.get_integration_status()
                            print(f"Health: {status['overall_health']:.2f} | "
                                  f"Agents: {status['total_agents']} | "
                                  f"Tasks: {status['enabled_tasks']}")
                    except KeyboardInterrupt:
                        print("\nâ¹ï¸ Stopping orchestrator...")
                else:
                    # Keep running
                    await asyncio.sleep(3600)  # Run for 1 hour
                
                await integration.stop_orchestrator()
        else:
            print("âŒ Integration failed")
            return 1
            
    except KeyboardInterrupt:
        print("\nâ¹ï¸ Integration interrupted")
        await integration.stop_orchestrator()
    except Exception as e:
        print(f"âŒ Error during integration: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
