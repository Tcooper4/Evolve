"""
Task Orchestrator System Integration

This module provides seamless integration between the TaskOrchestrator
and the existing Evolve trading platform components.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from core.task_orchestrator import TaskOrchestrator

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class EvolveSystemIntegration:
    """
    System integration for TaskOrchestrator with Evolve platform
    """

    def __init__(self, config_path: str = "config/task_schedule.yaml"):
        self.config_path = config_path
        self.orchestrator = None
        self.logger = logging.getLogger(__name__)

        # Integration status
        self.integration_status = {
            "initialized": False,
            "agents_connected": 0,
            "components_connected": 0,
            "last_update": None,
        }

    async def initialize_integration(self) -> bool:
        """Initialize the system integration"""
        try:
            self.logger.info("Initializing Task Orchestrator system integration...")

            # Create orchestrator
            self.orchestrator = TaskOrchestrator(self.config_path)

            # Connect with existing system components
            await self._connect_system_components()

            # Initialize agent connections
            await self._connect_agents()

            # Setup monitoring and health checks
            await self._setup_monitoring()

            self.integration_status["initialized"] = True
            self.integration_status["last_update"] = datetime.now().isoformat()

            self.logger.info("System integration initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize integration: {e}")
            return False

    async def _connect_system_components(self):
        """Connect with existing system components"""
        self.logger.info("Connecting with system components...")

        components_connected = 0

        # Connect with data management
        try:
            from trading.data.data_manager import DataManager

            if hasattr(self.orchestrator, "data_manager"):
                self.orchestrator.data_manager = DataManager()
                components_connected += 1
                self.logger.info("âœ… Connected with DataManager")
        except ImportError:
            self.logger.warning("âš ï¸ DataManager not available")

        # Connect with risk management
        try:
            from trading.risk.risk_manager import RiskManager

            if hasattr(self.orchestrator, "risk_manager"):
                self.orchestrator.risk_manager = RiskManager()
                components_connected += 1
                self.logger.info("âœ… Connected with RiskManager")
        except ImportError:
            self.logger.warning("âš ï¸ RiskManager not available")

        # Connect with portfolio management
        try:
            from trading.portfolio.portfolio_manager import PortfolioManager

            if hasattr(self.orchestrator, "portfolio_manager"):
                self.orchestrator.portfolio_manager = PortfolioManager()
                components_connected += 1
                self.logger.info("âœ… Connected with PortfolioManager")
        except ImportError:
            self.logger.warning("âš ï¸ PortfolioManager not available")

        # Connect with market analysis
        try:
            from market_analysis.market_analyzer import MarketAnalyzer

            if hasattr(self.orchestrator, "market_analyzer"):
                self.orchestrator.market_analyzer = MarketAnalyzer()
                components_connected += 1
                self.logger.info("âœ… Connected with MarketAnalyzer")
        except ImportError:
            self.logger.warning("âš ï¸ MarketAnalyzer not available")

        # Connect with reporting system
        try:
            from reporting.report_generator import ReportGenerator

            if hasattr(self.orchestrator, "report_generator"):
                self.orchestrator.report_generator = ReportGenerator()
                components_connected += 1
                self.logger.info("âœ… Connected with ReportGenerator")
        except ImportError:
            self.logger.warning("âš ï¸ ReportGenerator not available")

        # Connect with caching system
        try:
            from utils.cache_utils import CacheManager

            if hasattr(self.orchestrator, "cache_manager"):
                self.orchestrator.cache_manager = CacheManager()
                components_connected += 1
                self.logger.info("âœ… Connected with CacheManager")
        except ImportError:
            self.logger.warning("âš ï¸ CacheManager not available")

        self.integration_status["components_connected"] = components_connected
        self.logger.info(f"Connected with {components_connected} system components")

    async def _connect_agents(self):
        """Connect with existing agents"""
        self.logger.info("Connecting with agents...")

        agents_connected = 0

        # Connect with existing agents
        agent_connections = [
            ("model_innovation", "agents.model_innovation_agent.ModelInnovationAgent"),
            (
                "strategy_research",
                "agents.strategy_research_agent.StrategyResearchAgent",
            ),
            ("sentiment_fetch", "trading.nlp.sentiment_analyzer.SentimentAnalyzer"),
            ("risk_management", "trading.risk.risk_manager.RiskManager"),
            ("execution", "execution.execution_agent.ExecutionAgent"),
            ("explanation", "reporting.explainer_agent.ExplainerAgent"),
            ("data_sync", "trading.data.data_manager.DataManager"),
            (
                "performance_analysis",
                "trading.evaluation.performance_analyzer.PerformanceAnalyzer",
            ),
        ]

        for agent_name, import_path in agent_connections:
            try:
                module_name, class_name = import_path.rsplit(".", 1)
                module = __import__(module_name, fromlist=[class_name])
                agent_class = getattr(module, class_name)
                agent_instance = agent_class()

                self.orchestrator.agents[agent_name] = agent_instance
                agents_connected += 1
                self.logger.info(f"âœ… Connected with {agent_name}")

            except ImportError:
                self.logger.warning(f"âš ï¸ {agent_name} not available")
            except Exception as e:
                self.logger.warning(f"âš ï¸ Failed to connect {agent_name}: {e}")

        # Initialize agent status
        for agent_name in self.orchestrator.agents.keys():
            if agent_name not in self.orchestrator.agent_status:
                self.orchestrator.agent_status[
                    agent_name
                ] = self.orchestrator.AgentStatus(agent_name=agent_name)

        self.integration_status["agents_connected"] = agents_connected
        self.logger.info(f"Connected with {agents_connected} agents")

    async def _setup_monitoring(self):
        """Setup monitoring and health checks"""
        self.logger.info("Setting up monitoring...")

        # Setup health monitoring
        try:
            from system.health_monitor import SystemHealthMonitor

            self.orchestrator.health_monitor = SystemHealthMonitor()
            self.logger.info("âœ… Health monitoring setup")
        except ImportError:
            self.logger.warning("âš ï¸ Health monitoring not available")

        # Setup performance monitoring
        try:
            from system.performance_monitor import PerformanceMonitor

            self.orchestrator.performance_monitor = PerformanceMonitor()
            self.logger.info("âœ… Performance monitoring setup")
        except ImportError:
            self.logger.warning("âš ï¸ Performance monitoring not available")

    async def start_integrated_system(self) -> bool:
        """Start the integrated system"""
        if not self.integration_status["initialized"]:
            success = await self.initialize_integration()
            if not success:
                return False

        try:
            self.logger.info("Starting integrated system...")
            await self.orchestrator.start()

            # Start additional system components
            await self._start_system_components()

            self.logger.info("Integrated system started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start integrated system: {e}")
            return False

    async def _start_system_components(self):
        """Start additional system components"""
        self.logger.info("Starting system components...")

        # Start data feeds
        if hasattr(self.orchestrator, "data_manager"):
            try:
                await self.orchestrator.data_manager.start()
                self.logger.info("âœ… Data manager started")
            except Exception as e:
                self.logger.warning(f"âš ï¸ Failed to start data manager: {e}")

        # Start risk monitoring
        if hasattr(self.orchestrator, "risk_manager"):
            try:
                await self.orchestrator.risk_manager.start()
                self.logger.info("âœ… Risk manager started")
            except Exception as e:
                self.logger.warning(f"âš ï¸ Failed to start risk manager: {e}")

        # Start portfolio monitoring
        if hasattr(self.orchestrator, "portfolio_manager"):
            try:
                await self.orchestrator.portfolio_manager.start()
                self.logger.info("âœ… Portfolio manager started")
            except Exception as e:
                self.logger.warning(f"âš ï¸ Failed to start portfolio manager: {e}")

    async def stop_integrated_system(self):
        """Stop the integrated system"""
        self.logger.info("Stopping integrated system...")

        try:
            # Stop orchestrator
            if self.orchestrator and self.orchestrator.is_running:
                await self.orchestrator.stop()

            # Stop system components
            await self._stop_system_components()

            self.logger.info("Integrated system stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping integrated system: {e}")

    async def _stop_system_components(self):
        """Stop system components"""
        self.logger.info("Stopping system components...")

        # Stop data feeds
        if hasattr(self.orchestrator, "data_manager"):
            try:
                await self.orchestrator.data_manager.stop()
                self.logger.info("âœ… Data manager stopped")
            except Exception as e:
                self.logger.warning(f"âš ï¸ Failed to stop data manager: {e}")

        # Stop risk monitoring
        if hasattr(self.orchestrator, "risk_manager"):
            try:
                await self.orchestrator.risk_manager.stop()
                self.logger.info("âœ… Risk manager stopped")
            except Exception as e:
                self.logger.warning(f"âš ï¸ Failed to stop risk manager: {e}")

        # Stop portfolio monitoring
        if hasattr(self.orchestrator, "portfolio_manager"):
            try:
                await self.orchestrator.portfolio_manager.stop()
                self.logger.info("âœ… Portfolio manager stopped")
            except Exception as e:
                self.logger.warning(f"âš ï¸ Failed to stop portfolio manager: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        if not self.orchestrator:
            return {
                "status": "not_initialized",
                "integration_status": self.integration_status,
            }

        # Get orchestrator status
        orchestrator_status = self.orchestrator.get_system_status()

        # Get component status
        component_status = {}
        for component_name in [
            "data_manager",
            "risk_manager",
            "portfolio_manager",
            "market_analyzer",
            "report_generator",
            "cache_manager",
        ]:
            if hasattr(self.orchestrator, component_name):
                component = getattr(self.orchestrator, component_name)
                try:
                    if hasattr(component, "get_status"):
                        component_status[component_name] = component.get_status()
                    else:
                        component_status[component_name] = {"status": "running"}
                except Exception as e:
                    component_status[component_name] = {
                        "status": "error",
                        "error": str(e),
                    }

        return {
            "status": "running",
            "integration_status": self.integration_status,
            "orchestrator_status": orchestrator_status,
            "component_status": component_status,
            "timestamp": datetime.now().isoformat(),
        }

    async def execute_system_command(
        self, command: str, parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a system command through the orchestrator"""
        if not self.orchestrator:
            return {"status": "error", "message": "System not initialized"}

        try:
            # Map commands to tasks
            command_mapping = {
                "forecast": "model_innovation",
                "analyze": "strategy_research",
                "sentiment": "sentiment_fetch",
                "risk": "risk_management",
                "execute": "execution",
                "explain": "explanation",
                "sync": "data_sync",
                "performance": "performance_analysis",
            }

            # Find matching task
            task_name = None
            for cmd, task in command_mapping.items():
                if cmd in command.lower():
                    task_name = task
                    break

            if task_name and task_name in self.orchestrator.tasks:
                result = await self.orchestrator.execute_task_now(
                    task_name, parameters or {}
                )
                return {
                    "status": "success",
                    "command": command,
                    "task": task_name,
                    "result": result,
                }
            else:
                return {
                    "status": "error",
                    "message": f"No matching task found for command: {command}",
                }

        except Exception as e:
            return {"status": "error", "command": command, "error": str(e)}


# Convenience functions for easy integration
async def create_integrated_system(
    config_path: str = "config/task_schedule.yaml",
) -> EvolveSystemIntegration:
    """Create an integrated system instance"""
    integration = EvolveSystemIntegration(config_path)
    await integration.initialize_integration()
    return integration


async def start_integrated_system(
    config_path: str = "config/task_schedule.yaml",
) -> EvolveSystemIntegration:
    """Create and start an integrated system"""
    integration = await create_integrated_system(config_path)
    await integration.start_integrated_system()
    return integration


def get_system_integration_status() -> Dict[str, Any]:
    """Get system integration status without creating an instance"""
    try:
        # Check if orchestrator config exists
        config_path = "config/task_schedule.yaml"
        if not os.path.exists(config_path):
            return {
                "status": "not_configured",
                "message": "Task orchestrator configuration not found",
            }

        # Check if core orchestrator is available
        try:
            from core.task_orchestrator import TaskOrchestrator

            orchestrator = TaskOrchestrator(config_path)
            status = orchestrator.get_system_status()

            return {
                "status": "available",
                "total_tasks": status["total_tasks"],
                "enabled_tasks": status["enabled_tasks"],
                "overall_health": status["performance_metrics"]["overall_health"],
            }

        except ImportError:
            return {
                "status": "not_available",
                "message": "Task orchestrator not available",
            }

    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    # Example usage
    async def main():
        integration = await start_integrated_system()

        try:
            # Keep running
            await asyncio.sleep(3600)  # Run for 1 hour
        except KeyboardInterrupt:
            print("Stopping integrated system...")
        finally:
            await integration.stop_integrated_system()

    asyncio.run(main())
