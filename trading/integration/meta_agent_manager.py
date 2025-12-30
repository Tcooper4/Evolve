"""
Meta Agent Manager

Centralized management for meta agents that provide system-wide capabilities
including integration testing, log visualization, documentation analytics,
and security automation.
"""

import asyncio
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class MetaAgentStatus(Enum):
    """Status enumeration for meta agents."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class MetaAgentResult:
    """Result from meta agent execution."""

    agent_name: str
    status: MetaAgentStatus
    result: Any
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        return result


class MetaAgentManager:
    """
    Centralized manager for meta agents providing system-wide capabilities.

    Integrates:
    - Integration testing
    - Log visualization and analysis
    - Documentation generation and analytics
    - Security automation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the meta agent manager."""
        self.config = config or {}
        self.agents = {}
        self.agent_status = {}
        self.execution_history = []
        self.logger = logging.getLogger(f"{__name__}.MetaAgentManager")

        # Initialize meta agents
        self._initialize_meta_agents()

        self.logger.info("MetaAgentManager initialized successfully")

    def _initialize_meta_agents(self):
        """Initialize all available meta agents."""
        try:
            # Integration Test Handler
            try:
                from trading.meta_agents.integration_test_handler import (
                    IntegrationTestHandler,
                )

                self.agents["integration_test"] = IntegrationTestHandler()
                self.agent_status["integration_test"] = MetaAgentStatus.IDLE
                self.logger.info("âœ… IntegrationTestHandler initialized")
            except ImportError as e:
                self.logger.warning(f"âš ï¸ IntegrationTestHandler not available: {e}")

            # Log Visualization Handler
            try:
                from trading.meta_agents.log_visualization_handler import (
                    LogVisualizationHandler,
                )

                self.agents["log_visualization"] = LogVisualizationHandler()
                self.agent_status["log_visualization"] = MetaAgentStatus.IDLE
                self.logger.info("âœ… LogVisualizationHandler initialized")
            except ImportError as e:
                self.logger.warning(
                    f"âš ï¸ LogVisualizationHandler not available: {e}"
                )

            # Documentation Analytics
            try:
                from trading.meta_agents.documentation_analytics import (
                    DocumentationAnalytics,
                )

                self.agents["documentation_analytics"] = DocumentationAnalytics()
                self.agent_status["documentation_analytics"] = MetaAgentStatus.IDLE
                self.logger.info("âœ… DocumentationAnalytics initialized")
            except ImportError as e:
                self.logger.warning(f"âš ï¸ DocumentationAnalytics not available: {e}")

            # Automation Security
            try:
                from trading.meta_agents.automation_security import AutomationSecurity

                self.agents["automation_security"] = AutomationSecurity()
                self.agent_status["automation_security"] = MetaAgentStatus.IDLE
                self.logger.info("âœ… AutomationSecurity initialized")
            except ImportError as e:
                self.logger.warning(f"âš ï¸ AutomationSecurity not available: {e}")

            self.logger.info(f"Initialized {len(self.agents)} meta agents")

        except Exception as e:
            self.logger.error(f"Failed to initialize meta agents: {e}")

    async def execute_agent(self, agent_name: str, **kwargs) -> MetaAgentResult:
        """Execute a specific meta agent."""
        if agent_name not in self.agents:
            return MetaAgentResult(
                agent_name=agent_name,
                status=MetaAgentStatus.ERROR,
                result=None,
                error_message=f"Agent '{agent_name}' not found",
            )

        start_time = datetime.now()
        self.agent_status[agent_name] = MetaAgentStatus.RUNNING

        try:
            self.logger.info(f"Executing meta agent: {agent_name}")

            agent = self.agents[agent_name]

            # Execute agent based on type
            if agent_name == "integration_test":
                result = await self._execute_integration_test(agent, **kwargs)
            elif agent_name == "log_visualization":
                result = await self._execute_log_visualization(agent, **kwargs)
            elif agent_name == "documentation_analytics":
                result = await self._execute_documentation_analytics(agent, **kwargs)
            elif agent_name == "automation_security":
                result = await self._execute_automation_security(agent, **kwargs)
            else:
                # Generic execution for unknown agents
                if hasattr(agent, "execute"):
                    result = await agent.execute(**kwargs)
                elif hasattr(agent, "run"):
                    result = await agent.run(**kwargs)
                else:
                    result = agent(**kwargs) if callable(agent) else None

            execution_time = (datetime.now() - start_time).total_seconds()

            meta_result = MetaAgentResult(
                agent_name=agent_name,
                status=MetaAgentStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                timestamp=datetime.now(),
            )

            self.agent_status[agent_name] = MetaAgentStatus.COMPLETED
            self.execution_history.append(meta_result)

            self.logger.info(
                f"âœ… Meta agent '{agent_name}' completed in {execution_time:.2f}s"
            )
            return meta_result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"âŒ Meta agent '{agent_name}' failed: {e}")

            meta_result = MetaAgentResult(
                agent_name=agent_name,
                status=MetaAgentStatus.ERROR,
                result=None,
                error_message=str(e),
                execution_time=execution_time,
                timestamp=datetime.now(),
            )

            self.agent_status[agent_name] = MetaAgentStatus.ERROR
            self.execution_history.append(meta_result)

            return meta_result

    async def _execute_integration_test(self, agent, **kwargs) -> Dict[str, Any]:
        """Execute integration test handler."""
        try:
            if hasattr(agent, "run_full_test_suite"):
                return await agent.run_full_test_suite(**kwargs)
            elif hasattr(agent, "run_tests"):
                return await agent.run_tests(**kwargs)
            else:
                return {"status": "no_test_method_found"}
        except Exception as e:
            self.logger.error(f"Integration test execution failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _execute_log_visualization(self, agent, **kwargs) -> Dict[str, Any]:
        """Execute log visualization handler."""
        try:
            if hasattr(agent, "analyze_logs"):
                return await agent.analyze_logs(**kwargs)
            elif hasattr(agent, "visualize_logs"):
                return await agent.visualize_logs(**kwargs)
            else:
                return {"status": "no_visualization_method_found"}
        except Exception as e:
            self.logger.error(f"Log visualization execution failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _execute_documentation_analytics(self, agent, **kwargs) -> Dict[str, Any]:
        """Execute documentation analytics."""
        try:
            if hasattr(agent, "generate_docs"):
                return await agent.generate_docs(**kwargs)
            elif hasattr(agent, "analyze_documentation"):
                return await agent.analyze_documentation(**kwargs)
            else:
                return {"status": "no_documentation_method_found"}
        except Exception as e:
            self.logger.error(f"Documentation analytics execution failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _execute_automation_security(self, agent, **kwargs) -> Dict[str, Any]:
        """Execute automation security."""
        try:
            if hasattr(agent, "run_security_checks"):
                return await agent.run_security_checks(**kwargs)
            elif hasattr(agent, "check_security"):
                return await agent.check_security(**kwargs)
            else:
                return {"status": "no_security_method_found"}
        except Exception as e:
            self.logger.error(f"Automation security execution failed: {e}")
            return {"status": "error", "error": str(e)}

    async def run_system_integration_tests(self) -> MetaAgentResult:
        """Run comprehensive system integration tests."""
        return await self.execute_agent("integration_test", test_type="full_system")

    async def analyze_system_logs(self) -> MetaAgentResult:
        """Analyze system logs for insights."""
        return await self.execute_agent(
            "log_visualization", analysis_type="comprehensive"
        )

    async def generate_system_documentation(self) -> MetaAgentResult:
        """Generate comprehensive system documentation."""
        return await self.execute_agent(
            "documentation_analytics", doc_type="comprehensive"
        )

    async def run_security_audit(self) -> MetaAgentResult:
        """Run security audit and checks."""
        return await self.execute_agent(
            "automation_security", audit_type="comprehensive"
        )

    async def run_all_meta_agents(self) -> List[MetaAgentResult]:
        """Run all available meta agents."""
        results = []

        for agent_name in self.agents.keys():
            result = await self.execute_agent(agent_name)
            results.append(result)

        return results

    def get_agent_status(self, agent_name: str) -> Optional[MetaAgentStatus]:
        """Get status of a specific agent."""
        return self.agent_status.get(agent_name)

    def get_all_agent_statuses(self) -> Dict[str, MetaAgentStatus]:
        """Get status of all agents."""
        return self.agent_status.copy()

    def get_execution_history(
        self, agent_name: Optional[str] = None
    ) -> List[MetaAgentResult]:
        """Get execution history."""
        if agent_name:
            return [
                result
                for result in self.execution_history
                if result.agent_name == agent_name
            ]
        return self.execution_history.copy()

    def get_available_agents(self) -> List[str]:
        """Get list of available agents."""
        return list(self.agents.keys())

    def get_agent_info(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific agent."""
        if agent_name not in self.agents:
            return None

        agent = self.agents[agent_name]
        return {
            "name": agent_name,
            "type": type(agent).__name__,
            "status": self.agent_status.get(agent_name),
            "methods": [method for method in dir(agent) if not method.startswith("_")],
            "doc": getattr(agent, "__doc__", "No documentation available"),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all meta agents."""
        health_status = {
            "overall_health": "healthy",
            "agents": {},
            "timestamp": datetime.now().isoformat(),
        }

        for agent_name in self.agents.keys():
            try:
                # Quick health check for each agent
                agent = self.agents[agent_name]
                if hasattr(agent, "health_check"):
                    agent_health = await agent.health_check()
                else:
                    agent_health = {
                        "status": "unknown",
                        "message": "No health check method",
                    }

                health_status["agents"][agent_name] = agent_health

                if agent_health.get("status") == "error":
                    health_status["overall_health"] = "degraded"

            except Exception as e:
                health_status["agents"][agent_name] = {
                    "status": "error",
                    "message": str(e),
                }
                health_status["overall_health"] = "degraded"

        return health_status


# Convenience functions for easy access
async def run_system_integration_tests() -> MetaAgentResult:
    """Run system integration tests."""
    manager = MetaAgentManager()
    return await manager.run_system_integration_tests()


async def analyze_system_logs() -> MetaAgentResult:
    """Analyze system logs."""
    manager = MetaAgentManager()
    return await manager.analyze_system_logs()


async def generate_system_documentation() -> MetaAgentResult:
    """Generate system documentation."""
    manager = MetaAgentManager()
    return await manager.generate_system_documentation()


async def run_security_audit() -> MetaAgentResult:
    """Run security audit."""
    manager = MetaAgentManager()
    return await manager.run_security_audit()


if __name__ == "__main__":
    # Demo usage
    async def demo():
        print("ðŸ”§ Meta Agent Manager Demo")
        print("=" * 50)

        manager = MetaAgentManager()

        print(f"Available agents: {manager.get_available_agents()}")
        print(f"Agent statuses: {manager.get_all_agent_statuses()}")

        # Run integration tests
        print("\nðŸ§ª Running system integration tests...")
        result = await manager.run_system_integration_tests()
        print(f"Result: {result.status} - {result.result}")

        # Health check
        print("\nðŸ¥ Running health check...")
        health = await manager.health_check()
        print(f"Overall health: {health['overall_health']}")

        print("\nâœ… Demo completed!")

    asyncio.run(demo())
