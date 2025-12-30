"""
Task Orchestrator Example

This example demonstrates how to use the TaskOrchestrator to manage and schedule
all agents in the Evolve trading platform.

Features demonstrated:
- Creating and configuring the orchestrator
- Starting and stopping the system
- Monitoring task execution
- Handling errors and performance issues
- Custom task scheduling
- Real-time status monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

# Import the TaskOrchestrator
from core.task_orchestrator import (
    create_task_orchestrator,
    start_orchestrator,
)

# Import platform agents (these would be your actual agent implementations)
try:
    pass

    AGENTS_AVAILABLE = True
except ImportError:
    print("Note: Some agents not available, using mock implementations")
    AGENTS_AVAILABLE = False


class MockAgent:
    """Mock agent for demonstration purposes"""

    def __init__(self, name: str, success_rate: float = 0.9):
        self.name = name
        self.success_rate = success_rate
        self.execution_count = 0

    async def innovate_models(self, **kwargs):
        """Mock model innovation"""
        self.execution_count += 1
        await asyncio.sleep(2)  # Simulate work

        if self.success_rate > 0.8:
            return {
                "status": "success",
                "models_improved": 3,
                "performance_gain": 0.05,
                "execution_time": 2.0,
            }
        else:
            raise Exception("Model innovation failed")

    async def research_strategies(self, **kwargs):
        """Mock strategy research"""
        self.execution_count += 1
        await asyncio.sleep(3)

        if self.success_rate > 0.7:
            return {
                "status": "success",
                "strategies_analyzed": 5,
                "new_opportunities": 2,
                "market_insights": ["bullish", "volatile"],
            }
        else:
            raise Exception("Strategy research failed")

    async def fetch_sentiment(self, **kwargs):
        """Mock sentiment fetching"""
        self.execution_count += 1
        await asyncio.sleep(1)

        if self.success_rate > 0.9:
            return {
                "status": "success",
                "sentiment_score": 0.65,
                "sources_analyzed": 100,
                "confidence": 0.85,
            }
        else:
            raise Exception("Sentiment fetch failed")

    async def control_system(self, **kwargs):
        """Mock system control"""
        self.execution_count += 1
        await asyncio.sleep(0.5)

        if self.success_rate > 0.95:
            return {
                "status": "success",
                "system_health": 0.92,
                "active_agents": 7,
                "resource_usage": 0.45,
            }
        else:
            raise Exception("System control failed")

    async def manage_risk(self, **kwargs):
        """Mock risk management"""
        self.execution_count += 1
        await asyncio.sleep(1.5)

        if self.success_rate > 0.85:
            return {
                "status": "success",
                "risk_score": 0.23,
                "positions_monitored": 15,
                "alerts_generated": 2,
            }
        else:
            raise Exception("Risk management failed")

    async def execute_orders(self, **kwargs):
        """Mock order execution"""
        self.execution_count += 1
        await asyncio.sleep(0.5)

        if self.success_rate > 0.98:
            return {
                "status": "success",
                "orders_executed": 3,
                "total_volume": 15000,
                "slippage": 0.001,
            }
        else:
            raise Exception("Order execution failed")

    async def generate_explanations(self, **kwargs):
        """Mock explanation generation"""
        self.execution_count += 1
        await asyncio.sleep(2)

        if self.success_rate > 0.8:
            return {
                "status": "success",
                "explanations_generated": 5,
                "confidence_scores": [0.85, 0.92, 0.78, 0.89, 0.91],
                "insights": ["market_trend", "volatility_increase", "sentiment_shift"],
            }
        else:
            raise Exception("Explanation generation failed")

    async def check_health(self, **kwargs):
        """Mock health check"""
        self.execution_count += 1
        await asyncio.sleep(0.5)

        return {
            "status": "success",
            "health_score": 0.95,
            "active_processes": 12,
            "memory_usage": 0.65,
            "cpu_usage": 0.45,
        }

    async def sync_data(self, **kwargs):
        """Mock data synchronization"""
        self.execution_count += 1
        await asyncio.sleep(1)

        if self.success_rate > 0.9:
            return {
                "status": "success",
                "data_sources_synced": 3,
                "records_updated": 1500,
                "cache_cleared": True,
            }
        else:
            raise Exception("Data sync failed")

    async def analyze_performance(self, **kwargs):
        """Mock performance analysis"""
        self.execution_count += 1
        await asyncio.sleep(4)

        if self.success_rate > 0.7:
            return {
                "status": "success",
                "metrics_analyzed": 8,
                "performance_score": 0.87,
                "recommendations": ["increase_position_size", "adjust_risk_parameters"],
            }
        else:
            raise Exception("Performance analysis failed")


class TaskOrchestratorExample:
    """Example demonstrating TaskOrchestrator usage"""

    def __init__(self, config_path: str = "config/task_schedule.yaml"):
        self.config_path = config_path
        self.orchestrator = None
        self.mock_agents = {}

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def setup_mock_agents(self):
        """Setup mock agents for demonstration"""
        if not AGENTS_AVAILABLE:
            self.mock_agents = {
                "model_innovation": MockAgent("ModelInnovation", 0.95),
                "strategy_research": MockAgent("StrategyResearch", 0.88),
                "sentiment_fetch": MockAgent("SentimentFetch", 0.92),
                "meta_control": MockAgent("MetaControl", 0.98),
                "risk_management": MockAgent("RiskManagement", 0.90),
                "execution": MockAgent("Execution", 0.99),
                "explanation": MockAgent("Explanation", 0.85),
                "system_health": MockAgent("SystemHealth", 1.0),
                "data_sync": MockAgent("DataSync", 0.93),
                "performance_analysis": MockAgent("PerformanceAnalysis", 0.82),
            }

    async def create_orchestrator(self):
        """Create and configure the task orchestrator"""
        self.logger.info("Creating Task Orchestrator...")

        # Create orchestrator
        self.orchestrator = create_task_orchestrator(self.config_path)

        # Setup mock agents if needed
        if not AGENTS_AVAILABLE:
            self.setup_mock_agents()
            self.orchestrator.agents = self.mock_agents

            # Initialize agent status for mock agents
            for agent_name in self.mock_agents.keys():
                self.orchestrator.agent_status[
                    agent_name
                ] = self.orchestrator.AgentStatus(agent_name=agent_name)

        self.logger.info(
            f"Orchestrator created with {len(self.orchestrator.tasks)} tasks"
        )

    async def start_orchestrator(self):
        """Start the orchestrator"""
        self.logger.info("Starting Task Orchestrator...")
        await self.orchestrator.start()
        self.logger.info("Orchestrator started successfully")

    async def stop_orchestrator(self):
        """Stop the orchestrator"""
        if self.orchestrator and self.orchestrator.is_running:
            self.logger.info("Stopping Task Orchestrator...")
            await self.orchestrator.stop()
            self.logger.info("Orchestrator stopped successfully")

    async def monitor_execution(self, duration_minutes: int = 5):
        """Monitor orchestrator execution for a specified duration"""
        self.logger.info(f"Monitoring execution for {duration_minutes} minutes...")

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)

        while datetime.now() < end_time:
            # Get system status
            status = self.orchestrator.get_system_status()

            # Print status summary
            print(f"\n{'=' * 60}")
            print(f"System Status - {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'=' * 60}")
            print(f"Orchestrator Running: {status['orchestrator_running']}")
            print(f"Total Tasks: {status['total_tasks']}")
            print(f"Enabled Tasks: {status['enabled_tasks']}")
            print(f"Running Tasks: {status['running_tasks']}")
            print(
                f"Overall Health: {status['performance_metrics']['overall_health']:.2f}"
            )

            # Print agent status
            print(f"\nAgent Status:")
            print(
                f"{'Agent':<20} {'Running':<8} {'Health':<6} {'Success':<7} {'Failures':<8}"
            )
            print(f"{'-' * 50}")

            for agent_name, agent_status in status["agent_status"].items():
                print(
                    f"{agent_name:<20} {str(agent_status['is_running']):<8} "
                    f"{agent_status['health_score']:<6.2f} {agent_status['success_count']:<7} "
                    f"{agent_status['failure_count']:<8}"
                )

            # Print recent executions
            recent_executions = status["recent_executions"]
            if recent_executions:
                print(f"\nRecent Executions:")
                print(f"{'Task':<20} {'Status':<10} {'Duration':<10} {'Time':<20}")
                print(f"{'-' * 60}")

                for execution in recent_executions[-5:]:  # Last 5 executions
                    duration = execution.get("duration_seconds", 0)
                    print(
                        f"{execution['task_name']:<20} {execution['status']:<10} "
                        f"{duration:<10.2f} {execution['start_time'][11:19]}"
                    )

            # Wait before next status check
            await asyncio.sleep(30)  # Check every 30 seconds

    async def execute_specific_task(
        self, task_name: str, parameters: Dict[str, Any] = None
    ):
        """Execute a specific task immediately"""
        self.logger.info(f"Executing task: {task_name}")

        try:
            result = await self.orchestrator.execute_task_now(
                task_name, parameters or {}
            )
            self.logger.info(f"Task execution result: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to execute task {task_name}: {e}")
            return None

    async def demonstrate_task_management(self):
        """Demonstrate task management features"""
        self.logger.info("Demonstrating task management features...")

        # Get task status
        for task_name in ["model_innovation", "execution", "risk_management"]:
            if task_name in self.orchestrator.tasks:
                status = self.orchestrator.get_task_status(task_name)
                print(f"\nTask Status for {task_name}:")
                print(f"  Enabled: {status['enabled']}")
                print(f"  Interval: {status['interval_minutes']} minutes")
                print(f"  Priority: {status['priority']}")
                print(f"  Success Rate: {status['success_rate']:.2f}")
                print(f"  Error Count: {status['error_count']}")

        # Update task configuration
        self.logger.info("Updating task configuration...")
        self.orchestrator.update_task_config(
            "model_innovation",
            {
                "interval_minutes": 30,  # Change from daily to every 30 minutes
                "priority": "high",
            },
        )

        # Execute a specific task
        await self.execute_specific_task("system_health")

    async def demonstrate_error_handling(self):
        """Demonstrate error handling capabilities"""
        self.logger.info("Demonstrating error handling...")

        # Create a task that will fail
        if not AGENTS_AVAILABLE:
            # Create a failing mock agent
            failing_agent = MockAgent("FailingAgent", 0.0)  # 0% success rate
            self.orchestrator.agents["failing_task"] = failing_agent
            self.orchestrator.agent_status[
                "failing_task"
            ] = self.orchestrator.AgentStatus(agent_name="failing_task")

            # Execute the failing task
            await self.execute_specific_task("failing_task")

            # Check error handling
            status = self.orchestrator.get_task_status("failing_task")
            print(f"\nFailing task error count: {status['error_count']}")

    async def demonstrate_performance_monitoring(self):
        """Demonstrate performance monitoring"""
        self.logger.info("Demonstrating performance monitoring...")

        # Export status report
        report_path = self.orchestrator.export_status_report()
        print(f"\nStatus report exported to: {report_path}")

        # Get performance metrics
        status = self.orchestrator.get_system_status()
        print(f"\nPerformance Metrics:")
        print(
            f"  Overall Health: {status['performance_metrics']['overall_health']:.2f}"
        )
        print(f"  Success Rates: {status['performance_metrics']['success_rates']}")
        print(f"  Error Counts: {status['performance_metrics']['error_counts']}")

    async def run_comprehensive_demo(self, duration_minutes: int = 10):
        """Run a comprehensive demonstration"""
        self.logger.info("Starting comprehensive Task Orchestrator demonstration...")

        try:
            # Create orchestrator
            await self.create_orchestrator()

            # Start orchestrator
            await self.start_orchestrator()

            # Demonstrate task management
            await self.demonstrate_task_management()

            # Demonstrate error handling
            await self.demonstrate_error_handling()

            # Monitor execution
            await self.monitor_execution(duration_minutes)

            # Demonstrate performance monitoring
            await self.demonstrate_performance_monitoring()

        except KeyboardInterrupt:
            self.logger.info("Demo interrupted by user")
        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
        finally:
            # Stop orchestrator
            await self.stop_orchestrator()

            # Print final statistics
            if self.orchestrator:
                final_status = self.orchestrator.get_system_status()
                print(f"\n{'=' * 60}")
                print("FINAL STATISTICS")
                print(f"{'=' * 60}")
                print(f"Total Executions: {len(self.orchestrator.task_executions)}")
                print(
                    f"Overall Health: {final_status['performance_metrics']['overall_health']:.2f}"
                )

                if not AGENTS_AVAILABLE:
                    print(f"\nMock Agent Execution Counts:")
                    for agent_name, agent in self.mock_agents.items():
                        print(f"  {agent_name}: {agent.execution_count} executions")


async def quick_start_example():
    """Quick start example using convenience functions"""
    print("Quick Start Example - Task Orchestrator")
    print("=" * 50)

    try:
        # Start orchestrator using convenience function
        orchestrator = await start_orchestrator()

        print("Orchestrator started successfully!")
        print("Press Ctrl+C to stop...")

        # Keep running
        await asyncio.sleep(60)  # Run for 1 minute

    except KeyboardInterrupt:
        print("\nStopping orchestrator...")
    finally:
        if "orchestrator" in locals():
            await orchestrator.stop()
        print("Orchestrator stopped.")


async def custom_config_example():
    """Example with custom configuration"""
    print("Custom Configuration Example")
    print("=" * 40)

    # Create custom configuration
    custom_config = {
        "orchestrator": {
            "enabled": True,
            "max_concurrent_tasks": 3,
            "default_timeout_minutes": 10,
            "health_check_interval_minutes": 2,
        },
        "tasks": {
            "custom_task": {
                "enabled": True,
                "interval_minutes": 2,
                "priority": "high",
                "dependencies": [],
                "conditions": {},
                "parameters": {"custom_param": "value"},
            }
        },
    }

    # Save custom config
    config_path = "config/custom_task_schedule.yaml"
    import yaml

    with open(config_path, "w") as f:
        yaml.dump(custom_config, f)

    # Create orchestrator with custom config
    orchestrator = create_task_orchestrator(config_path)

    # Add custom agent
    custom_agent = MockAgent("CustomAgent", 0.95)
    orchestrator.agents["custom_task"] = custom_agent
    orchestrator.agent_status["custom_task"] = orchestrator.AgentStatus(
        agent_name="custom_task"
    )

    # Start and run
    await orchestrator.start()
    print("Custom orchestrator started!")

    # Monitor for a short time
    await asyncio.sleep(30)

    # Stop
    await orchestrator.stop()
    print("Custom orchestrator stopped.")


def main():
    """Main function to run examples"""
    import argparse

    parser = argparse.ArgumentParser(description="Task Orchestrator Examples")
    parser.add_argument(
        "--example",
        choices=["comprehensive", "quick", "custom"],
        default="comprehensive",
        help="Example to run",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Duration in minutes for comprehensive demo",
    )

    args = parser.parse_args()

    if args.example == "comprehensive":
        # Run comprehensive demo
        example = TaskOrchestratorExample()
        asyncio.run(example.run_comprehensive_demo(args.duration))

    elif args.example == "quick":
        # Run quick start example
        asyncio.run(quick_start_example())

    elif args.example == "custom":
        # Run custom configuration example
        asyncio.run(custom_config_example())


if __name__ == "__main__":
    main()
