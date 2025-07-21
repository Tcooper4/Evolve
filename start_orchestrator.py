"""
Task Orchestrator Startup Script

This script provides various ways to start the Task Orchestrator,
including standalone mode, integrated mode, and monitoring mode.
"""

import asyncio
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def start_orchestrator_standalone(config_path: str = "config/task_schedule.yaml"):
    """Start Task Orchestrator in standalone mode"""
    try:
        logger.info("Starting Task Orchestrator in standalone mode...")

        from core.task_orchestrator import start_orchestrator

        orchestrator = await start_orchestrator(config_path)

        # Check agent registration status
        await _check_agent_registration()

        logger.info("Task Orchestrator started successfully")
        print(
            "Orchestrator started successfully. All services are now "
            "running and coordinated."
        )
        logger.info("Press Ctrl+C to stop...")

        # Keep running
        try:
            while True:
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        finally:
            await orchestrator.stop()
            logger.info("Task Orchestrator stopped")

    except Exception as e:
        logger.error(f"Failed to start orchestrator: {e}")
        return False

    return True


async def start_integrated_system(config_path: str = "config/task_schedule.yaml"):
    """Start Task Orchestrator integrated with existing system"""
    try:
        logger.info("Starting integrated system...")

        from system.orchestrator_integration import start_integrated_system

        integration = await start_integrated_system(config_path)

        # Check agent registration status
        await _check_agent_registration()

        logger.info("Integrated system started successfully")
        logger.info("Press Ctrl+C to stop...")

        # Keep running
        try:
            while True:
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        finally:
            await integration.stop_integrated_system()
            logger.info("Integrated system stopped")

    except Exception as e:
        logger.error(f"Failed to start integrated system: {e}")
        return False

    return True


async def start_with_monitoring(
    config_path: str = "config/task_schedule.yaml", duration_minutes: int = 60
):
    """Start orchestrator with monitoring for a specified duration"""
    try:
        logger.info(
            f"Starting orchestrator with monitoring for {duration_minutes} minutes..."
        )

        from core.task_orchestrator import start_orchestrator

        orchestrator = await start_orchestrator(config_path)

        # Check agent registration status
        await _check_agent_registration()

        logger.info("Task Orchestrator started with monitoring")

        # Monitor for specified duration
        start_time = datetime.now()
        end_time = start_time.replace(minute=start_time.minute + duration_minutes)

        while datetime.now() < end_time:
            # Get status every 30 seconds
            status = orchestrator.get_system_status()

            print(f"\n{'=' * 60}")
            print(f"Orchestrator Status - {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'=' * 60}")
            print(f"Running: {status['orchestrator_running']}")
            print(f"Total Tasks: {status['total_tasks']}")
            print(f"Enabled Tasks: {status['enabled_tasks']}")
            print(f"Running Tasks: {status['running_tasks']}")
            print(
                f"Overall Health: {status['performance_metrics']['overall_health']:.2f}"
            )

            # Show agent status
            print(f"\nAgent Status:")
            for agent_name, agent_status in status["agent_status"].items():
                status_icon = "✅" if agent_status["healthy"] else "❌"
                print(f"  {status_icon} {agent_name}: {agent_status['status']}")

            # Show task status
            print(f"\nTask Status:")
            for task_name, task_status in status["task_status"].items():
                status_icon = "🟢" if task_status["running"] else "🔴"
                print(f"  {status_icon} {task_name}: {task_status['status']}")

            await asyncio.sleep(30)

        logger.info("Monitoring period completed")

    except Exception as e:
        logger.error(f"Failed to start orchestrator with monitoring: {e}")
        return False

    return True


async def _check_agent_registration():
    """Check if agents are properly registered"""
    try:
        logger.info("Checking agent registration...")

        # Import agent registry
        try:
            from agents.agent_registry import AgentRegistry

            registry = AgentRegistry()

            # Check registered agents
            registered_agents = registry.get_registered_agents()

            if registered_agents:
                logger.info(f"Found {len(registered_agents)} registered agents:")
                for agent_name, agent_info in registered_agents.items():
                    logger.info(
                        f"  - {agent_name}: {agent_info.get('status', 'unknown')}"
                    )
            else:
                logger.warning("No agents found in registry")

        except ImportError:
            logger.warning("Agent registry not available")
        except Exception as e:
            logger.error(f"Error checking agent registry: {e}")

        # Check agent availability
        try:
            from agents.agent_controller import AgentController

            controller = AgentController()

            available_agents = controller.get_available_agents()

            if available_agents:
                logger.info(f"Found {len(available_agents)} available agents")
            else:
                logger.warning("No agents available")

        except ImportError:
            logger.warning("Agent controller not available")
        except Exception as e:
            logger.error(f"Error checking agent availability: {e}")

    except Exception as e:
        logger.error(f"Error in agent registration check: {e}")


def check_system_requirements():
    """Check if system meets requirements"""
    logger.info("Checking system requirements...")

    requirements_met = True

    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher required")
        requirements_met = False
    else:
        logger.info(f"Python version: {sys.version}")

    # Check required packages
    required_packages = [
        "asyncio",
        "logging",
        "datetime",
        "typing",
        "pandas",
        "numpy",
        "aiohttp",
        "fastapi",
    ]

    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} available")
        except ImportError:
            logger.error(f"✗ {package} not available")
            requirements_met = False

    # Check configuration files
    config_files = ["config/task_schedule.yaml", "config/agent_config.json"]

    for config_file in config_files:
        try:
            with open(config_file, "r") as f:
                f.read()
            logger.info(f"✓ {config_file} found")
        except FileNotFoundError:
            logger.warning(f"⚠ {config_file} not found")
        except Exception as e:
            logger.error(f"✗ Error reading {config_file}: {e}")
            requirements_met = False

    return requirements_met


def show_system_status():
    """Show current system status"""
    logger.info("Checking system status...")

    try:
        from core.task_orchestrator import TaskOrchestrator

        orchestrator = TaskOrchestrator()
        status = orchestrator.get_system_status()

        print(f"\n{'=' * 60}")
        print("SYSTEM STATUS")
        print(f"{'=' * 60}")
        print(f"Orchestrator Running: {status['orchestrator_running']}")
        print(f"Total Tasks: {status['total_tasks']}")
        print(f"Enabled Tasks: {status['enabled_tasks']}")
        print(f"Running Tasks: {status['running_tasks']}")
        print(f"Overall Health: {status['performance_metrics']['overall_health']:.2f}")

        print(f"\nAgent Status:")
        for agent_name, agent_status in status["agent_status"].items():
            status_icon = "✅" if agent_status["healthy"] else "❌"
            print(f"  {status_icon} {agent_name}: {agent_status['status']}")

        print(f"\nTask Status:")
        for task_name, task_status in status["task_status"].items():
            status_icon = "🟢" if task_status["running"] else "🔴"
            print(f"  {status_icon} {task_name}: {task_status['status']}")

    except Exception as e:
        logger.error(f"Error getting system status: {e}")


def main():
    """Main entry point"""
    print("🚀 Task Orchestrator Startup")
    print("=" * 50)

    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].lower()

    try:
        if command == "standalone":
            config_path = (
                sys.argv[2] if len(sys.argv) > 2 else "config/task_schedule.yaml"
            )
            asyncio.run(start_orchestrator_standalone(config_path))
        elif command == "integrated":
            config_path = (
                sys.argv[2] if len(sys.argv) > 2 else "config/task_schedule.yaml"
            )
            asyncio.run(start_integrated_system(config_path))
        elif command == "monitor":
            config_path = (
                sys.argv[2] if len(sys.argv) > 2 else "config/task_schedule.yaml"
            )
            duration = int(sys.argv[3]) if len(sys.argv) > 3 else 60
            asyncio.run(start_with_monitoring(config_path, duration))
        elif command == "check":
            if check_system_requirements():
                print("✅ System requirements met")
            else:
                print("❌ System requirements not met")
        elif command == "status":
            show_system_status()
        else:
            print(f"❌ Unknown command: {command}")
            print_usage()

    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")


def print_usage():
    """Print usage information"""
    print("\nUsage:")
    print("  python start_orchestrator.py <command> [options]")
    print("\nCommands:")
    print("  standalone [config]     - Start in standalone mode")
    print("  integrated [config]     - Start in integrated mode")
    print("  monitor [config] [min]  - Start with monitoring")
    print("  check                   - Check system requirements")
    print("  status                  - Show system status")
    print("\nExamples:")
    print("  python start_orchestrator.py standalone")
    print("  python start_orchestrator.py monitor config/my_config.yaml 30")
    print("  python start_orchestrator.py check")


if __name__ == "__main__":
    main()
