"""
Main Entry Point for Evolve Trading System

This module provides the main entry point for the Evolve trading system,
including command-line interface, system health checks, and various
launch options for different interfaces.
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Any, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for the Evolve trading system.

    Provides a command-line interface for launching different components
    and checking system health.
    """
    print("🚀 Evolve Trading System")
    print("=" * 50)

    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].lower()

    try:
        if command == "health":
            result = check_system_health()
            print_health_result(result)
        elif command == "orchestrator":
            config_path = (
                sys.argv[2] if len(sys.argv) > 2 else "config/orchestrator_config.yaml"
            )
            monitor = "--monitor" in sys.argv
            result = launch_task_orchestrator(config_path, monitor)
            print_result(result)
        elif command == "streamlit":
            result = launch_streamlit_interface()
            print_result(result)
        elif command == "terminal":
            result = launch_terminal_interface()
            print_result(result)
        elif command == "api":
            result = launch_api_interface()
            print_result(result)
        elif command == "unified":
            result = launch_unified_interface()
            print_result(result)
        elif command == "execute":
            if len(sys.argv) < 3:
                print("❌ Error: Command required for execute mode")
                print_usage()
                return
            cmd = sys.argv[2]
            result = execute_command(cmd)
            print_result(result)
        else:
            print(f"❌ Unknown command: {command}")
            print_usage()

    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")


def print_usage():
    """Print usage information."""
    print("\nUsage:")
    print("  python main.py <command> [options]")
    print("\nCommands:")
    print("  health                    - Check system health")
    print("  orchestrator [config]     - Launch task orchestrator")
    print("  streamlit                 - Launch Streamlit interface")
    print("  terminal                  - Launch terminal interface")
    print("  api                       - Launch API interface")
    print("  unified                   - Launch unified interface")
    print("  execute <command>         - Execute a specific command")
    print("\nOptions:")
    print("  --monitor                 - Enable monitoring (for orchestrator)")
    print("\nExamples:")
    print("  python main.py health")
    print("  python main.py orchestrator config/my_config.yaml --monitor")
    print("  python main.py streamlit")
    print("  python main.py execute 'run_strategy RSI_Strategy'")


def launch_task_orchestrator(config_path: str, monitor: bool = False) -> Dict[str, Any]:
    """
    Launch the Task Orchestrator.

    Args:
        config_path: Path to orchestrator configuration file
        monitor: Enable real-time monitoring

    Returns:
        Dict[str, Any]: Orchestrator status
    """
    try:
        logger.info("Launching Task Orchestrator...")

        # Import orchestrator
        try:
            from core.task_orchestrator import start_orchestrator
        except ImportError as e:
            logger.error(f"Failed to import TaskOrchestrator: {e}")
            return {
                "status": "error",
                "error": f"TaskOrchestrator not available: {e}",
                "timestamp": datetime.now().isoformat(),
            }

        async def run_orchestrator():
            """Run the orchestrator with monitoring if enabled"""
            try:
                # Start orchestrator
                orchestrator = await start_orchestrator(config_path)
                logger.info("Task Orchestrator started successfully")

                if monitor:
                    logger.info("Starting real-time monitoring...")
                    # Run monitoring for a period
                    await asyncio.sleep(300)  # Monitor for 5 minutes
                else:
                    # Keep running indefinitely
                    while True:
                        await asyncio.sleep(60)  # Check every minute

            except KeyboardInterrupt:
                logger.info("Stopping Task Orchestrator...")
                await orchestrator.stop()
                logger.info("Task Orchestrator stopped")
            except Exception as e:
                logger.error(f"Orchestrator error: {e}")
                if "orchestrator" in locals():
                    await orchestrator.stop()

        # Run the orchestrator
        asyncio.run(run_orchestrator())

        return {
            "status": "success",
            "message": "Task Orchestrator completed",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error launching Task Orchestrator: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def check_system_health() -> Dict[str, Any]:
    """
    Check system health and component status.

    Returns:
        Dict[str, Any]: System health information
    """
    try:
        logger.info("Checking system health")

        # Check Task Orchestrator availability
        orchestrator_health = {
            "status": "unknown",
            "message": "Task Orchestrator not checked",
        }

        try:
            from core.task_orchestrator import TaskOrchestrator

            orchestrator = TaskOrchestrator()
            status = orchestrator.get_system_status()
            orchestrator_health = {
                "status": "healthy" if status["total_tasks"] > 0 else "warning",
                "message": f"Found {status['total_tasks']} configured tasks",
                "overall_health": status["performance_metrics"]["overall_health"],
            }
        except Exception as e:
            orchestrator_health = {
                "status": "error",
                "message": f"Task Orchestrator error: {e}",
            }

        from interface.unified_interface import UnifiedInterface

        interface = UnifiedInterface()
        health = interface.system_health

        # Display health information
        print("\n" + "=" * 60)
        print("🏥 SYSTEM HEALTH CHECK")
        print("=" * 60)

        print(f"📊 Overall Health: {health['overall_health']}")
        print(f"⏰ Timestamp: {health['timestamp']}")
        print(f"🔄 System Status: {health['status']}")

        print("\n📋 Component Status:")
        for component, status in health["components"].items():
            status_icon = "✅" if status["healthy"] else "❌"
            print(f"  {status_icon} {component}: {status['status']}")

        print(f"\n🎯 Task Orchestrator: {orchestrator_health['status']}")
        print(f"   {orchestrator_health['message']}")

        if "overall_health" in orchestrator_health:
            print(f"   Health Score: {orchestrator_health['overall_health']}")

        return {
            "status": "success",
            "health": health,
            "orchestrator_health": orchestrator_health,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error checking system health: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def execute_command(command: str) -> Dict[str, Any]:
    """
    Execute a specific command through the unified interface.

    Args:
        command: Command to execute

    Returns:
        Dict[str, Any]: Command execution result
    """
    try:
        logger.info(f"Executing command: {command}")

        from interface.unified_interface import UnifiedInterface

        interface = UnifiedInterface()
        result = interface.execute_command(command)

        return {
            "status": "success",
            "command": command,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error executing command: {e}")
        return {
            "status": "error",
            "command": command,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def launch_streamlit_interface() -> Dict[str, Any]:
    """
    Launch the Streamlit web interface.

    Returns:
        Dict[str, Any]: Launch status
    """
    try:
        logger.info("Launching Streamlit interface...")

        import os
        import subprocess

        # Check if streamlit is available
        try:
            pass
        except ImportError:
            return {
                "status": "error",
                "error": "Streamlit not installed. Run: pip install streamlit",
                "timestamp": datetime.now().isoformat(),
            }

        # Launch streamlit
        app_path = os.path.join(os.path.dirname(__file__), "app.py")
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", app_path])

        return {
            "status": "success",
            "message": "Streamlit interface launched",
            "url": "http://localhost:8501",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error launching Streamlit interface: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def launch_terminal_interface() -> Dict[str, Any]:
    """
    Launch the terminal-based interface.

    Returns:
        Dict[str, Any]: Launch status
    """
    try:
        logger.info("Launching terminal interface...")

        from interface.unified_interface import UnifiedInterface

        interface = UnifiedInterface()
        interface.start_terminal_interface()

        return {
            "status": "success",
            "message": "Terminal interface launched",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error launching terminal interface: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def launch_api_interface() -> Dict[str, Any]:
    """
    Launch the FastAPI interface.

    Returns:
        Dict[str, Any]: Launch status
    """
    try:
        logger.info("Launching API interface...")

        import uvicorn
        from fastapi import FastAPI

        app = FastAPI(title="Evolve Trading API", version="1.0.0")

        @app.get("/")
        async def root():
            return {"message": "Evolve Trading API", "status": "running"}

        @app.get("/health")
        async def health_check():
            return check_system_health()

        @app.post("/command")
        async def execute_command(command: str):
            return execute_command(command)

        # Run the API server
        uvicorn.run(app, host="0.0.0.0", port=8000)

        return {
            "status": "success",
            "message": "API interface launched",
            "url": "http://localhost:8000",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error launching API interface: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def launch_unified_interface() -> Dict[str, Any]:
    """
    Launch the unified interface with all options.

    Returns:
        Dict[str, Any]: Launch status
    """
    try:
        logger.info("Launching unified interface...")

        from interface.unified_interface import UnifiedInterface

        interface = UnifiedInterface()
        interface.start()

        return {
            "status": "success",
            "message": "Unified interface launched",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error launching unified interface: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def print_health_result(result: Dict[str, Any]):
    """Print health check result."""
    if result["status"] == "success":
        print("✅ Health check completed successfully")
    else:
        print(f"❌ Health check failed: {result.get('error', 'Unknown error')}")


def print_result(result: Dict[str, Any]):
    """Print command result."""
    if result["status"] == "success":
        print(f"✅ {result.get('message', 'Operation completed successfully')}")
        if "url" in result:
            print(f"🌐 Access at: {result['url']}")
    else:
        print(f"❌ Operation failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
