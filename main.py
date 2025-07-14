#!/usr/bin/env python3
"""
Main Entry Point for Evolve Trading Platform

This is the production-ready main entry point that provides multiple
access methods to the trading system with proper error handling and
fallback mechanisms.
"""

import argparse
import logging
import os
import sys
import asyncio
from datetime import datetime
from typing import Any, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/main.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    """
    Main entry point with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Evolve Trading Platform - Production-Ready AI Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --streamlit                    # Launch Streamlit interface
  python main.py --terminal                     # Launch terminal interface
  python main.py --command "forecast AAPL 7d"   # Execute specific command
  python main.py --api                          # Launch API server
  python main.py --health                       # Check system health
  python main.py --orchestrator                 # Launch Task Orchestrator
  python main.py --orchestrator --monitor       # Launch with monitoring
        """,
    )

    # Interface options
    parser.add_argument(
        "--streamlit", action="store_true", help="Launch Streamlit web interface"
    )
    parser.add_argument(
        "--terminal", action="store_true", help="Launch terminal interface"
    )
    parser.add_argument("--api", action="store_true", help="Launch REST API server")
    parser.add_argument(
        "--unified",
        action="store_true",
        help="Launch unified interface with multiple access methods",
    )

    # Task Orchestrator options
    parser.add_argument(
        "--orchestrator", action="store_true", help="Launch Task Orchestrator"
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Enable real-time monitoring"
    )
    parser.add_argument(
        "--orchestrator-config",
        type=str,
        default="config/task_schedule.yaml",
        help="Path to orchestrator configuration file",
    )

    # Command execution
    parser.add_argument(
        "--command",
        type=str,
        help='Execute a specific command (e.g., "forecast AAPL 7d")',
    )

    # System options
    parser.add_argument(
        "--health", action="store_true", help="Check system health and component status"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/system_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        logger.info("Starting Evolve Trading Platform")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")

        # Check system health
        if args.health:
            return check_system_health()

        # Execute command
        if args.command:
            return execute_command(args.command)

        # Launch Task Orchestrator
        if args.orchestrator:
            return launch_task_orchestrator(args.orchestrator_config, args.monitor)

        # Launch interface
        if args.streamlit:
            return launch_streamlit_interface()
        elif args.terminal:
            return launch_terminal_interface()
        elif args.api:
            return launch_api_interface()
        elif args.unified:
            return launch_unified_interface()
        else:
            # Default to unified interface
            return launch_unified_interface()

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down gracefully")
        return {"status": "interrupted", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


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
                if 'orchestrator' in locals():
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
            "message": "Task Orchestrator not checked"
        }
        
        try:
            from core.task_orchestrator import TaskOrchestrator
            orchestrator = TaskOrchestrator()
            status = orchestrator.get_system_status()
            orchestrator_health = {
                "status": "healthy" if status['total_tasks'] > 0 else "warning",
                "message": f"Found {status['total_tasks']} configured tasks",
                "overall_health": status['performance_metrics']['overall_health']
            }
        except Exception as e:
            orchestrator_health = {
                "status": "error",
                "message": f"Task Orchestrator error: {e}"
            }

        from interface.unified_interface import UnifiedInterface

        interface = UnifiedInterface()
        health = interface.system_health

        # Display health information
        print("\n" + "=" * 60)
        print("EVOLVE TRADING PLATFORM - SYSTEM HEALTH")
        print("=" * 60)

        print(f"Overall Status: {health['overall_status'].upper()}")
        print(
            f"Healthy Components: {health['healthy_components']}/{health['total_components']}"
        )
        print(f"Timestamp: {health['timestamp']}")

        print("\nComponent Status:")
        print("-" * 40)
        
        # Add Task Orchestrator to component status
        components = health["components"].copy()
        components["Task Orchestrator"] = orchestrator_health
        
        for name, component_health in components.items():
            status = component_health.get("status", "unknown")
            status_icon = {
                "healthy": "✅",
                "fallback": "⚠️",
                "error": "❌",
                "unknown": "❓",
            }.get(status, "❓")

            print(f"{status_icon} {name}: {status}")

            if "message" in component_health:
                print(f"    └─ {component_health['message']}")

        print("\n" + "=" * 60)

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
    Execute a specific command.

    Args:
        command: Command to execute

    Returns:
        Dict[str, Any]: Command execution results
    """
    try:
        logger.info(f"Executing command: {command}")

        from interface.unified_interface import UnifiedInterface

        interface = UnifiedInterface()
        result = interface.process_command(command)

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
    Launch Streamlit web interface.

    Returns:
        Dict[str, Any]: Interface status
    """
    try:
        logger.info("Launching Streamlit interface...")

        import subprocess
        import sys

        # Launch Streamlit app
        result = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app.py"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Streamlit interface launched successfully",
                "timestamp": datetime.now().isoformat(),
            }
        else:
            return {
                "status": "error",
                "error": f"Streamlit failed: {result.stderr}",
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
    Launch terminal interface.

    Returns:
        Dict[str, Any]: Interface status
    """
    try:
        logger.info("Launching terminal interface...")

        from interface.unified_interface import UnifiedInterface

        interface = UnifiedInterface()
        interface.run_terminal()

        return {
            "status": "success",
            "message": "Terminal interface completed",
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
    Launch REST API server.

    Returns:
        Dict[str, Any]: Interface status
    """
    try:
        logger.info("Launching API server...")

        from fastapi import FastAPI
        import uvicorn
        from interface.unified_interface import UnifiedInterface

        app = FastAPI(title="Evolve Trading Platform API")
        interface = UnifiedInterface()

        @app.get("/")
        async def root():
            return {"message": "Evolve Trading Platform API", "status": "running"}

        @app.get("/health")
        async def health_check():
            return interface.system_health

        @app.post("/command")
        async def execute_command(command: str):
            return interface.process_command(command)

        # Run the API server
        uvicorn.run(app, host="0.0.0.0", port=8000)

        return {
            "status": "success",
            "message": "API server launched successfully",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error launching API server: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def launch_unified_interface() -> Dict[str, Any]:
    """
    Launch unified interface with multiple access methods.

    Returns:
        Dict[str, Any]: Interface status
    """
    try:
        logger.info("Launching unified interface...")

        from interface.unified_interface import UnifiedInterface

        interface = UnifiedInterface()
        interface.run()

        return {
            "status": "success",
            "message": "Unified interface completed",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error launching unified interface: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


if __name__ == "__main__":
    result = main()
    if isinstance(result, dict) and result.get("status") == "error":
        sys.exit(1)
