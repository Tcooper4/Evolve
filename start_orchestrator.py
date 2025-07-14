#!/usr/bin/env python3
"""
Evolve Task Orchestrator Startup Script

This script provides easy startup and integration of the Task Orchestrator
with the existing Evolve trading platform.
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/orchestrator_startup.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


async def start_orchestrator_standalone(config_path: str = "config/task_schedule.yaml"):
    """Start Task Orchestrator in standalone mode"""
    try:
        logger.info("Starting Task Orchestrator in standalone mode...")
        
        from core.task_orchestrator import start_orchestrator
        
        orchestrator = await start_orchestrator(config_path)
        
        logger.info("Task Orchestrator started successfully")
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


async def start_with_monitoring(config_path: str = "config/task_schedule.yaml", duration_minutes: int = 60):
    """Start orchestrator with monitoring for a specified duration"""
    try:
        logger.info(f"Starting orchestrator with monitoring for {duration_minutes} minutes...")
        
        from core.task_orchestrator import start_orchestrator
        
        orchestrator = await start_orchestrator(config_path)
        
        logger.info("Task Orchestrator started with monitoring")
        
        # Monitor for specified duration
        start_time = datetime.now()
        end_time = start_time.replace(minute=start_time.minute + duration_minutes)
        
        while datetime.now() < end_time:
            # Get status every 30 seconds
            status = orchestrator.get_system_status()
            
            print(f"\n{'='*60}")
            print(f"Orchestrator Status - {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*60}")
            print(f"Running: {status['orchestrator_running']}")
            print(f"Total Tasks: {status['total_tasks']}")
            print(f"Enabled Tasks: {status['enabled_tasks']}")
            print(f"Running Tasks: {status['running_tasks']}")
            print(f"Overall Health: {status['performance_metrics']['overall_health']:.2f}")
            
            # Show agent status
            print(f"\nAgent Status:")
            for agent_name, agent_status in status['agent_status'].items():
                health_icon = "🟢" if agent_status['health_score'] > 0.8 else "🟡" if agent_status['health_score'] > 0.6 else "🔴"
                print(f"  {health_icon} {agent_name}: {agent_status['health_score']:.2f}")
            
            await asyncio.sleep(30)
        
        logger.info("Monitoring period completed, shutting down...")
        await orchestrator.stop()
        
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted, shutting down...")
        if 'orchestrator' in locals():
            await orchestrator.stop()
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        return False
    
    return True


def check_system_requirements():
    """Check if system requirements are met"""
    logger.info("Checking system requirements...")
    
    requirements_met = True
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ required")
        requirements_met = False
    else:
        logger.info(f"✅ Python {sys.version.split()[0]} - OK")
    
    # Check required directories
    required_dirs = ["logs", "config", "core", "agents"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            logger.error(f"Required directory '{dir_name}' not found")
            requirements_met = False
        else:
            logger.info(f"✅ Directory '{dir_name}' - OK")
    
    # Check configuration file
    config_path = "config/task_schedule.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Configuration file '{config_path}' not found")
        requirements_met = False
    else:
        logger.info(f"✅ Configuration file '{config_path}' - OK")
    
    # Check core modules
    try:
        from core.task_orchestrator import TaskOrchestrator
        logger.info("✅ TaskOrchestrator module - OK")
    except ImportError as e:
        logger.error(f"TaskOrchestrator module not available: {e}")
        requirements_met = False
    
    return requirements_met


def show_system_status():
    """Show current system status"""
    logger.info("Checking system status...")
    
    try:
        from system.orchestrator_integration import get_system_integration_status
        
        status = get_system_integration_status()
        
        print(f"\n{'='*60}")
        print("EVOLVE TASK ORCHESTRATOR - SYSTEM STATUS")
        print(f"{'='*60}")
        
        status_icon = {
            "available": "🟢",
            "not_available": "🔴",
            "not_configured": "🟡",
            "error": "🔴"
        }.get(status.get("status", "unknown"), "❓")
        
        print(f"Status: {status_icon} {status.get('status', 'unknown').title()}")
        
        if status.get("status") == "available":
            print(f"Total Tasks: {status.get('total_tasks', 0)}")
            print(f"Enabled Tasks: {status.get('enabled_tasks', 0)}")
            print(f"Overall Health: {status.get('overall_health', 0):.1%}")
        else:
            print(f"Message: {status.get('message', 'No message available')}")
        
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"{'='*60}")
        
        return status.get("status") == "available"
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        return False


def main():
    """Main function with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Evolve Task Orchestrator Startup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_orchestrator.py --standalone              # Start standalone orchestrator
  python start_orchestrator.py --integrated              # Start integrated system
  python start_orchestrator.py --monitor --duration 30   # Monitor for 30 minutes
  python start_orchestrator.py --check                   # Check system requirements
  python start_orchestrator.py --status                  # Show system status
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--standalone", action="store_true", 
                           help="Start Task Orchestrator in standalone mode")
    mode_group.add_argument("--integrated", action="store_true", 
                           help="Start integrated system with existing components")
    mode_group.add_argument("--monitor", action="store_true", 
                           help="Start with monitoring mode")
    mode_group.add_argument("--check", action="store_true", 
                           help="Check system requirements")
    mode_group.add_argument("--status", action="store_true", 
                           help="Show current system status")
    
    # Optional arguments
    parser.add_argument("--config", type=str, default="config/task_schedule.yaml",
                       help="Path to orchestrator configuration file")
    parser.add_argument("--duration", type=int, default=60,
                       help="Duration in minutes for monitoring mode")
    parser.add_argument("--log-level", type=str, 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        logger.info("Evolve Task Orchestrator Startup Script")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Check system requirements
        if args.check:
            if check_system_requirements():
                logger.info("✅ All system requirements met")
                return 0
            else:
                logger.error("❌ System requirements not met")
                return 1
        
        # Show system status
        if args.status:
            if show_system_status():
                return 0
            else:
                return 1
        
        # Check requirements before starting
        if not check_system_requirements():
            logger.error("❌ System requirements not met. Cannot start orchestrator.")
            return 1
        
        # Start appropriate mode
        if args.standalone:
            success = asyncio.run(start_orchestrator_standalone(args.config))
        elif args.integrated:
            success = asyncio.run(start_integrated_system(args.config))
        elif args.monitor:
            success = asyncio.run(start_with_monitoring(args.config, args.duration))
        else:
            logger.error("No valid mode specified")
            return 1
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Startup script interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Startup script failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 