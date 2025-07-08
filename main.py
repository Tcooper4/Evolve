#!/usr/bin/env python3
"""
Main Entry Point for Evolve Trading Platform

This is the production-ready main entry point that provides multiple
access methods to the trading system with proper error handling and
fallback mechanisms.
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log'),
        logging.StreamHandler(sys.stdout)
    ]
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
        """
    )
    
    # Interface options
    parser.add_argument(
        '--streamlit', 
        action='store_true',
        help='Launch Streamlit web interface'
    )
    parser.add_argument(
        '--terminal', 
        action='store_true',
        help='Launch terminal interface'
    )
    parser.add_argument(
        '--api', 
        action='store_true',
        help='Launch REST API server'
    )
    parser.add_argument(
        '--unified', 
        action='store_true',
        help='Launch unified interface with multiple access methods'
    )
    
    # Command execution
    parser.add_argument(
        '--command', 
        type=str,
        help='Execute a specific command (e.g., "forecast AAPL 7d")'
    )
    
    # System options
    parser.add_argument(
        '--health', 
        action='store_true',
        help='Check system health and component status'
    )
    parser.add_argument(
        '--config', 
        type=str,
        default='config/system_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--log-level', 
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
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
        return {'status': 'interrupted', 'timestamp': datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}

def check_system_health() -> Dict[str, Any]:
    """
    Check system health and component status.
    
    Returns:
        Dict[str, Any]: System health information
    """
    try:
        logger.info("Checking system health")
        
        from interface.unified_interface import UnifiedInterface
        
        interface = UnifiedInterface()
        health = interface.system_health
        
        # Display health information
        print("\n" + "="*60)
        print("EVOLVE TRADING PLATFORM - SYSTEM HEALTH")
        print("="*60)
        
        print(f"Overall Status: {health['overall_status'].upper()}")
        print(f"Healthy Components: {health['healthy_components']}/{health['total_components']}")
        print(f"Timestamp: {health['timestamp']}")
        
        print("\nComponent Status:")
        print("-" * 40)
        for name, component_health in health['components'].items():
            status = component_health.get('status', 'unknown')
            status_icon = {
                'healthy': '✅',
                'fallback': '⚠️',
                'error': '❌',
                'unknown': '❓'
            }.get(status, '❓')
            
            print(f"{status_icon} {name}: {status}")
            
            if 'message' in component_health:
                print(f"    └─ {component_health['message']}")
        
        print("\n" + "="*60)
        
        return {
            'status': 'success',
            'health': health,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking system health: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
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
        
        # Parse command
        parts = command.lower().split()
        
        if len(parts) < 2:
            return {
                'status': 'error',
                'error': 'Invalid command format. Use: <action> <symbol> [parameters]',
                'timestamp': datetime.now().isoformat()
            }
        
        action = parts[0]
        symbol = parts[1].upper()
        
        if action == 'forecast':
            days = int(parts[2]) if len(parts) > 2 else 7
            model = parts[3] if len(parts) > 3 else 'ensemble'
            
            result = interface._generate_forecast(symbol, '1d', model, days)
            
            print(f"\nForecast for {symbol}:")
            print(f"Model: {result.get('model', 'Unknown')}")
            print(f"Confidence: {result.get('confidence', 0):.1%}")
            print(f"Status: {result.get('status', 'Unknown')}")
            
            if 'forecast_values' in result:
                print(f"Forecast values: {result['forecast_values'][:5]}...")
            
            return {
                'status': 'success',
                'command': command,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
        
        elif action == 'health':
            return check_system_health()
        
        else:
            return {
                'status': 'error',
                'error': f'Unknown command: {action}',
                'timestamp': datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def launch_streamlit_interface() -> Dict[str, Any]:
    """
    Launch the Streamlit web interface.
    
    Returns:
        Dict[str, Any]: Interface launch results
    """
    try:
        logger.info("Launching Streamlit interface")
        
        import subprocess
        import sys
        
        # Launch Streamlit
        cmd = [sys.executable, '-m', 'streamlit', 'run', 'interface/unified_interface.py']
        
        print("🚀 Launching Evolve Trading Platform...")
        print("📊 Streamlit interface will open in your browser")
        print("🔗 URL: http://localhost:8501")
        print("\nPress Ctrl+C to stop the server")
        
        subprocess.run(cmd)
        
        return {
            'status': 'success',
            'interface': 'streamlit',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error launching Streamlit interface: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def launch_terminal_interface() -> Dict[str, Any]:
    """
    Launch the terminal interface.
    
    Returns:
        Dict[str, Any]: Interface launch results
    """
    try:
        logger.info("Launching terminal interface")
        
        print("🚀 Evolve Trading Platform - Terminal Interface")
        print("=" * 50)
        
        # Simple terminal interface
        while True:
            print("\nAvailable commands:")
            print("1. forecast <symbol> [days] [model] - Generate forecast")
            print("2. health - Check system health")
            print("3. quit - Exit")
            
            try:
                command = input("\nEnter command: ").strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                elif command.lower() == 'health':
                    check_system_health()
                elif command.lower().startswith('forecast'):
                    execute_command(command)
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        return {
            'status': 'success',
            'interface': 'terminal',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error launching terminal interface: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def launch_api_interface() -> Dict[str, Any]:
    """
    Launch the REST API interface.
    
    Returns:
        Dict[str, Any]: Interface launch results
    """
    try:
        logger.info("Launching API interface")
        
        print("🚀 Evolve Trading Platform - API Interface")
        print("📡 REST API server starting...")
        print("🔗 API will be available at: http://localhost:8000")
        print("\nPress Ctrl+C to stop the server")
        
        # TODO: Implement API server
        print("API interface not yet implemented")
        
        return {
            'status': 'not_implemented',
            'interface': 'api',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error launching API interface: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def launch_unified_interface() -> Dict[str, Any]:
    """
    Launch the unified interface.
    
    Returns:
        Dict[str, Any]: Interface launch results
    """
    try:
        logger.info("Launching unified interface")
        
        from interface.unified_interface import UnifiedInterface
        
        interface = UnifiedInterface()
        result = interface.run()
        
        return {
            'status': 'success',
            'interface': 'unified',
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error launching unified interface: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    result = main()
    
    if result.get('status') == 'error':
        sys.exit(1)
    elif result.get('status') == 'interrupted':
        sys.exit(0)
    else:
        sys.exit(0) 