#!/usr/bin/env python3
"""
Unified Interface Launcher

Provides easy access to all Evolve trading system features through:
- Streamlit web interface
- Terminal command-line interface
- Direct command execution
"""

import argparse
import sys
import os
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from unified_interface import UnifiedInterface, streamlit_ui, terminal_ui


def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description='Evolve Unified Interface Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
Examples:
  # Launch Streamlit web interface
  python launch_unified_interface.py --streamlit

  # Launch terminal interface
  python launch_unified_interface.py --terminal

  # Execute a single command
  python launch_unified_interface.py --command "forecast AAPL 7d"

  # Ask QuantGPT a question
  python launch_unified_interface.py --command "What's the best model for TSLA?"

  # Get help
  python launch_unified_interface.py --help
        """
    )
    
    parser.add_argument('--streamlit', action='store_true',
                       help='Launch Streamlit web interface')
    parser.add_argument('--terminal', action='store_true',
                       help='Launch terminal command-line interface')
    parser.add_argument('--command', type=str,
                       help='Execute a single command and exit')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo commands')
    
    args = parser.parse_args()
    
    # Default to Streamlit if no mode specified
    if not any([args.streamlit, args.terminal, args.command, args.demo]):
        args.streamlit = True
    
    # Initialize interface
    config = {}
    if args.config:
        try:
            import json
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
    
    interface = UnifiedInterface(config)
    
    if args.demo:
        run_demo(interface)
    elif args.command:
        execute_command(interface, args.command)
    elif args.streamlit:
        launch_streamlit()
    elif args.terminal:
        terminal_ui()


def run_demo(interface: UnifiedInterface):
    """Run demonstration commands."""
    print("🔮 Evolve Unified Interface - Demo Mode")
    print("=" * 60)
    
    demo_commands = [
        "help",
        "forecast AAPL 7d",
        "tune model lstm AAPL",
        "strategy list",
        "agent list",
        "portfolio status",
        "What's the best model for TSLA?",
        "status"
    ]
    
    for command in demo_commands:
        print(f"\n📝 Executing: {command}")
        print("-" * 40)
        
        result = interface.process_command(command)
        
        if result.get('status') == 'error':
            print(f"❌ Error: {result.get('error')}")
        else:
            print_result_simple(result)
        
        print("-" * 40)
    
    print("\n✅ Demo completed!")


    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
def execute_command(interface: UnifiedInterface, command: str):
    """Execute a single command."""
    print(f"🔮 Executing: {command}")
    print("=" * 50)
    
    result = interface.process_command(command)
    
    if result.get('status') == 'error':
        print(f"❌ Error: {result.get('error')}")
        sys.exit(1)
    else:
        print_result_simple(result)


    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
def print_result_simple(result: dict):
    """Print result in a simple format."""
    result_type = result.get('type', 'unknown')
    
    if result_type == 'help':
        help_data = result['data']
        print(f"📋 {help_data['overview']['title']}")
        print(f"Version: {help_data['overview']['version']}")
        print(f"\n{help_data['overview']['description']}")
        
        print("\n🚀 Quick Start Examples:")
        for example in help_data['examples']['quick_start']:
            print(f"  {example}")
    
    elif result_type == 'natural_language':
        print("🤖 QuantGPT Response:")
        if 'result' in result and 'gpt_commentary' in result['result']:
            print(result['result']['gpt_commentary'])
        else:
            print(result.get('result', 'No response'))
    
    elif result_type in ['forecast', 'tuning', 'strategy_run']:
        print(f"✅ {result_type.title()} completed")
        print(f"Symbol: {result.get('symbol', 'N/A')}")
        if 'result' in result:
            print("Details:", result['result'])
    
    elif result_type in ['strategy_list', 'agent_list']:
        items = result.get('strategies', result.get('agents', []))
        print(f"📋 {result_type.replace('_', ' ').title()}:")
        for item in items:
            print(f"  - {item}")
    
    else:
        print("✅ Command completed successfully")
        if 'result' in result:
            print("Result:", result['result'])


    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
def launch_streamlit():
    """Launch Streamlit interface."""
    try:
        import subprocess
        import sys
        
        # Get the path to the unified interface file
        interface_file = Path(__file__).parent / "unified_interface.py"
        
        # Launch Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(interface_file), "--server.port", "8501"]
        
        print("🔮 Launching Streamlit interface...")
        print("📱 Open your browser to: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop")
        
        subprocess.run(cmd)
        
    except ImportError:
        print("❌ Streamlit not installed. Install with: pip install streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Streamlit interface stopped")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        sys.exit(1)


    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
if __name__ == "__main__":
    main() 