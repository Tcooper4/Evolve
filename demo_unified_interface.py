#!/usr/bin/env python3
"""
Demo Unified Interface

Demonstrates the unified interface capabilities.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def demo_unified_interface():
    """Demonstrate unified interface functionality."""
    print("🔮 Evolve Unified Interface Demo")
    print("=" * 60)
    print("This demo shows how to access all features through one interface.")
    print("=" * 60)
    
    try:
        # Import unified interface
        from unified_interface import UnifiedInterface
        print("✅ Successfully imported UnifiedInterface")
        
        # Initialize interface
        interface = UnifiedInterface()
        print("✅ Successfully initialized interface")
        
        # Demo commands
        demo_commands = [
            {
                'command': 'help',
                'description': 'Get help information'
            },
            {
                'command': 'forecast AAPL 7d',
                'description': 'Generate 7-day forecast for AAPL'
            },
            {
                'command': 'tune model lstm AAPL',
                'description': 'Tune LSTM model for AAPL'
            },
            {
                'command': 'strategy list',
                'description': 'List available strategies'
            },
            {
                'command': 'agent list',
                'description': 'List available agents'
            },
            {
                'command': 'portfolio status',
                'description': 'Get portfolio status'
            },
            {
                'command': 'What\'s the best model for TSLA?',
                'description': 'Natural language query via QuantGPT'
            },
            {
                'command': 'status',
                'description': 'Check system status'
            }
        ]
        
        print(f"\n📝 Running {len(demo_commands)} demo commands...")
        print("-" * 60)
        
        for i, demo in enumerate(demo_commands, 1):
            print(f"\n{i}. {demo['description']}")
            print(f"   Command: {demo['command']}")
            print("   " + "-" * 40)
            
            # Execute command
            result = interface.process_command(demo['command'])
            
            if result.get('status') == 'error':
                print(f"   ❌ Error: {result.get('error')}")
            else:
                print(f"   ✅ Success!")
                result_type = result.get('type', 'unknown')
                print(f"   Type: {result_type}")
                
                # Show relevant result info
                if result_type == 'help':
                    help_data = result['data']
                    print(f"   Features: {len(help_data['features'])} available")
                elif result_type == 'natural_language':
                    print("   QuantGPT processed the query")
                elif result_type in ['forecast', 'tuning', 'strategy_run']:
                    symbol = result.get('symbol', 'N/A')
                    print(f"   Symbol: {symbol}")
                elif result_type in ['strategy_list', 'agent_list']:
                    items = result.get('strategies', result.get('agents', []))
                    print(f"   Items: {len(items)} found")
                elif result_type == 'status':
                    print("   System status retrieved")
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("\n📋 What you can do next:")
        print("1. Launch Streamlit interface: streamlit run app.py")
        print("2. Use terminal interface: python unified_interface.py --terminal")
        print("3. Execute commands: python unified_interface.py --command 'help'")
        print("4. Ask questions: python unified_interface.py --command 'What is the best model for AAPL?'")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure unified_interface.py is in the current directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def show_usage_examples():
    """Show usage examples."""
    print("\n📚 Usage Examples")
    print("=" * 60)
    
    examples = [
        {
            'category': 'Basic Commands',
            'examples': [
                'help',
                'status',
                'forecast AAPL 7d',
                'tune model lstm AAPL'
            ]
        },
        {
            'category': 'Strategy Commands',
            'examples': [
                'strategy list',
                'strategy run bollinger AAPL',
                'backtest macd TSLA'
            ]
        },
        {
            'category': 'Agent Commands',
            'examples': [
                'agent list',
                'agent status',
                'start agent model_builder'
            ]
        },
        {
            'category': 'Portfolio Commands',
            'examples': [
                'portfolio status',
                'portfolio rebalance',
                'risk analysis'
            ]
        },
        {
            'category': 'Natural Language Queries',
            'examples': [
                'What\'s the best model for TSLA?',
                'Should I buy AAPL now?',
                'Analyze BTCUSDT market conditions',
                'What\'s the trading signal for NVDA?'
            ]
        }
    ]
    
    for category in examples:
        print(f"\n{category['category']}:")
        for example in category['examples']:
            print(f"  {example}")

def main():
    """Main demo function."""
    demo_unified_interface()
    show_usage_examples()

if __name__ == "__main__":
    main() 