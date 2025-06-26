#!/usr/bin/env python3
"""
QuantGPT Demonstration

A simple demonstration of the QuantGPT natural language interface.
"""

import sys
import os
import time
from pathlib import Path

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from services.quant_gpt import QuantGPT


def demo_quant_gpt():
    """Demonstrate QuantGPT functionality."""
    
    print("🤖 QuantGPT Trading Interface Demonstration")
    print("=" * 60)
    print("This demo shows how to use natural language to interact with the trading system.")
    print("=" * 60)
    
    # Initialize QuantGPT
    print("\n🔧 Initializing QuantGPT...")
    quant_gpt = QuantGPT(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        redis_host='localhost',
        redis_port=6379
    )
    
    print("✅ QuantGPT initialized successfully!")
    
    # Demo queries
    demo_queries = [
        {
            'query': "Give me the best model for NVDA over 90 days",
            'description': "Model recommendation query"
        },
        {
            'query': "Should I long TSLA this week?",
            'description': "Trading signal query"
        },
        {
            'query': "Analyze BTCUSDT market conditions",
            'description': "Market analysis query"
        },
        {
            'query': "What's the trading signal for AAPL?",
            'description': "Another trading signal query"
        }
    ]
    
    print(f"\n📝 Running {len(demo_queries)} demo queries...")
    print("-" * 60)
    
    for i, demo in enumerate(demo_queries, 1):
        query = demo['query']
        description = demo['description']
        
        print(f"\n🎯 Demo {i}: {description}")
        print(f"Query: '{query}'")
        print("-" * 50)
        
        try:
            # Process the query
            start_time = time.time()
            result = quant_gpt.process_query(query)
            processing_time = time.time() - start_time
            
            # Display results
            if result.get('status') == 'success':
                parsed = result.get('parsed_intent', {})
                results = result.get('results', {})
                commentary = result.get('gpt_commentary', '')
                
                print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
                print(f"🎯 Intent: {parsed.get('intent', 'unknown')}")
                print(f"📈 Symbol: {parsed.get('symbol', 'N/A')}")
                print(f"⏰ Timeframe: {parsed.get('timeframe', 'N/A')}")
                print(f"📅 Period: {parsed.get('period', 'N/A')}")
                
                # Display action-specific results
                action = results.get('action', 'unknown')
                print(f"🔧 Action: {action}")
                
                if action == 'model_recommendation':
                    best_model = results.get('best_model')
                    if best_model:
                        print(f"🏆 Best Model: {best_model['model_type'].upper()}")
                        print(f"📊 Model Score: {best_model['evaluation'].get('overall_score', 0):.2f}")
                        print(f"📈 Models Built: {results.get('models_built', 0)}")
                        print(f"🔍 Models Evaluated: {results.get('models_evaluated', 0)}")
                
                elif action == 'trading_signal':
                    signal = results.get('signal', {})
                    if signal:
                        print(f"📊 Signal: {signal['signal']}")
                        print(f"💪 Strength: {signal['strength']}")
                        print(f"🎯 Confidence: {signal['confidence']:.1%}")
                        print(f"🧠 Model Score: {signal['model_score']:.2f}")
                        print(f"💭 Reasoning: {signal['reasoning']}")
                
                elif action == 'market_analysis':
                    print(f"📊 Market Data Available: {'Yes' if results.get('market_data') else 'No'}")
                    print(f"📈 Plots Generated: {len(results.get('plots', []))}")
                    print(f"🤖 Model Analysis: {'Available' if results.get('model_analysis') else 'Not available'}")
                
                # Display GPT commentary
                if commentary:
                    print(f"\n🤖 GPT Commentary:")
                    print("-" * 30)
                    print(commentary)
                
                print("✅ Query processed successfully!")
            
            else:
                error = result.get('error', 'Unknown error')
                print(f"❌ Error: {error}")
                print("💡 This might be due to missing services or data.")
            
        except Exception as e:
            print(f"❌ Exception: {e}")
            print("💡 This might be due to missing dependencies or services.")
        
        print("\n" + "=" * 60)
    
    # Show available parameters
    print("\n📋 Available Parameters")
    print("-" * 30)
    print(f"Symbols: {', '.join(quant_gpt.trading_context['available_symbols'])}")
    print(f"Timeframes: {', '.join(quant_gpt.trading_context['available_timeframes'])}")
    print(f"Periods: {', '.join(quant_gpt.trading_context['available_periods'])}")
    print(f"Models: {', '.join(quant_gpt.trading_context['available_models'])}")
    
    # Clean up
    print("\n🧹 Cleaning up...")
    quant_gpt.close()
    print("✅ Demo completed!")
    
    print("\n" + "=" * 60)
    print("🎉 QuantGPT Demonstration Complete!")
    print("=" * 60)
    print("\n💡 Tips for using QuantGPT:")
    print("- Be specific about symbols, timeframes, and periods")
    print("- Ask for model recommendations, trading signals, or market analysis")
    print("- Use natural language - no need to learn specific commands")
    print("- The system will automatically route your query to the right services")
    print("\n🚀 Ready to start trading with natural language!")


def main():
    """Main function."""
    try:
        demo_quant_gpt()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("💡 Make sure Redis is running and all services are available")


if __name__ == "__main__":
    main() 