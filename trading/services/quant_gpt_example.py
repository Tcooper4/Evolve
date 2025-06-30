#!/usr/bin/env python3
"""
QuantGPT Usage Example

Demonstrates how to use the QuantGPT interface for natural language
trading queries.
"""

import sys
import os
import time
from pathlib import Path

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from services.quant_gpt import QuantGPT
from services.service_client import ServiceClient


def example_queries():
    """Run example queries through QuantGPT."""
    
    print("🤖 QuantGPT Trading Interface Examples")
    print("=" * 60)
    
    # Initialize QuantGPT (direct usage)
    print("Initializing QuantGPT...")
    quant_gpt = QuantGPT(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        redis_host='localhost',
        redis_port=6379
    )
    
    # Example queries
    queries = [
        "Give me the best model for NVDA over 90 days",
        "Should I long TSLA this week?",
        "Analyze BTCUSDT market conditions",
        "What's the trading signal for AAPL?",
        "Find the optimal model for GOOGL on 1h timeframe"
    ]
    
    print(f"\nRunning {len(queries)} example queries...")
    print("-" * 60)
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 40)
        
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
                
                print(f"✅ Success (processed in {processing_time:.2f}s)")
                print(f"Intent: {parsed.get('intent', 'unknown')}")
                print(f"Symbol: {parsed.get('symbol', 'N/A')}")
                print(f"Timeframe: {parsed.get('timeframe', 'N/A')}")
                print(f"Period: {parsed.get('period', 'N/A')}")
                
                # Display action-specific results
                action = results.get('action', 'unknown')
                print(f"Action: {action}")
                
                if action == 'model_recommendation':
                    best_model = results.get('best_model')
                    if best_model:
                        print(f"Best Model: {best_model['model_type'].upper()}")
                        print(f"Model Score: {best_model['evaluation'].get('overall_score', 0):.2f}")
                
                elif action == 'trading_signal':
                    signal = results.get('signal', {})
                    if signal:
                        print(f"Signal: {signal['signal']}")
                        print(f"Strength: {signal['strength']}")
                        print(f"Confidence: {signal['confidence']:.1%}")
                
                # Display GPT commentary
                if commentary:
                    print(f"\n🤖 GPT Commentary:")
                    print("-" * 30)
                    print(commentary)
            
            else:
                error = result.get('error', 'Unknown error')
                print(f"❌ Error: {error}")
            
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print("\n" + "=" * 60)
    
    # Clean up
    quant_gpt.close()


    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
def example_service_client():
    """Demonstrate QuantGPT usage through ServiceClient."""
    
    print("\n🔗 QuantGPT via ServiceClient Example")
    print("=" * 60)
    
    # Initialize ServiceClient
    print("Initializing ServiceClient...")
    client = ServiceClient(
        redis_host='localhost',
        redis_port=6379
    )
    
    # Example queries for service client
    service_queries = [
        "What's the best model for MSFT?",
        "Should I buy AMZN now?",
        "Analyze ETHUSDT market"
    ]
    
    print(f"\nRunning {len(service_queries)} queries via ServiceClient...")
    print("-" * 60)
    
    for i, query in enumerate(service_queries, 1):
        print(f"\n📝 Service Query {i}: {query}")
        print("-" * 40)
        
        try:
            # Process query via service
            start_time = time.time()
            result = client.process_natural_language_query(query)
            processing_time = time.time() - start_time
            
            if result:
                print(f"✅ Service Response (processed in {processing_time:.2f}s)")
                print(f"Type: {result.get('type', 'unknown')}")
                
                if result.get('type') == 'query_processed':
                    query_result = result.get('result', {})
                    if query_result.get('status') == 'success':
                        parsed = query_result.get('parsed_intent', {})
                        print(f"Intent: {parsed.get('intent', 'unknown')}")
                        print(f"Symbol: {parsed.get('symbol', 'N/A')}")
                        
                        commentary = query_result.get('gpt_commentary', '')
                        if commentary:
                            print(f"\n🤖 GPT Commentary:")
                            print("-" * 30)
                            print(commentary[:200] + "..." if len(commentary) > 200 else commentary)
                    else:
                        print(f"❌ Query Error: {query_result.get('error', 'Unknown error')}")
                else:
                    print(f"❌ Service Error: {result.get('error', 'Unknown error')}")
            else:
                print("❌ No response from service")
            
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print("\n" + "=" * 60)
    
    # Get available symbols
    print("\n📊 Available Symbols and Parameters")
    print("-" * 40)
    
    try:
        symbols_result = client.get_available_symbols()
        if symbols_result and symbols_result.get('type') == 'available_symbols':
            symbols = symbols_result.get('symbols', [])
            timeframes = symbols_result.get('timeframes', [])
            periods = symbols_result.get('periods', [])
            models = symbols_result.get('models', [])
            
            print(f"Symbols: {', '.join(symbols)}")
            print(f"Timeframes: {', '.join(timeframes)}")
            print(f"Periods: {', '.join(periods)}")
            print(f"Models: {', '.join(models)}")
        else:
            print("❌ Could not retrieve available symbols")
    
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Clean up
    client.close()


    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
def main():
    """Main function to run examples."""
    try:
        # Check if Redis is available
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379)
            r.ping()
            print("✅ Redis connection successful")
        except Exception as e:
            print(f"⚠️  Redis not available: {e}")
            print("Running direct QuantGPT examples only...")
            example_queries()
            return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        
        # Run both examples
        example_queries()
        example_service_client()
        
        print("\n🎉 All examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")


if __name__ == "__main__":
    main() 