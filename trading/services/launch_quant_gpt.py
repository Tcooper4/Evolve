#!/usr/bin/env python3
"""
QuantGPT Service Launcher

Launches the QuantGPT interface as a standalone service.
"""

import sys
import os
import signal
import logging
from pathlib import Path

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from services.quant_gpt import QuantGPT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/quant_gpt_service.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    if hasattr(signal_handler, 'quant_gpt'):
        signal_handler.quant_gpt.close()
    sys.exit(0)


def main():
    """Main function to launch the QuantGPT service."""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        logger.info("Starting QuantGPT service...")
        
        # Get OpenAI API key from environment
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            logger.warning("OPENAI_API_KEY not found in environment. GPT commentary will be disabled.")
        
        # Initialize QuantGPT
        quant_gpt = QuantGPT(
            openai_api_key=openai_key,
            redis_host='localhost',
            redis_port=6379,
            redis_db=0
        )
        
        # Store reference for signal handler
        signal_handler.quant_gpt = quant_gpt
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info("QuantGPT service started successfully")
        logger.info("Ready to process natural language queries")
        
        # Interactive mode
        print("\n🤖 QuantGPT Trading Interface")
        print("=" * 50)
        print("Enter your trading queries in natural language:")
        print("Examples:")
        print("- 'Give me the best model for NVDA over 90 days'")
        print("- 'Should I long TSLA this week?'")
        print("- 'Analyze BTCUSDT market conditions'")
        print("- 'What's the trading signal for AAPL?'")
        print("Type 'quit' to exit")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                query = input("\n💬 Query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    logger.info("User requested exit")
                    break
                
                if not query:
                    continue
                
                # Process the query
                logger.info(f"Processing query: {query}")
                result = quant_gpt.process_query(query)
                
                # Display results
                print("\n📊 Results:")
                print("-" * 30)
                
                if result.get('status') == 'success':
                    # Display parsed intent
                    parsed = result.get('parsed_intent', {})
                    print(f"Intent: {parsed.get('intent', 'unknown')}")
                    print(f"Symbol: {parsed.get('symbol', 'N/A')}")
                    print(f"Timeframe: {parsed.get('timeframe', 'N/A')}")
                    print(f"Period: {parsed.get('period', 'N/A')}")
                    
                    # Display results summary
                    results = result.get('results', {})
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
                    commentary = result.get('gpt_commentary', '')
                    if commentary:
                        print(f"\n🤖 GPT Commentary:")
                        print("-" * 30)
                        print(commentary)
                
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"❌ Error: {error}")
                
                print("\n" + "=" * 50)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"❌ Error: {e}")
        
        logger.info("QuantGPT service shutting down...")
        quant_gpt.close()
        
    except Exception as e:
        logger.error(f"Error starting QuantGPT service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 