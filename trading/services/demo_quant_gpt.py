#!/usr/bin/env python3
"""
QuantGPT Demonstration

A simple demonstration of the QuantGPT natural language interface.
"""

import sys
import os
import time
from pathlib import Path
import datetime
import logging

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from services.quant_gpt import QuantGPT

logger = logging.getLogger(__name__)

def demo_quant_gpt() -> dict:
    """Demonstrate QuantGPT functionality."""
    
    try:
        logger.info("🤖 QuantGPT Trading Interface Demonstration")
        logger.info("=" * 60)
        logger.info("This demo shows how to use natural language to interact with the trading system.")
        logger.info("=" * 60)
        
        # Initialize QuantGPT
        logger.info("\n🔧 Initializing QuantGPT...")
        quant_gpt = QuantGPT(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            redis_host='localhost',
            redis_port=6379
        )
        
        logger.info("✅ QuantGPT initialized successfully!")
        
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
        
        logger.info(f"\n📝 Running {len(demo_queries)} demo queries...")
        logger.info("-" * 60)
        
        for i, demo in enumerate(demo_queries, 1):
            query = demo['query']
            description = demo['description']
            
            logger.info(f"\n🎯 Demo {i}: {description}")
            logger.info(f"Query: '{query}'")
            logger.info("-" * 50)
            
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
                    
                    logger.info(f"⏱️  Processing Time: {processing_time:.2f} seconds")
                    logger.info(f"🎯 Intent: {parsed.get('intent', 'unknown')}")
                    logger.info(f"📈 Symbol: {parsed.get('symbol', 'N/A')}")
                    logger.info(f"⏰ Timeframe: {parsed.get('timeframe', 'N/A')}")
                    logger.info(f"📅 Period: {parsed.get('period', 'N/A')}")
                    
                    # Display action-specific results
                    action = results.get('action', 'unknown')
                    logger.info(f"�� Action: {action}")
                    
                    if action == 'model_recommendation':
                        best_model = results.get('best_model')
                        if best_model:
                            logger.info(f"🏆 Best Model: {best_model['model_type'].upper()}")
                            logger.info(f"📊 Model Score: {best_model['evaluation'].get('overall_score', 0):.2f}")
                            logger.info(f"📈 Models Built: {results.get('models_built', 0)}")
                            logger.info(f"🔍 Models Evaluated: {results.get('models_evaluated', 0)}")
                    
                    elif action == 'trading_signal':
                        signal = results.get('signal', {})
                        if signal:
                            logger.info(f"📊 Signal: {signal['signal']}")
                            logger.info(f"💪 Strength: {signal['strength']}")
                            logger.info(f"🎯 Confidence: {signal['confidence']:.1%}")
                            logger.info(f"🧠 Model Score: {signal['model_score']:.2f}")
                            logger.info(f"💭 Reasoning: {signal['reasoning']}")
                    
                    elif action == 'market_analysis':
                        logger.info(f"📊 Market Data Available: {'Yes' if results.get('market_data') else 'No'}")
                        logger.info(f"📈 Plots Generated: {len(results.get('plots', []))}")
                        logger.info(f"🤖 Model Analysis: {'Available' if results.get('model_analysis') else 'Not available'}")
                    
                    # Display GPT commentary
                    if commentary:
                        logger.info(f"\n🤖 GPT Commentary:")
                        logger.info("-" * 30)
                        logger.info(commentary)
                    
                    logger.info("✅ Query processed successfully!")
                
                else:
                    error = result.get('error', 'Unknown error')
                    logger.error(f"❌ Error: {error}")
                    logger.error("💡 This might be due to missing services or data.")
                
            except Exception as e:
                logger.error(f"❌ Exception: {e}")
                logger.error("💡 This might be due to missing dependencies or services.")
            
            logger.info("\n" + "=" * 60)
        
        # Show available parameters
        logger.info("\n📋 Available Parameters")
        logger.info("-" * 30)
        logger.info(f"Symbols: {', '.join(quant_gpt.trading_context['available_symbols'])}")
        logger.info(f"Timeframes: {', '.join(quant_gpt.trading_context['available_timeframes'])}")
        logger.info(f"Periods: {', '.join(quant_gpt.trading_context['available_periods'])}")
        logger.info(f"Models: {', '.join(quant_gpt.trading_context['available_models'])}")
        
        # Clean up
        logger.info("\n🧹 Cleaning up...")
        quant_gpt.close()
        logger.info("✅ Demo completed!")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 QuantGPT Demonstration Complete!")
        logger.info("=" * 60)
        logger.info("\n💡 Tips for using QuantGPT:")
        logger.info("- Be specific about symbols, timeframes, and periods")
        logger.info("- Ask for model recommendations, trading signals, or market analysis")
        logger.info("- Use natural language - no need to learn specific commands")
        logger.info("- The system will automatically route your query to the right services")
        logger.info("\n🚀 Ready to start trading with natural language!")
        return {
            'success': True,
            'message': 'QuantGPT demo completed successfully',
            'timestamp': datetime.datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.datetime.now().isoformat()
        }

def main() -> dict:
    """Main function."""
    try:
        result = demo_quant_gpt()
        return {
            'success': True,
            'message': 'Demo main completed',
            'timestamp': datetime.datetime.now().isoformat(),
            'demo_result': result
        }
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  Demo interrupted by user")
        return {
            'success': False,
            'error': 'Demo interrupted by user',
            'timestamp': datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"\n❌ Demo failed: {e}")
        logger.error("💡 Make sure Redis is running and all services are available")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.datetime.now().isoformat()
        }

if __name__ == "__main__":
    main() 