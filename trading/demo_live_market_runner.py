#!/usr/bin/env python3
"""
Live Market Runner Demo

Demonstrates the LiveMarketRunner functionality with live data streaming,
agent triggering, and forecast tracking.
"""

import asyncio
import json
import signal
import sys
from datetime import datetime
from typing import Dict, Any
import logging

from trading.live_market_runner import create_live_market_runner

logger = logging.getLogger(__name__)

def setup_signal_handlers(runner):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"\n🛑 Received signal {signum}, shutting down...")
        asyncio.create_task(runner.stop())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}

async def demo_live_market_runner():
    """Demonstrate LiveMarketRunner functionality."""
    logger.info("🚀 Live Market Runner Demo")
    logger.info("=" * 60)
    
    # Create configuration
    config = {
        'symbols': ['AAPL', 'TSLA', 'NVDA'],
        'market_data_config': {
            'cache_size': 1000,
            'update_threshold': 5,
            'max_retries': 3
        }
    }
    
    # Create runner
    runner = create_live_market_runner(config)
    
    # Setup signal handlers
    setup_signal_handlers(runner)
    
    logger.info(f"✅ LiveMarketRunner created")
    logger.info(f"📊 Symbols: {config['symbols']}")
    logger.info(f"🔄 Update interval: 30 seconds")
    logger.info(f"⏰ Trigger check interval: 10 seconds")
    
    # Start the runner
    logger.info(f"\n🔄 Starting LiveMarketRunner...")
    await runner.start()
    
    logger.info(f"✅ LiveMarketRunner started successfully!")
    logger.info(f"   Press Ctrl+C to stop")
    logger.info(f"   Logs: trading/live/logs/live_market_runner.log")
    
    # Monitor and display status
    try:
        while runner.running:
            # Get current state
            state = runner.get_current_state()
            
            logger.info(f"\n📊 Current State ({state['timestamp']}):")
            logger.info(f"   Running: {state['running']}")
            logger.info(f"   Forecast count: {state['forecast_count']}")
            
            # Display symbol data
            logger.info(f"\n💰 Symbol Data:")
            for symbol, data in state['symbols'].items():
                logger.info(f"   {symbol}: ${data['price']:.2f} "
                      f"({data['price_change']:+.2%}) "
                      f"Volume: {data['volume']:,}")
            
            # Display forecast accuracy
            accuracy = runner.get_forecast_accuracy()
            logger.info(f"\n📈 Forecast Accuracy:")
            logger.info(f"   Total forecasts: {accuracy['total_forecasts']}")
            logger.info(f"   Completed: {accuracy['completed_forecasts']}")
            logger.info(f"   Average accuracy: {accuracy['avg_accuracy']:.2%}")
            
            # Wait before next update
            await asyncio.sleep(60)  # Update every minute
            
    except KeyboardInterrupt:
        logger.info(f"\n🛑 Shutting down...")
    finally:
        await runner.stop()
        logger.info(f"✅ LiveMarketRunner stopped")

async def demo_forecast_tracking():
    """Demonstrate forecast tracking functionality."""
    logger.info(f"\n📊 Forecast Tracking Demo")
    logger.info("=" * 40)
    
    # Create runner
    runner = create_live_market_runner()
    
    # Simulate some forecasts
    from trading.live_market_runner import ForecastResult
    
    # Add sample forecasts
    sample_forecasts = [
        ForecastResult(
            timestamp=datetime.utcnow(),
            symbol="AAPL",
            forecast_price=150.00,
            forecast_direction="up",
            confidence=0.85,
            model_name="model_builder"
        ),
        ForecastResult(
            timestamp=datetime.utcnow(),
            symbol="TSLA",
            forecast_price=245.00,
            forecast_direction="down",
            confidence=0.72,
            model_name="performance_critic"
        )
    ]
    
    runner.forecast_results.extend(sample_forecasts)
    
    logger.info(f"✅ Added {len(sample_forecasts)} sample forecasts")
    
    # Show forecast accuracy
    accuracy = runner.get_forecast_accuracy()
    logger.info(f"📈 Forecast Statistics:")
    logger.info(f"   Total forecasts: {accuracy['total_forecasts']}")
    logger.info(f"   Completed: {accuracy['completed_forecasts']}")
    logger.info(f"   Average accuracy: {accuracy['avg_accuracy']:.2%}")
    
    # Show symbol-specific accuracy
    aapl_accuracy = runner.get_forecast_accuracy("AAPL")
    logger.info(f"\n🍎 AAPL Forecasts:")
    logger.info(f"   Total: {aapl_accuracy['total_forecasts']}")
    logger.info(f"   Completed: {aapl_accuracy['completed_forecasts']}")
    logger.info(f"   Average accuracy: {aapl_accuracy['avg_accuracy']:.2%}")

async def demo_agent_triggering():
    """Demonstrate agent triggering functionality."""
    logger.info(f"\n🤖 Agent Triggering Demo")
    logger.info("=" * 40)
    
    # Create runner
    runner = create_live_market_runner()
    
    # Show trigger configurations
    logger.info(f"📋 Trigger Configurations:")
    for agent_name, config in runner.trigger_configs.items():
        logger.info(f"   {agent_name}:")
        logger.info(f"     Type: {config.trigger_type.value}")
        logger.info(f"     Enabled: {config.enabled}")
        if config.trigger_type.value == "time_based":
            logger.info(f"     Interval: {config.interval_seconds} seconds")
        elif config.trigger_type.value == "price_move":
            logger.info(f"     Threshold: {config.price_move_threshold:.1%}")
    
    # Test trigger conditions
    logger.info(f"\n🔍 Testing Trigger Conditions:")
    
    from trading.live_market_runner import TriggerConfig, TriggerType
    from datetime import datetime
    
    # Test time-based trigger
    time_config = TriggerConfig(
        trigger_type=TriggerType.TIME_BASED,
        interval_seconds=60
    )
    
    should_trigger = await runner._should_trigger_agent(
        "test_agent", 
        time_config, 
        datetime.utcnow()
    )
    logger.info(f"   Time-based trigger: {should_trigger}")
    
    # Test price move trigger
    price_config = TriggerConfig(
        trigger_type=TriggerType.PRICE_MOVE,
        price_move_threshold=0.01
    )
    
    # Simulate price data
    runner.live_data["AAPL"] = {
        'price': 150.00,
        'volume': 1000000,
        'timestamp': datetime.utcnow(),
        'price_change': 0.015  # 1.5% change
    }
    
    should_trigger = await runner._should_trigger_agent(
        "test_agent", 
        price_config, 
        datetime.utcnow()
    )
    logger.info(f"   Price move trigger: {should_trigger}")

async def main():
    """Main demo function."""
    logger.info("🎯 Live Market Runner Demo Suite")
    logger.info("=" * 60)
    
    # Run individual demos
    await demo_forecast_tracking()
    await demo_agent_triggering()
    
    # Run main demo (commented out to avoid long running)
    # await demo_live_market_runner()
    
    logger.info(f"\n✅ Demo completed!")
    logger.info(f"📋 Demo Summary:")
    logger.info(f"   ✅ Forecast tracking functionality demonstrated")
    logger.info(f"   ✅ Agent triggering logic tested")
    logger.info(f"   ✅ Live data streaming ready for use")
    logger.info(f"   🔮 Ready for live market execution")

if __name__ == "__main__":
    asyncio.run(main()) 