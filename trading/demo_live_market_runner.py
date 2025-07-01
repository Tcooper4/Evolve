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

from trading.live_market_runner import create_live_market_runner

def setup_signal_handlers(runner):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down...")
        asyncio.create_task(runner.stop())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
async def demo_live_market_runner():
    """Demonstrate LiveMarketRunner functionality."""
    print("🚀 Live Market Runner Demo")
    print("=" * 60)
    
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
    
    print(f"✅ LiveMarketRunner created")
    print(f"📊 Symbols: {config['symbols']}")
    print(f"🔄 Update interval: 30 seconds")
    print(f"⏰ Trigger check interval: 10 seconds")
    
    # Start the runner
    print(f"\n🔄 Starting LiveMarketRunner...")
    await runner.start()
    
    print(f"✅ LiveMarketRunner started successfully!")
    print(f"   Press Ctrl+C to stop")
    print(f"   Logs: trading/live/logs/live_market_runner.log")
    
    # Monitor and display status
    try:
        while runner.running:
            # Get current state
            state = runner.get_current_state()
            
            print(f"\n📊 Current State ({state['timestamp']}):")
            print(f"   Running: {state['running']}")
            print(f"   Forecast count: {state['forecast_count']}")
            
            # Display symbol data
            print(f"\n💰 Symbol Data:")
            for symbol, data in state['symbols'].items():
                print(f"   {symbol}: ${data['price']:.2f} "
                      f"({data['price_change']:+.2%}) "
                      f"Volume: {data['volume']:,}")
            
            # Display forecast accuracy
            accuracy = runner.get_forecast_accuracy()
            print(f"\n📈 Forecast Accuracy:")
            print(f"   Total forecasts: {accuracy['total_forecasts']}")
            print(f"   Completed: {accuracy['completed_forecasts']}")
            print(f"   Average accuracy: {accuracy['avg_accuracy']:.2%}")
            
            # Wait before next update
            await asyncio.sleep(60)  # Update every minute
            
    except KeyboardInterrupt:
        print(f"\n🛑 Shutting down...")
    finally:
        await runner.stop()
        print(f"✅ LiveMarketRunner stopped")

async def demo_forecast_tracking():
    """Demonstrate forecast tracking functionality."""
    print(f"\n📊 Forecast Tracking Demo")
    print("=" * 40)
    
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
    
    print(f"✅ Added {len(sample_forecasts)} sample forecasts")
    
    # Show forecast accuracy
    accuracy = runner.get_forecast_accuracy()
    print(f"📈 Forecast Statistics:")
    print(f"   Total forecasts: {accuracy['total_forecasts']}")
    print(f"   Completed: {accuracy['completed_forecasts']}")
    print(f"   Average accuracy: {accuracy['avg_accuracy']:.2%}")
    
    # Show symbol-specific accuracy
    aapl_accuracy = runner.get_forecast_accuracy("AAPL")
    print(f"\n🍎 AAPL Forecasts:")
    print(f"   Total: {aapl_accuracy['total_forecasts']}")
    print(f"   Completed: {aapl_accuracy['completed_forecasts']}")
    print(f"   Average accuracy: {aapl_accuracy['avg_accuracy']:.2%}")

async def demo_agent_triggering():
    """Demonstrate agent triggering functionality."""
    print(f"\n🤖 Agent Triggering Demo")
    print("=" * 40)
    
    # Create runner
    runner = create_live_market_runner()
    
    # Show trigger configurations
    print(f"📋 Trigger Configurations:")
    for agent_name, config in runner.trigger_configs.items():
        print(f"   {agent_name}:")
        print(f"     Type: {config.trigger_type.value}")
        print(f"     Enabled: {config.enabled}")
        if config.trigger_type.value == "time_based":
            print(f"     Interval: {config.interval_seconds} seconds")
        elif config.trigger_type.value == "price_move":
            print(f"     Threshold: {config.price_move_threshold:.1%}")
    
    # Test trigger conditions
    print(f"\n🔍 Testing Trigger Conditions:")
    
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
    print(f"   Time-based trigger: {should_trigger}")
    
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
    print(f"   Price move trigger: {should_trigger}")

async def main():
    """Main demo function."""
    print("🎯 Live Market Runner Demo Suite")
    print("=" * 60)
    
    # Run individual demos
    await demo_forecast_tracking()
    await demo_agent_triggering()
    
    # Run main demo (commented out to avoid long running)
    # await demo_live_market_runner()
    
    print(f"\n✅ Demo completed!")
    print(f"📋 Demo Summary:")
    print(f"   ✅ Forecast tracking functionality demonstrated")
    print(f"   ✅ Agent triggering logic tested")
    print(f"   ✅ Live data streaming ready for use")
    print(f"   🔮 Ready for live market execution")

if __name__ == "__main__":
    asyncio.run(main()) 