#!/usr/bin/env python3
"""
Test Live Market Runner

Test the LiveMarketRunner functionality.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
import logging

from trading.live_market_runner import create_live_market_runner, ForecastResult

logger = logging.getLogger(__name__)

async def test_live_market_runner():
    """Test the live market runner."""
    logger.info("🧪 Testing Live Market Runner")
    logger.info("=" * 50)
    
    # Create runner
    config = {
        'symbols': ['AAPL', 'TSLA'],
        'market_data_config': {
            'cache_size': 100,
            'update_threshold': 5,
            'max_retries': 3
        }
    }
    
    runner = create_live_market_runner(config)
    
    logger.info("✅ LiveMarketRunner created successfully")
    
    # Test initialization
    logger.info("\n📋 Testing initialization...")
    await runner._initialize_market_data()
    
    # Check live data
    logger.info(f"📊 Live data symbols: {list(runner.live_data.keys())}")
    for symbol, data in runner.live_data.items():
        logger.info(f"   {symbol}: ${data['price']:.2f}")
    
    # Test trigger configurations
    logger.info(f"\n🔧 Testing trigger configurations...")
    logger.info(f"   Trigger configs: {list(runner.trigger_configs.keys())}")
    
    for agent_name, config in runner.trigger_configs.items():
        logger.info(f"   {agent_name}: {config.trigger_type.value}, enabled: {config.enabled}")
    
    # Test forecast tracking
    logger.info(f"\n📈 Testing forecast tracking...")
    
    # Add test forecast
    test_forecast = ForecastResult(
        timestamp=datetime.utcnow(),
        symbol="AAPL",
        forecast_price=150.00,
        forecast_direction="up",
        confidence=0.85,
        model_name="test_model"
    )
    
    runner.forecast_results.append(test_forecast)
    logger.info(f"   Added test forecast for {test_forecast.symbol}")
    
    # Test forecast accuracy calculation
    accuracy = runner.get_forecast_accuracy()
    logger.info(f"   Total forecasts: {accuracy['total_forecasts']}")
    logger.info(f"   Completed forecasts: {accuracy['completed_forecasts']}")
    logger.info(f"   Average accuracy: {accuracy['avg_accuracy']:.2%}")
    
    # Test state retrieval
    logger.info(f"\n📊 Testing state retrieval...")
    state = runner.get_current_state()
    logger.info(f"   State keys: {list(state.keys())}")
    logger.info(f"   Running: {state['running']}")
    logger.info(f"   Forecast count: {state['forecast_count']}")
    
    logger.info("\n✅ All tests completed!")

async def test_forecast_tracking():
    """Test forecast tracking functionality."""
    logger.info("\n📈 Testing Forecast Tracking")
    logger.info("=" * 40)
    
    # Create runner
    runner = create_live_market_runner()
    
    # Add multiple test forecasts
    test_forecasts = [
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
        ),
        ForecastResult(
            timestamp=datetime.utcnow(),
            symbol="AAPL",
            forecast_price=148.00,
            forecast_direction="down",
            confidence=0.68,
            model_name="model_builder"
        )
    ]
    
    runner.forecast_results.extend(test_forecasts)
    logger.info(f"✅ Added {len(test_forecasts)} test forecasts")
    
    # Test accuracy calculation
    accuracy = runner.get_forecast_accuracy()
    logger.info(f"📊 Overall accuracy: {accuracy['avg_accuracy']:.2%}")
    
    # Test symbol-specific accuracy
    aapl_accuracy = runner.get_forecast_accuracy("AAPL")
    logger.info(f"🍎 AAPL accuracy: {aapl_accuracy['avg_accuracy']:.2%}")
    
    # Test model-specific accuracy
    model_forecasts = [f for f in runner.forecast_results if f.model_name == "model_builder"]
    logger.info(f"🤖 Model builder forecasts: {len(model_forecasts)}")
    
    logger.info("✅ Forecast tracking tests completed!")

async def test_agent_triggering():
    """Test agent triggering functionality."""
    logger.info("\n🤖 Testing Agent Triggering")
    logger.info("=" * 40)
    
    # Create runner
    runner = create_live_market_runner()
    
    # Test trigger conditions
    from trading.live_market_runner import TriggerConfig, TriggerType
    from datetime import datetime, timedelta
    
    # Test time-based trigger
    time_config = TriggerConfig(
        trigger_type=TriggerType.TIME_BASED,
        interval_seconds=60
    )
    
    # Should trigger (no previous trigger)
    should_trigger = await runner._should_trigger_agent(
        "test_agent", 
        time_config, 
        datetime.utcnow()
    )
    logger.info(f"   Time-based trigger (no previous): {should_trigger}")
    
    # Should not trigger (recent trigger)
    runner.last_triggers["test_agent"] = datetime.utcnow() - timedelta(seconds=30)
    should_trigger = await runner._should_trigger_agent(
        "test_agent", 
        time_config, 
        datetime.utcnow()
    )
    logger.info(f"   Time-based trigger (recent): {should_trigger}")
    
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
    logger.info(f"   Price move trigger (1.5% change): {should_trigger}")
    
    # Test small price change
    runner.live_data["AAPL"]["price_change"] = 0.005  # 0.5% change
    should_trigger = await runner._should_trigger_agent(
        "test_agent", 
        price_config, 
        datetime.utcnow()
    )
    logger.info(f"   Price move trigger (0.5% change): {should_trigger}")
    
    logger.info("✅ Agent triggering tests completed!")

async def main():
    """Main test function."""
    await test_live_market_runner()
    await test_forecast_tracking()
    await test_agent_triggering()
    
    logger.info(f"\n🎉 All tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(main()) 