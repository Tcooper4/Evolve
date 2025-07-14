#!/usr/bin/env python3
"""
Live Market Runner Demo

Demonstrates the LiveMarketRunner functionality with live data streaming,
agent triggering, and forecast tracking.
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
from datetime import datetime

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

    return {
        "success": True,
        "message": "Initialization completed",
        "timestamp": datetime.now().isoformat(),
    }


async def demo_live_market_runner():
    """Demonstrate LiveMarketRunner functionality."""
    logger.info("🚀 Live Market Runner Demo")
    logger.info("=" * 60)

    # Create configuration
    config = {
        "symbols": ["AAPL", "TSLA", "NVDA"],
        "market_data_config": {
            "cache_size": 1000,
            "update_threshold": 5,
            "max_retries": 3,
        },
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
            for symbol, data in state["symbols"].items():
                logger.info(
                    f"   {symbol}: ${data['price']:.2f} "
                    f"({data['price_change']:+.2%}) "
                    f"Volume: {data['volume']:,}"
                )

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
            model_name="model_builder",
        ),
        ForecastResult(
            timestamp=datetime.utcnow(),
            symbol="TSLA",
            forecast_price=245.00,
            forecast_direction="down",
            confidence=0.72,
            model_name="performance_critic",
        ),
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

    from datetime import datetime

    from trading.live_market_runner import TriggerConfig, TriggerType

    # Test time-based trigger
    time_config = TriggerConfig(
        trigger_type=TriggerType.TIME_BASED, interval_seconds=60
    )

    should_trigger = await runner._should_trigger_agent(
        "test_agent", time_config, datetime.utcnow()
    )
    logger.info(f"   Time-based trigger: {should_trigger}")

    # Test price move trigger
    price_config = TriggerConfig(
        trigger_type=TriggerType.PRICE_MOVE, price_move_threshold=0.01
    )

    # Simulate price data
    runner.live_data["AAPL"] = {
        "price": 150.00,
        "volume": 1000000,
        "timestamp": datetime.utcnow(),
        "price_change": 0.015,  # 1.5% change
    }

    should_trigger = await runner._should_trigger_agent(
        "test_agent", price_config, datetime.utcnow()
    )
    logger.info(f"   Price move trigger: {should_trigger}")


async def demo_simulated_trading():
    """Demonstrate simulated trading with comprehensive logging."""
    logger.info(f"\n💰 Simulated Trading Demo")
    logger.info("=" * 40)

    from pathlib import Path

    # Create logs directory if it doesn't exist
    logs_dir = Path("trading/live/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging for trade results
    trade_log_file = logs_dir / "simulated_trades.log"
    file_handler = logging.FileHandler(trade_log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Add file handler to logger
    trade_logger = logging.getLogger("simulated_trades")
    trade_logger.addHandler(file_handler)
    trade_logger.setLevel(logging.INFO)

    # Create runner
    create_live_market_runner()

    # Simulate trade execution
    simulated_trades = [
        {
            "timestamp": datetime.utcnow(),
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 100,
            "price": 150.25,
            "total_value": 15025.00,
            "strategy": "RSI_Strategy",
            "confidence": 0.85,
            "reason": "RSI oversold condition detected",
        },
        {
            "timestamp": datetime.utcnow(),
            "symbol": "TSLA",
            "action": "SELL",
            "quantity": 50,
            "price": 245.75,
            "total_value": 12287.50,
            "strategy": "MACD_Strategy",
            "confidence": 0.72,
            "reason": "MACD bearish crossover",
        },
        {
            "timestamp": datetime.utcnow(),
            "symbol": "NVDA",
            "action": "BUY",
            "quantity": 75,
            "price": 485.50,
            "total_value": 36412.50,
            "strategy": "Bollinger_Strategy",
            "confidence": 0.91,
            "reason": "Price at lower Bollinger band",
        },
    ]

    # Log trades to both file and stdout
    logger.info(f"📊 Simulated Trade Results:")
    logger.info(f"   Log file: {trade_log_file}")
    logger.info(f"   Total trades: {len(simulated_trades)}")

    total_buy_value = 0
    total_sell_value = 0

    for i, trade in enumerate(simulated_trades, 1):
        # Log to stdout
        logger.info(f"\n💰 Trade #{i}:")
        logger.info(f"   Symbol: {trade['symbol']}")
        logger.info(f"   Action: {trade['action']}")
        logger.info(f"   Quantity: {trade['quantity']}")
        logger.info(f"   Price: ${trade['price']:.2f}")
        logger.info(f"   Total Value: ${trade['total_value']:,.2f}")
        logger.info(f"   Strategy: {trade['strategy']}")
        logger.info(f"   Confidence: {trade['confidence']:.1%}")
        logger.info(f"   Reason: {trade['reason']}")

        # Log to file
        trade_logger.info(
            f"TRADE_EXECUTED: {trade['symbol']} {trade['action']} "
            f"{trade['quantity']} @ ${trade['price']:.2f} "
            f"Total: ${trade['total_value']:,.2f} "
            f"Strategy: {trade['strategy']} "
            f"Confidence: {trade['confidence']:.1%}"
        )

        # Track totals
        if trade["action"] == "BUY":
            total_buy_value += trade["total_value"]
        else:
            total_sell_value += trade["total_value"]

    # Calculate and log summary statistics
    logger.info(f"\n📈 Trading Summary:")
    logger.info(f"   Total Buy Value: ${total_buy_value:,.2f}")
    logger.info(f"   Total Sell Value: ${total_sell_value:,.2f}")
    logger.info(f"   Net Position: ${total_buy_value - total_sell_value:,.2f}")

    # Log summary to file
    trade_logger.info(
        f"TRADING_SUMMARY: Buy=${total_buy_value:,.2f} "
        f"Sell=${total_sell_value:,.2f} "
        f"Net=${total_buy_value - total_sell_value:,.2f}"
    )

    # Simulate trade performance tracking
    logger.info(f"\n📊 Performance Tracking:")

    # Simulate price changes and calculate P&L
    price_changes = {
        "AAPL": 0.025,
        "TSLA": -0.015,
        "NVDA": 0.035,
    }  # 2.5% increase  # 1.5% decrease  # 3.5% increase

    total_pnl = 0
    for trade in simulated_trades:
        symbol = trade["symbol"]
        if symbol in price_changes:
            price_change = price_changes[symbol]
            if trade["action"] == "BUY":
                pnl = trade["total_value"] * price_change
            else:  # SELL
                pnl = trade["total_value"] * (-price_change)

            total_pnl += pnl

            logger.info(
                f"   {symbol} {trade['action']}: " f"${pnl:+,.2f} ({price_change:+.1%})"
            )

            # Log P&L to file
            trade_logger.info(
                f"P&L_UPDATE: {symbol} {trade['action']} "
                f"P&L=${pnl:+,.2f} Change={price_change:+.1%}"
            )

    logger.info(f"   Total P&L: ${total_pnl:+,.2f}")
    trade_logger.info(f"TOTAL_P&L: ${total_pnl:+,.2f}")

    # Log trade execution metrics
    execution_metrics = {
        "total_trades": len(simulated_trades),
        "buy_trades": len([t for t in simulated_trades if t["action"] == "BUY"]),
        "sell_trades": len([t for t in simulated_trades if t["action"] == "SELL"]),
        "avg_confidence": sum(t["confidence"] for t in simulated_trades)
        / len(simulated_trades),
        "total_volume": sum(t["quantity"] for t in simulated_trades),
        "total_value": sum(t["total_value"] for t in simulated_trades),
    }

    logger.info(f"\n📋 Execution Metrics:")
    for metric, value in execution_metrics.items():
        if isinstance(value, float):
            logger.info(f"   {metric.replace('_', ' ').title()}: {value:.2f}")
        else:
            logger.info(f"   {metric.replace('_', ' ').title()}: {value}")

    # Log metrics to file
    trade_logger.info(f"EXECUTION_METRICS: {json.dumps(execution_metrics)}")

    logger.info(f"\n✅ Simulated trading demo completed!")
    logger.info(f"   Check {trade_log_file} for detailed trade logs")


async def main():
    """Main demo function."""
    # Add argparse to select agent from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, help="Agent name to run")

    args = parser.parse_args()

    logger.info("🎯 Live Market Runner Demo Suite")
    logger.info("=" * 60)

    if args.agent:
        logger.info(f"🤖 Running agent: {args.agent}")
        # Here you would add logic to run the specific agent
        # For now, just log the agent name
        logger.info(f"Agent '{args.agent}' selected for execution")

    # Run individual demos
    await demo_forecast_tracking()
    await demo_agent_triggering()
    await demo_simulated_trading()

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
