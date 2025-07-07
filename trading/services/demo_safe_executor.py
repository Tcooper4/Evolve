#!/usr/bin/env python3
"""
Safe Executor Demonstration

Demonstrates safe execution of user-defined models and strategies.
"""

import sys
import os
import time
import json
from pathlib import Path
import logging

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from services.service_client import ServiceClient
from utils.safe_executor import SafeExecutor, ExecutionStatus

logger = logging.getLogger(__name__)

def demo_safe_executor():
    """Demonstrate SafeExecutor functionality."""
    
    logger.info("🛡️ Safe Executor Demonstration")
    logger.info("=" * 60)
    logger.info("This demo shows how to safely execute user-defined models and strategies.")
    logger.info("=" * 60)
    
    # Initialize ServiceClient
    logger.info("\n🔧 Initializing ServiceClient...")
    client = ServiceClient(
        redis_host='localhost',
        redis_port=6379
    )
    
    logger.info("✅ ServiceClient initialized successfully!")
    
    # Demo 1: Safe Model Execution
    logger.info("\n🎯 Demo 1: Safe Model Execution")
    logger.info("-" * 40)
    
    model_code = '''

import numpy as np
import pandas as pd

def main(input_data):
    # Simple moving average model
    prices = input_data.get('prices', [100, 101, 102, 103, 104])
    window = input_data.get('window', 3)
    
    if len(prices) < window:
        return {"error": "Not enough data"}
    
    ma = np.mean(prices[-window:])
    prediction = ma * 1.01  # Simple prediction
    
    return {
        "prediction": prediction,
        "moving_average": ma,
        "confidence": 0.7
    }
'''
    
    input_data = {
        'prices': [100, 101, 102, 103, 104, 105, 106],
        'window': 3
    }
    
    logger.info("Executing simple moving average model...")
    result = client.execute_model_safely(
        model_code=model_code,
        model_name="simple_ma_model",
        input_data=input_data,
        model_type="custom"
    )
    
    if result and result.get('type') == 'model_executed':
        execution_result = result.get('result', {})
        status = execution_result.get('status')
        
        logger.info(f"Status: {status}")
        logger.info(f"Execution Time: {execution_result.get('execution_time', 0):.2f}s")
        
        if status == 'success':
            return_value = execution_result.get('return_value')
            logger.info(f"Prediction: {return_value.get('prediction', 'N/A')}")
            logger.info(f"Moving Average: {return_value.get('moving_average', 'N/A')}")
            logger.info(f"Confidence: {return_value.get('confidence', 'N/A')}")
        else:
            logger.info(f"Error: {execution_result.get('error', 'Unknown error')}")
    
    # Demo 2: Safe Strategy Execution
    logger.info("\n🎯 Demo 2: Safe Strategy Execution")
    logger.info("-" * 40)
    
    strategy_code = '''
import numpy as np

def main(input_data):
    market_data = input_data.get('market_data', {})
    parameters = input_data.get('parameters', {})
    
    # Simple RSI strategy
    prices = market_data.get('prices', [100, 101, 102, 103, 104])
    rsi = market_data.get('rsi', 65)
    
    # Strategy logic
    if rsi > 70:
        signal = "SELL"
        confidence = 0.8
    elif rsi < 30:
        signal = "BUY"
        confidence = 0.8
    else:
        signal = "HOLD"
        confidence = 0.5
    
    return {
        "signal": signal,
        "confidence": confidence,
        "rsi": rsi,
        "reasoning": f"RSI is {rsi}, indicating {'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'} conditions"
    }
'''
    
    market_data = {
        'prices': [100, 101, 102, 103, 104, 105, 106],
        'rsi': 75
    }
    
    parameters = {
        'rsi_oversold': 30,
        'rsi_overbought': 70
    }
    
    logger.info("Executing RSI strategy...")
    result = client.execute_strategy_safely(
        strategy_code=strategy_code,
        strategy_name="rsi_strategy",
        market_data=market_data,
        parameters=parameters
    )
    
    if result and result.get('type') == 'strategy_executed':
        execution_result = result.get('result', {})
        status = execution_result.get('status')
        
        logger.info(f"Status: {status}")
        logger.info(f"Execution Time: {execution_result.get('execution_time', 0):.2f}s")
        
        if status == 'success':
            return_value = execution_result.get('return_value')
            logger.info(f"Signal: {return_value.get('signal', 'N/A')}")
            logger.info(f"Confidence: {return_value.get('confidence', 'N/A')}")
            logger.info(f"RSI: {return_value.get('rsi', 'N/A')}")
            logger.info(f"Reasoning: {return_value.get('reasoning', 'N/A')}")
        else:
            logger.info(f"Error: {execution_result.get('error', 'Unknown error')}")
    
    # Demo 3: Safe Indicator Execution
    logger.info("\n🎯 Demo 3: Safe Indicator Execution")
    logger.info("-" * 40)
    
    indicator_code = '''
import numpy as np

def main(input_data):
    price_data = input_data.get('price_data', {})
    parameters = input_data.get('parameters', {})
    
    # Calculate MACD indicator
    prices = price_data.get('prices', [100, 101, 102, 103, 104, 105, 106])
    fast_period = parameters.get('fast_period', 12)
    slow_period = parameters.get('slow_period', 26)
    
    if len(prices) < slow_period:
        return {"error": "Not enough data for MACD calculation"}
    
    # Simple MACD calculation
    fast_ma = np.mean(prices[-fast_period:])
    slow_ma = np.mean(prices[-slow_period:])
    macd = fast_ma - slow_ma
    
    return {
        "macd": macd,
        "fast_ma": fast_ma,
        "slow_ma": slow_ma,
        "signal": "BULLISH" if macd > 0 else "BEARISH"
    }
'''
    
    price_data = {
        'prices': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130]
    }
    
    parameters = {
        'fast_period': 12,
        'slow_period': 26
    }
    
    logger.info("Executing MACD indicator...")
    result = client.execute_indicator_safely(
        indicator_code=indicator_code,
        indicator_name="macd_indicator",
        price_data=price_data,
        parameters=parameters
    )
    
    if result and result.get('type') == 'indicator_executed':
        execution_result = result.get('result', {})
        status = execution_result.get('status')
        
        logger.info(f"Status: {status}")
        logger.info(f"Execution Time: {execution_result.get('execution_time', 0):.2f}s")
        
        if status == 'success':
            return_value = execution_result.get('return_value')
            logger.info(f"MACD: {return_value.get('macd', 'N/A')}")
            logger.info(f"Fast MA: {return_value.get('fast_ma', 'N/A')}")
            logger.info(f"Slow MA: {return_value.get('slow_ma', 'N/A')}")
            logger.info(f"Signal: {return_value.get('signal', 'N/A')}")
        else:
            logger.info(f"Error: {execution_result.get('error', 'Unknown error')}")
    
    # Demo 4: Error Handling (Dangerous Code)
    logger.info("\n🎯 Demo 4: Error Handling (Dangerous Code)")
    logger.info("-" * 40)
    
    dangerous_code = '''
import os
import subprocess

def main(input_data):
    # This should be blocked by the safe executor
    os.system("rm -rf /")  # Dangerous command
    return {"status": "dangerous"}
'''
    
    logger.info("Attempting to execute dangerous code...")
    result = client.execute_model_safely(
        model_code=dangerous_code,
        model_name="dangerous_model",
        input_data={},
        model_type="custom"
    )
    
    if result and result.get('type') == 'model_executed':
        execution_result = result.get('result', {})
        status = execution_result.get('status')
        
        logger.info(f"Status: {status}")
        
        if status == 'validation_error':
            logger.info("✅ Dangerous code was properly blocked!")
            logger.info(f"Error: {execution_result.get('error', 'Unknown error')}")
        else:
            logger.info("❌ Dangerous code was not properly blocked")
    else:
        logger.info("❌ No response from service")
    
    # Demo 5: Timeout Handling
    logger.info("\n🎯 Demo 5: Timeout Handling")
    logger.info("-" * 40)
    
    timeout_code = '''
import time

def main(input_data):
    # This should timeout
    time.sleep(10)  # Sleep for 10 seconds
    return {"status": "completed"}
'''
    
    logger.info("Executing code that should timeout...")
    result = client.execute_model_safely(
        model_code=timeout_code,
        model_name="timeout_model",
        input_data={},
        model_type="custom"
    )
    
    if result and result.get('type') == 'model_executed':
        execution_result = result.get('result', {})
        status = execution_result.get('status')
        
        logger.info(f"Status: {status}")
        logger.info(f"Execution Time: {execution_result.get('execution_time', 0):.2f}s")
        
        if status == 'timeout':
            logger.info("✅ Code was properly timed out!")
        else:
            logger.info(f"Unexpected status: {status}")
    else:
        logger.info("❌ No response from service")
    
    # Demo 6: Get Statistics
    logger.info("\n🎯 Demo 6: Get SafeExecutor Statistics")
    logger.info("-" * 40)
    
    stats_result = client.get_safe_executor_statistics()
    if stats_result and stats_result.get('type') == 'statistics':
        stats = stats_result.get('statistics', {})
        logger.info(f"Total Executions: {stats.get('total_executions', 0)}")
        logger.info(f"Successful Executions: {stats.get('successful_executions', 0)}")
        logger.info(f"Failed Executions: {stats.get('failed_executions', 0)}")
        logger.info(f"Success Rate: {stats.get('success_rate', 0):.1%}")
        logger.info(f"Total Execution Time: {stats.get('total_execution_time', 0):.2f}s")
        logger.info(f"Average Execution Time: {stats.get('average_execution_time', 0):.2f}s")
    else:
        logger.info("❌ Could not retrieve statistics")
    
    # Clean up
    logger.info("\n🧹 Cleaning up...")
    cleanup_result = client.cleanup_safe_executor()
    if cleanup_result and cleanup_result.get('type') == 'cleanup_completed':
        logger.info("✅ Cleanup completed successfully")
    else:
        logger.info("❌ Cleanup failed")
    
    client.close()
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Safe Executor Demonstration Complete!")
    logger.info("=" * 60)
    logger.info("\n💡 Key Features Demonstrated:")
    logger.info("- Safe execution of user-defined models")
    logger.info("- Strategy execution with market data")
    logger.info("- Technical indicator calculation")
    logger.info("- Security validation (blocks dangerous code)")
    logger.info("- Timeout protection")
    logger.info("- Resource monitoring and statistics")
    logger.info("\n🚀 Ready to safely execute custom trading code!")

def main():
    """Main function."""
    try:
        demo_safe_executor()
        return {
            "status": "completed",
            "demo_type": "safe_executor",
            "result": "success"
        }
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  Demo interrupted by user")
        return {
            "status": "interrupted",
            "demo_type": "safe_executor",
            "result": "user_interrupted"
        }
    except Exception as e:
        logger.error(f"\n❌ Demo failed: {e}")
        logger.error("💡 Make sure Redis is running and the SafeExecutor service is available")
        return {
            "status": "failed",
            "demo_type": "safe_executor",
            "error": str(e),
            "result": "error"
        }

if __name__ == "__main__":
    main() 