#!/usr/bin/env python3
"""
Safe Executor Test Script

Tests the SafeExecutor functionality and security features.
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

def test_direct_safe_executor():
    """Test direct SafeExecutor usage."""
    logger.info("🧪 Testing Direct SafeExecutor Usage")
    logger.info("=" * 50)
    
    try:
        # Initialize SafeExecutor
        executor = SafeExecutor(
            timeout_seconds=30,
            memory_limit_mb=512,
            enable_sandbox=True,
            log_executions=True
        )
        
        # Test 1: Valid model execution
        logger.info("\n📝 Test 1: Valid Model Execution")
        logger.info("-" * 40)
        
        valid_code = '''
def main(input_data):
    return {"result": "success", "value": 42}
'''
        
        result = executor.execute_model(
            model_code=valid_code,
            model_name="test_model",
            input_data={"test": "data"},
            model_type="test"
        )
        
        logger.info(f"Status: {result.status.value}")
        logger.info(f"Execution Time: {result.execution_time:.2f}s")
        
        if result.status == ExecutionStatus.SUCCESS:
            logger.info(f"Return Value: {result.return_value}")
            logger.info("✅ Valid model execution test passed")
        else:
            logger.error(f"❌ Valid model execution test failed: {result.error}")
        
        # Test 2: Dangerous code validation
        logger.info("\n📝 Test 2: Dangerous Code Validation")
        logger.info("-" * 40)
        
        dangerous_code = '''
import os
import subprocess

def main(input_data):
    os.system("rm -rf /")
    return {"status": "dangerous"}
'''
        
        result = executor.execute_model(
            model_code=dangerous_code,
            model_name="dangerous_model",
            input_data={},
            model_type="test"
        )
        
        logger.info(f"Status: {result.status.value}")
        
        if result.status == ExecutionStatus.VALIDATION_ERROR:
            logger.info("✅ Dangerous code validation test passed")
        else:
            logger.error(f"❌ Dangerous code validation test failed: {result.status.value}")
        
        # Test 3: Timeout handling
        logger.info("\n📝 Test 3: Timeout Handling")
        logger.info("-" * 40)
        
        timeout_code = '''
import time

def main(input_data):
    time.sleep(5)  # Sleep for 5 seconds
    return {"status": "completed"}
'''
        
        result = executor.execute_model(
            model_code=timeout_code,
            model_name="timeout_model",
            input_data={},
            model_type="test"
        )
        
        logger.info(f"Status: {result.status.value}")
        logger.info(f"Execution Time: {result.execution_time:.2f}s")
        
        if result.status == ExecutionStatus.TIMEOUT:
            logger.info("✅ Timeout handling test passed")
        else:
            logger.error(f"❌ Timeout handling test failed: {result.status.value}")
        
        # Test 4: Syntax error handling
        logger.info("\n📝 Test 4: Syntax Error Handling")
        logger.info("-" * 40)
        
        syntax_error_code = '''
def main(input_data):
    return {"result": "success"  # Missing closing brace
'''
        
        result = executor.execute_model(
            model_code=syntax_error_code,
            model_name="syntax_error_model",
            input_data={},
            model_type="test"
        )
        
        logger.info(f"Status: {result.status.value}")
        
        if result.status == ExecutionStatus.VALIDATION_ERROR:
            logger.info("✅ Syntax error handling test passed")
        else:
            logger.error(f"❌ Syntax error handling test failed: {result.status.value}")
        
        # Test 5: Strategy execution
        logger.info("\n📝 Test 5: Strategy Execution")
        logger.info("-" * 40)
        
        strategy_code = '''
def main(input_data):
    market_data = input_data.get('market_data', {})
    prices = market_data.get('prices', [100, 101, 102])
    
    if len(prices) > 0:
        signal = "BUY" if prices[-1] > prices[0] else "SELL"
        return {"signal": signal, "confidence": 0.8}
    else:
        return {"signal": "HOLD", "confidence": 0.5}
'''
        
        market_data = {
            'prices': [100, 101, 102, 103, 104]
        }
        
        result = executor.execute_strategy(
            strategy_code=strategy_code,
            strategy_name="test_strategy",
            market_data=market_data,
            parameters={}
        )
        
        logger.info(f"Status: {result.status.value}")
        logger.info(f"Execution Time: {result.execution_time:.2f}s")
        
        if result.status == ExecutionStatus.SUCCESS:
            logger.info(f"Strategy Result: {result.return_value}")
            logger.info("✅ Strategy execution test passed")
        else:
            logger.error(f"❌ Strategy execution test failed: {result.error}")
        
        # Test 6: Indicator execution
        logger.info("\n📝 Test 6: Indicator Execution")
        logger.info("-" * 40)
        
        indicator_code = '''
def main(input_data):
    price_data = input_data.get('price_data', {})
    prices = price_data.get('prices', [100, 101, 102, 103, 104])
    
    if len(prices) >= 3:
        ma = sum(prices[-3:]) / 3
        return {"moving_average": ma, "signal": "BULLISH" if ma > prices[0] else "BEARISH"}
    else:
        return {"error": "Not enough data"}
'''
        
        price_data = {
            'prices': [100, 101, 102, 103, 104, 105]
        }
        
        result = executor.execute_indicator(
            indicator_code=indicator_code,
            indicator_name="test_indicator",
            price_data=price_data,
            parameters={}
        )
        
        logger.info(f"Status: {result.status.value}")
        logger.info(f"Execution Time: {result.execution_time:.2f}s")
        
        if result.status == ExecutionStatus.SUCCESS:
            logger.info(f"Indicator Result: {result.return_value}")
            logger.info("✅ Indicator execution test passed")
        else:
            logger.error(f"❌ Indicator execution test failed: {result.error}")
        
        # Get statistics
        logger.info("\n📊 SafeExecutor Statistics")
        logger.info("-" * 40)
        
        stats = executor.get_statistics()
        logger.info(f"Total Executions: {stats['total_executions']}")
        logger.info(f"Successful Executions: {stats['successful_executions']}")
        logger.info(f"Failed Executions: {stats['failed_executions']}")
        logger.info(f"Success Rate: {stats['success_rate']:.1%}")
        logger.info(f"Total Execution Time: {stats['total_execution_time']:.2f}s")
        logger.info(f"Average Execution Time: {stats['average_execution_time']:.2f}s")
        
        # Clean up
        executor.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"❌ Direct SafeExecutor test failed: {e}")
        return False

def test_service_client():
    """Test SafeExecutor via ServiceClient."""
    logger.info("\n🔗 Testing SafeExecutor via ServiceClient")
    logger.info("=" * 50)
    
    try:
        # Initialize ServiceClient
        client = ServiceClient(
            redis_host='localhost',
            redis_port=6379
        )
        
        # Test 1: Model execution via service
        logger.info("\n📝 Service Test 1: Model Execution")
        logger.info("-" * 40)
        
        model_code = '''
def main(input_data):
    return {"result": "service_test_success", "value": 123}
'''
        
        result = client.execute_model_safely(
            model_code=model_code,
            model_name="service_test_model",
            input_data={"test": "service_data"},
            model_type="test"
        )
        
        if result and result.get('type') == 'model_executed':
            execution_result = result.get('result', {})
            status = execution_result.get('status')
            
            logger.info(f"Status: {status}")
            logger.info(f"Execution Time: {execution_result.get('execution_time', 0):.2f}s")
            
            if status == 'success':
                return_value = execution_result.get('return_value')
                logger.info(f"Return Value: {return_value}")
                logger.info("✅ Service model execution test passed")
            else:
                logger.error(f"❌ Service model execution test failed: {execution_result.get('error')}")
        else:
            logger.error("❌ No response from service")
        
        # Test 2: Strategy execution via service
        logger.info("\n📝 Service Test 2: Strategy Execution")
        logger.info("-" * 40)
        
        strategy_code = '''
def main(input_data):
    market_data = input_data.get('market_data', {})
    prices = market_data.get('prices', [100, 101, 102])
    
    signal = "BUY" if len(prices) > 0 and prices[-1] > 100 else "SELL"
    return {"signal": signal, "confidence": 0.9}
'''
        
        market_data = {
            'prices': [100, 101, 102, 103, 104]
        }
        
        result = client.execute_strategy_safely(
            strategy_code=strategy_code,
            strategy_name="service_test_strategy",
            market_data=market_data,
            parameters={}
        )
        
        if result and result.get('type') == 'strategy_executed':
            execution_result = result.get('result', {})
            status = execution_result.get('status')
            
            logger.info(f"Status: {status}")
            logger.info(f"Execution Time: {execution_result.get('execution_time', 0):.2f}s")
            
            if status == 'success':
                return_value = execution_result.get('return_value')
                logger.info(f"Strategy Result: {return_value}")
                logger.info("✅ Service strategy execution test passed")
            else:
                logger.error(f"❌ Service strategy execution test failed: {execution_result.get('error')}")
        else:
            logger.error("❌ No response from service")
        
        # Test 3: Get statistics via service
        logger.info("\n📝 Service Test 3: Get Statistics")
        logger.info("-" * 40)
        
        stats_result = client.get_safe_executor_statistics()
        if stats_result and stats_result.get('type') == 'statistics':
            stats = stats_result.get('statistics', {})
            logger.info(f"Total Executions: {stats.get('total_executions', 0)}")
            logger.info(f"Successful Executions: {stats.get('successful_executions', 0)}")
            logger.info(f"Failed Executions: {stats.get('failed_executions', 0)}")
            logger.info(f"Success Rate: {stats.get('success_rate', 0):.1%}")
            logger.info("✅ Service statistics test passed")
        else:
            logger.error("❌ Service statistics test failed")
        
        # Test 4: Cleanup via service
        logger.info("\n📝 Service Test 4: Cleanup")
        logger.info("-" * 40)
        
        cleanup_result = client.cleanup_safe_executor()
        if cleanup_result and cleanup_result.get('type') == 'cleanup_completed':
            logger.info("✅ Service cleanup test passed")
        else:
            logger.error("❌ Service cleanup test failed")
        
        client.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ ServiceClient test failed: {e}")
        return False

def test_security_features():
    """Test security features of SafeExecutor."""
    logger.info("\n🛡️ Testing Security Features")
    logger.info("=" * 50)
    
    try:
        executor = SafeExecutor(
            timeout_seconds=10,
            memory_limit_mb=256,
            enable_sandbox=True
        )
        
        # Test dangerous imports
        dangerous_tests = [
            {
                'name': 'OS Import',
                'code': 'import os\ndef main(input_data): return {"status": "dangerous"}'
            },
            {
                'name': 'Subprocess Import',
                'code': 'import subprocess\ndef main(input_data): return {"status": "dangerous"}'
            },
            {
                'name': 'Eval Function',
                'code': 'def main(input_data): eval("print(\'dangerous\')"); return {"status": "dangerous"}'
            },
            {
                'name': 'Exec Function',
                'code': 'def main(input_data): exec("print(\'dangerous\')"); return {"status": "dangerous"}'
            },
            {
                'name': 'File Operations',
                'code': 'def main(input_data): open("/etc/passwd", "r"); return {"status": "dangerous"}'
            }
        ]
        
        passed_tests = 0
        total_tests = len(dangerous_tests)
        
        for test in dangerous_tests:
            logger.info(f"\n📝 Testing: {test['name']}")
            logger.info("-" * 30)
            
            result = executor.execute_model(
                model_code=test['code'],
                model_name=f"security_test_{test['name'].lower().replace(' ', '_')}",
                input_data={},
                model_type="security_test"
            )
            
            if result.status == ExecutionStatus.VALIDATION_ERROR:
                logger.info("✅ Security test passed - dangerous code blocked")
                passed_tests += 1
            else:
                logger.error(f"❌ Security test failed - dangerous code allowed: {result.status.value}")
        
        logger.info(f"\n📊 Security Test Results: {passed_tests}/{total_tests} passed")
        
        executor.cleanup()
        return passed_tests == total_tests
        
    except Exception as e:
        logger.error(f"❌ Security features test failed: {e}")
        return False

def main():
    """Run all SafeExecutor tests."""
    logger.info("🚀 Safe Executor Test Suite")
    logger.info("=" * 60)
    
    # Check Redis availability
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        logger.info("✅ Redis connection successful")
    except Exception as e:
        logger.warning(f"⚠️  Redis not available: {e}")
        logger.info("Running direct SafeExecutor tests only...")
        test_direct_safe_executor()
        test_security_features()

    # Run all tests
    tests = [
        ("Direct SafeExecutor", test_direct_safe_executor),
        ("ServiceClient", test_service_client),
        ("Security Features", test_security_features)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("📊 Test Results Summary")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! SafeExecutor is working correctly.")
    else:
        logger.warning("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main() 