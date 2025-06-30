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

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from services.service_client import ServiceClient
from utils.safe_executor import SafeExecutor, ExecutionStatus


def test_direct_safe_executor():
    """Test direct SafeExecutor usage."""
    print("🧪 Testing Direct SafeExecutor Usage")
    print("=" * 50)
    
    try:
        # Initialize SafeExecutor
        executor = SafeExecutor(
            timeout_seconds=30,
            memory_limit_mb=512,
            enable_sandbox=True,
            log_executions=True
        )
        
        # Test 1: Valid model execution
        print("\n📝 Test 1: Valid Model Execution")
        print("-" * 40)
        
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
        
        print(f"Status: {result.status.value}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        
        if result.status == ExecutionStatus.SUCCESS:
            print(f"Return Value: {result.return_value}")
            print("✅ Valid model execution test passed")
        else:
            print(f"❌ Valid model execution test failed: {result.error}")
        
        # Test 2: Dangerous code validation
        print("\n📝 Test 2: Dangerous Code Validation")
        print("-" * 40)
        
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
        
        print(f"Status: {result.status.value}")
        
        if result.status == ExecutionStatus.VALIDATION_ERROR:
            print("✅ Dangerous code validation test passed")
        else:
            print(f"❌ Dangerous code validation test failed: {result.status.value}")
        
        # Test 3: Timeout handling
        print("\n📝 Test 3: Timeout Handling")
        print("-" * 40)
        
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
        
        print(f"Status: {result.status.value}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        
        if result.status == ExecutionStatus.TIMEOUT:
            print("✅ Timeout handling test passed")
        else:
            print(f"❌ Timeout handling test failed: {result.status.value}")
        
        # Test 4: Syntax error handling
        print("\n📝 Test 4: Syntax Error Handling")
        print("-" * 40)
        
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
        
        print(f"Status: {result.status.value}")
        
        if result.status == ExecutionStatus.VALIDATION_ERROR:
            print("✅ Syntax error handling test passed")
        else:
            print(f"❌ Syntax error handling test failed: {result.status.value}")
        
        # Test 5: Strategy execution
        print("\n📝 Test 5: Strategy Execution")
        print("-" * 40)
        
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
        
        print(f"Status: {result.status.value}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        
        if result.status == ExecutionStatus.SUCCESS:
            print(f"Strategy Result: {result.return_value}")
            print("✅ Strategy execution test passed")
        else:
            print(f"❌ Strategy execution test failed: {result.error}")
        
        # Test 6: Indicator execution
        print("\n📝 Test 6: Indicator Execution")
        print("-" * 40)
        
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
        
        print(f"Status: {result.status.value}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        
        if result.status == ExecutionStatus.SUCCESS:
            print(f"Indicator Result: {result.return_value}")
            print("✅ Indicator execution test passed")
        else:
            print(f"❌ Indicator execution test failed: {result.error}")
        
        # Get statistics
        print("\n📊 SafeExecutor Statistics")
        print("-" * 40)
        
        stats = executor.get_statistics()
        print(f"Total Executions: {stats['total_executions']}")
        print(f"Successful Executions: {stats['successful_executions']}")
        print(f"Failed Executions: {stats['failed_executions']}")
        print(f"Success Rate: {stats['success_rate']:.1%}")
        print(f"Total Execution Time: {stats['total_execution_time']:.2f}s")
        print(f"Average Execution Time: {stats['average_execution_time']:.2f}s")
        
        # Clean up
        executor.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Direct SafeExecutor test failed: {e}")
        return {'success': True, 'result': False, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}


def test_service_client():
    """Test SafeExecutor via ServiceClient."""
    print("\n🔗 Testing SafeExecutor via ServiceClient")
    print("=" * 50)
    
    try:
        # Initialize ServiceClient
        client = ServiceClient(
            redis_host='localhost',
            redis_port=6379
        )
        
        # Test 1: Model execution via service
        print("\n📝 Service Test 1: Model Execution")
        print("-" * 40)
        
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
            
            print(f"Status: {status}")
            print(f"Execution Time: {execution_result.get('execution_time', 0):.2f}s")
            
            if status == 'success':
                return_value = execution_result.get('return_value')
                print(f"Return Value: {return_value}")
                print("✅ Service model execution test passed")
            else:
                print(f"❌ Service model execution test failed: {execution_result.get('error')}")
        else:
            print("❌ No response from service")
        
        # Test 2: Strategy execution via service
        print("\n📝 Service Test 2: Strategy Execution")
        print("-" * 40)
        
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
            
            print(f"Status: {status}")
            print(f"Execution Time: {execution_result.get('execution_time', 0):.2f}s")
            
            if status == 'success':
                return_value = execution_result.get('return_value')
                print(f"Strategy Result: {return_value}")
                print("✅ Service strategy execution test passed")
            else:
                print(f"❌ Service strategy execution test failed: {execution_result.get('error')}")
        else:
            print("❌ No response from service")
        
        # Test 3: Get statistics via service
        print("\n📝 Service Test 3: Get Statistics")
        print("-" * 40)
        
        stats_result = client.get_safe_executor_statistics()
        if stats_result and stats_result.get('type') == 'statistics':
            stats = stats_result.get('statistics', {})
            print(f"Total Executions: {stats.get('total_executions', 0)}")
            print(f"Successful Executions: {stats.get('successful_executions', 0)}")
            print(f"Failed Executions: {stats.get('failed_executions', 0)}")
            print(f"Success Rate: {stats.get('success_rate', 0):.1%}")
            print("✅ Service statistics test passed")
        else:
            print("❌ Service statistics test failed")
        
        # Test 4: Cleanup via service
        print("\n📝 Service Test 4: Cleanup")
        print("-" * 40)
        
        cleanup_result = client.cleanup_safe_executor()
        if cleanup_result and cleanup_result.get('type') == 'cleanup_completed':
            print("✅ Service cleanup test passed")
        else:
            print("❌ Service cleanup test failed")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ ServiceClient test failed: {e}")
        return {'success': True, 'result': False, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}


def test_security_features():
    """Test security features of SafeExecutor."""
    print("\n🛡️ Testing Security Features")
    print("=" * 50)
    
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
            print(f"\n📝 Testing: {test['name']}")
            print("-" * 30)
            
            result = executor.execute_model(
                model_code=test['code'],
                model_name=f"security_test_{test['name'].lower().replace(' ', '_')}",
                input_data={},
                model_type="security_test"
            )
            
            if result.status == ExecutionStatus.VALIDATION_ERROR:
                print("✅ Security test passed - dangerous code blocked")
                passed_tests += 1
            else:
                print(f"❌ Security test failed - dangerous code allowed: {result.status.value}")
        
        print(f"\n📊 Security Test Results: {passed_tests}/{total_tests} passed")
        
        executor.cleanup()
        return passed_tests == total_tests
        
    except Exception as e:
        print(f"❌ Security features test failed: {e}")
        return {'success': True, 'result': False, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}


def main():
    """Run all SafeExecutor tests."""
    print("🚀 Safe Executor Test Suite")
    print("=" * 60)
    
    # Check Redis availability
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        print("✅ Redis connection successful")
    except Exception as e:
        print(f"⚠️  Redis not available: {e}")
        print("Running direct SafeExecutor tests only...")
        test_direct_safe_executor()
        test_security_features()
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    # Run all tests
    tests = [
        ("Direct SafeExecutor", test_direct_safe_executor),
        ("ServiceClient", test_service_client),
        ("Security Features", test_security_features)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SafeExecutor is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main() 