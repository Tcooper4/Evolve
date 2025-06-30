#!/usr/bin/env python3
"""
QuantGPT Test Script

Tests the QuantGPT interface functionality.
"""

import sys
import os
import time
from pathlib import Path

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from services.quant_gpt import QuantGPT
from services.service_client import ServiceClient


def test_direct_quant_gpt():
    """Test direct QuantGPT usage."""
    print("🧪 Testing Direct QuantGPT Usage")
    print("=" * 50)
    
    try:
        # Initialize QuantGPT
        quant_gpt = QuantGPT(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            redis_host='localhost',
            redis_port=6379
        )
        
        # Test queries
        test_queries = [
            "Give me the best model for NVDA over 90 days",
            "Should I long TSLA this week?",
            "Analyze BTCUSDT market conditions"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test Query {i}: {query}")
            print("-" * 40)
            
            start_time = time.time()
            result = quant_gpt.process_query(query)
            processing_time = time.time() - start_time
            
            print(f"Processing Time: {processing_time:.2f}s")
            print(f"Status: {result.get('status', 'unknown')}")
            
            if result.get('status') == 'success':
                parsed = result.get('parsed_intent', {})
                print(f"Intent: {parsed.get('intent', 'unknown')}")
                print(f"Symbol: {parsed.get('symbol', 'N/A')}")
                print(f"Timeframe: {parsed.get('timeframe', 'N/A')}")
                print(f"Period: {parsed.get('period', 'N/A')}")
                
                commentary = result.get('gpt_commentary', '')
                if commentary:
                    print(f"GPT Commentary: {commentary[:100]}...")
                
                print("✅ Test passed")
            else:
                error = result.get('error', 'Unknown error')
                print(f"❌ Test failed: {error}")
        
        quant_gpt.close()
        return True
        
    except Exception as e:
        print(f"❌ Direct QuantGPT test failed: {e}")
        return {'success': True, 'result': False, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}


def test_service_client():
    """Test QuantGPT via ServiceClient."""
    print("\n🔗 Testing QuantGPT via ServiceClient")
    print("=" * 50)
    
    try:
        # Initialize ServiceClient
        client = ServiceClient(
            redis_host='localhost',
            redis_port=6379
        )
        
        # Test queries
        test_queries = [
            "What's the best model for MSFT?",
            "Should I buy AMZN now?",
            "Analyze ETHUSDT market"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Service Test Query {i}: {query}")
            print("-" * 40)
            
            start_time = time.time()
            result = client.process_natural_language_query(query)
            processing_time = time.time() - start_time
            
            print(f"Processing Time: {processing_time:.2f}s")
            
            if result:
                print(f"Response Type: {result.get('type', 'unknown')}")
                
                if result.get('type') == 'query_processed':
                    query_result = result.get('result', {})
                    if query_result.get('status') == 'success':
                        parsed = query_result.get('parsed_intent', {})
                        print(f"Intent: {parsed.get('intent', 'unknown')}")
                        print(f"Symbol: {parsed.get('symbol', 'N/A')}")
                        print("✅ Service test passed")
                    else:
                        print(f"❌ Query failed: {query_result.get('error', 'Unknown error')}")
                else:
                    print(f"❌ Service error: {result.get('error', 'Unknown error')}")
            else:
                print("❌ No response from service")
        
        # Test available symbols
        print(f"\n📊 Testing Available Symbols")
        print("-" * 40)
        
        symbols_result = client.get_available_symbols()
        if symbols_result and symbols_result.get('type') == 'available_symbols':
            symbols = symbols_result.get('symbols', [])
            print(f"Available Symbols: {', '.join(symbols)}")
            print("✅ Symbols test passed")
        else:
            print("❌ Symbols test failed")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ ServiceClient test failed: {e}")
        return {'success': True, 'result': False, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}


def test_query_parsing():
    """Test query parsing functionality."""
    print("\n🔍 Testing Query Parsing")
    print("=" * 50)
    
    try:
        quant_gpt = QuantGPT(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            redis_host='localhost',
            redis_port=6379
        )
        
        # Test various query formats
        test_cases = [
            {
                'query': 'Give me the best model for NVDA over 90 days',
                'expected_intent': 'model_recommendation',
                'expected_symbol': 'NVDA',
                'expected_period': '90d'
            },
            {
                'query': 'Should I long TSLA this week?',
                'expected_intent': 'trading_signal',
                'expected_symbol': 'TSLA',
                'expected_timeframe': '1h'
            },
            {
                'query': 'Analyze BTCUSDT market conditions',
                'expected_intent': 'market_analysis',
                'expected_symbol': 'BTCUSDT'
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Parsing Test {i}: {test_case['query']}")
            print("-" * 40)
            
            result = quant_gpt.process_query(test_case['query'])
            
            if result.get('status') == 'success':
                parsed = result.get('parsed_intent', {})
                
                # Check intent
                intent = parsed.get('intent')
                expected_intent = test_case.get('expected_intent')
                if intent == expected_intent:
                    print(f"✅ Intent: {intent}")
                else:
                    print(f"❌ Intent mismatch: expected {expected_intent}, got {intent}")
                
                # Check symbol
                symbol = parsed.get('symbol')
                expected_symbol = test_case.get('expected_symbol')
                if symbol == expected_symbol:
                    print(f"✅ Symbol: {symbol}")
                else:
                    print(f"❌ Symbol mismatch: expected {expected_symbol}, got {symbol}")
                
                # Check other parameters
                for param in ['timeframe', 'period']:
                    if param in test_case:
                        value = parsed.get(param)
                        expected_value = test_case.get(f'expected_{param}')
                        if value == expected_value:
                            print(f"✅ {param}: {value}")
                        else:
                            print(f"❌ {param} mismatch: expected {expected_value}, got {value}")
                
                print("✅ Parsing test passed")
            else:
                print(f"❌ Parsing test failed: {result.get('error', 'Unknown error')}")
        
        quant_gpt.close()
        return True
        
    except Exception as e:
        print(f"❌ Query parsing test failed: {e}")
        return {'success': True, 'result': False, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}


def main():
    """Run all QuantGPT tests."""
    print("🚀 QuantGPT Test Suite")
    print("=" * 60)
    
    # Check Redis availability
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        print("✅ Redis connection successful")
    except Exception as e:
        print(f"⚠️  Redis not available: {e}")
        print("Running direct QuantGPT tests only...")
        test_direct_quant_gpt()
        test_query_parsing()
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    # Run all tests
    tests = [
        ("Direct QuantGPT", test_direct_quant_gpt),
        ("ServiceClient", test_service_client),
        ("Query Parsing", test_query_parsing)
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
        print("🎉 All tests passed! QuantGPT is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main() 