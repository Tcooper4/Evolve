#!/usr/bin/env python3
"""
Simple test script for the report generation system.
"""

import sys
from pathlib import Path

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all imports work."""
    try:
        from report.report_generator import ReportGenerator, generate_quick_report
        from report.report_client import ReportClient
        from report.report_service import ReportService
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_report():
    """Test basic report generation."""
    try:
        from report.report_generator import generate_quick_report
        
        # Sample data
        trade_data = {
            'trades': [
                {'pnl': 100, 'duration': 3600},
                {'pnl': -50, 'duration': 1800},
                {'pnl': 200, 'duration': 7200}
            ]
        }
        
        model_data = {
            'predictions': [100, 102, 98, 105, 103],
            'actuals': [100, 101, 99, 104, 102]
        }
        
        strategy_data = {
            'strategy_name': 'Test Strategy',
            'symbol': 'AAPL',
            'timeframe': '1h',
            'signals': ['BUY', 'SELL', 'BUY'],
            'market_conditions': {'trend': 'bullish'},
            'performance': {'total_return': 0.15},
            'parameters': {'param1': 10}
        }
        
        # Generate report
        report_data = generate_quick_report(
            trade_data=trade_data,
            model_data=model_data,
            strategy_data=strategy_data,
            symbol='AAPL',
            timeframe='1h',
            period='7d'
        )
        
        print(f"✅ Report generated successfully: {report_data['report_id']}")
        print(f"📊 Trade metrics: {report_data['trade_metrics'].total_trades} trades")
        print(f"🤖 Model metrics: MSE = {report_data['model_metrics'].mse:.4f}")
        print(f"📁 Files created: {list(report_data['files'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Report generation error: {e}")
        return False

def test_client():
    """Test report client."""
    try:
        from report.report_client import ReportClient
        
        client = ReportClient()
        status = client.get_service_status()
        
        print(f"✅ Client initialized successfully")
        print(f"🔍 Service status: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Client error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Report Generation System")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Report Test", test_basic_report),
        ("Client Test", test_client)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Report system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 