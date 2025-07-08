#!/usr/bin/env python3
"""
Test Script for Commentary Engine Integration

This script tests the integration of the new Commentary Engine with the main system,
including the Commentary Agent and Commentary Service.
"""

import sys
import os
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_commentary_imports():
    """Test that the commentary components can be imported."""
    print("🔧 Testing Commentary Imports...")
    
    try:
        from trading.commentary import CommentaryEngine, CommentaryType, create_commentary_engine
        from trading.agents.commentary_agent import CommentaryAgent, create_commentary_agent
        from trading.services.commentary_service import CommentaryService, create_commentary_service
        print("✅ All commentary components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_commentary_engine():
    """Test the Commentary Engine functionality."""
    print("\n🔧 Testing Commentary Engine...")
    
    try:
        from trading.commentary import create_commentary_engine, CommentaryType, CommentaryRequest
        
        # Create commentary engine
        engine = create_commentary_engine()
        print("✅ Commentary Engine created")
        
        # Test engine statistics
        stats = engine.get_commentary_statistics()
        print(f"✅ Engine statistics: {stats['total_commentaries']} commentaries")
        
        return True
        
    except Exception as e:
        print(f"❌ Commentary Engine error: {e}")
        return False

def test_commentary_agent():
    """Test the Commentary Agent functionality."""
    print("\n🔧 Testing Commentary Agent...")
    
    try:
        from trading.agents.commentary_agent import create_commentary_agent
        
        # Create commentary agent
        agent = create_commentary_agent()
        print("✅ Commentary Agent created")
        
        # Test agent statistics
        stats = agent.get_commentary_statistics()
        print(f"✅ Agent statistics: {stats['total_commentaries']} commentaries")
        
        return True
        
    except Exception as e:
        print(f"❌ Commentary Agent error: {e}")
        return False

def test_commentary_service():
    """Test the Commentary Service functionality."""
    print("\n🔧 Testing Commentary Service...")
    
    try:
        from trading.services.commentary_service import create_commentary_service
        
        # Create commentary service
        service = create_commentary_service()
        print("✅ Commentary Service created")
        
        # Test service statistics
        stats = service.get_service_statistics()
        print(f"✅ Service statistics: {stats['service_stats']['total_requests']} requests")
        
        return True
        
    except Exception as e:
        print(f"❌ Commentary Service error: {e}")
        return False

async def test_async_commentary():
    """Test async commentary generation."""
    print("\n🔧 Testing Async Commentary Generation...")
    
    try:
        from trading.commentary import create_commentary_engine, CommentaryType, CommentaryRequest
        
        # Create commentary engine
        engine = create_commentary_engine()
        
        # Create sample trade data
        trade_data = {
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 100,
            'price': 150.0,
            'signal_strength': 0.8,
            'model_confidence': 0.85,
            'timestamp': datetime.now()
        }
        
        # Create sample market data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        market_data = pd.DataFrame({
            'close': 150 + np.cumsum(np.random.normal(0, 1, 100)),
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # Create commentary request
        request = CommentaryRequest(
            commentary_type=CommentaryType.TRADE_EXPLANATION,
            symbol='AAPL',
            timestamp=datetime.now(),
            trade_data=trade_data,
            market_data=market_data
        )
        
        # Generate commentary
        response = await engine.generate_commentary(request)
        
        print(f"✅ Generated commentary: {response.title}")
        print(f"✅ Confidence: {response.confidence_score:.2f}")
        print(f"✅ Insights: {len(response.key_insights)}")
        print(f"✅ Recommendations: {len(response.recommendations)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Async commentary error: {e}")
        return False

def test_agent_integration():
    """Test integration with the agent system."""
    print("\n🔧 Testing Agent Integration...")
    
    try:
        from trading.agents import CommentaryAgent, create_commentary_agent
        from trading.agents.base_agent_interface import BaseAgent
        
        # Test inheritance
        assert issubclass(CommentaryAgent, BaseAgent)
        print("✅ CommentaryAgent inherits from BaseAgent")
        
        # Create agent
        agent = create_commentary_agent()
        
        # Test required methods exist
        assert hasattr(agent, 'generate_commentary'), "CommentaryAgent missing generate_commentary"
        assert hasattr(agent, 'explain_trade'), "CommentaryAgent missing explain_trade"
        assert hasattr(agent, 'analyze_performance'), "CommentaryAgent missing analyze_performance"
        assert hasattr(agent, 'assess_risk'), "CommentaryAgent missing assess_risk"
        
        print("✅ CommentaryAgent has all required methods")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent integration error: {e}")
        return False

def test_service_integration():
    """Test integration with the service system."""
    print("\n🔧 Testing Service Integration...")
    
    try:
        from trading.services.commentary_service import CommentaryService, create_commentary_service
        
        # Create service
        service = create_commentary_service()
        
        # Test service methods exist
        assert hasattr(service, 'explain_trade'), "CommentaryService missing explain_trade"
        assert hasattr(service, 'analyze_performance'), "CommentaryService missing analyze_performance"
        assert hasattr(service, 'assess_risk'), "CommentaryService missing assess_risk"
        assert hasattr(service, 'generate_daily_summary'), "CommentaryService missing generate_daily_summary"
        assert hasattr(service, 'get_service_statistics'), "CommentaryService missing get_service_statistics"
        
        print("✅ CommentaryService has all required methods")
        
        return True
        
    except Exception as e:
        print(f"❌ Service integration error: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting Commentary Engine Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Commentary Imports", test_commentary_imports),
        ("Commentary Engine", test_commentary_engine),
        ("Commentary Agent", test_commentary_agent),
        ("Commentary Service", test_commentary_service),
        ("Agent Integration", test_agent_integration),
        ("Service Integration", test_service_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    # Test async functionality
    print(f"\n📋 Running: Async Commentary Generation")
    try:
        if await test_async_commentary():
            passed += 1
            print(f"✅ Async Commentary Generation: PASSED")
        else:
            print(f"❌ Async Commentary Generation: FAILED")
    except Exception as e:
        print(f"❌ Async Commentary Generation: ERROR - {e}")
    
    total += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Commentary Engine integration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 