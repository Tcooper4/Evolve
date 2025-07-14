#!/usr/bin/env python3
"""
Test script for the refactored prompt router.

This script demonstrates the key features of the refactored prompt router:
- Memory management
- Automatic weight updates
- Modular agent selection
- Performance tracking
"""

import sys
import time
from datetime import datetime
from agents.prompt_router_refactored import (
    get_prompt_router,
    PromptContext,
    RequestType
)


def test_basic_functionality():
    """Test basic prompt routing functionality."""
    print("🧪 Testing Basic Functionality")
    print("=" * 50)
    
    router = get_prompt_router()
    
    # Test different types of prompts
    test_prompts = [
        "What stocks should I invest in today?",
        "Forecast AAPL for the next 7 days",
        "What's the best RSI strategy for TSLA?",
        "How is the market performing?",
        "Analyze my portfolio performance"
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        result = router.handle_prompt(prompt)
        
        print(f"   ✅ Success: {result.get('success', False)}")
        print(f"   🤖 Agent: {result.get('agent_used', 'Unknown')}")
        print(f"   ⏱️  Response Time: {result.get('response_time', 0):.3f}s")
        print(f"   💬 Message: {result.get('message', 'No message')[:100]}...")
        
        if result.get('model_search_recommended'):
            print(f"   🔍 Model Search Recommended: {result.get('search_reason', 'Unknown')}")


def test_memory_functionality():
    """Test memory management functionality."""
    print("\n🧠 Testing Memory Functionality")
    print("=" * 50)
    
    router = get_prompt_router()
    
    # Test similar prompts to see memory hits
    similar_prompts = [
        "What stocks should I buy today?",
        "Which stocks should I invest in?",
        "What are the best stocks to buy?",
        "Recommend some stocks for investment"
    ]
    
    for i, prompt in enumerate(similar_prompts):
        print(f"\n📝 Similar Prompt {i+1}: {prompt}")
        result = router.handle_prompt(prompt)
        
        # Check if we can access memory information
        if hasattr(router.processor, 'memory_manager'):
            memory_hits = result.get('memory_hits', 0)
            print(f"   🧠 Memory Hits: {memory_hits}")
            
            if memory_hits > 0:
                print(f"   📚 Similar prompts found in memory")


def test_performance_tracking():
    """Test performance tracking and weight updates."""
    print("\n📊 Testing Performance Tracking")
    print("=" * 50)
    
    router = get_prompt_router()
    
    # Generate some traffic to build performance data
    test_prompts = [
        "Forecast MSFT for next week",
        "What's the best strategy for GOOGL?",
        "Should I invest in AMZN?",
        "Analyze TSLA performance",
        "What stocks are trending today?"
    ]
    
    print("Generating test traffic...")
    for prompt in test_prompts:
        router.handle_prompt(prompt)
        time.sleep(0.1)  # Small delay to simulate real usage
    
    # Get performance report
    print("\n📈 Performance Report:")
    report = router.get_performance_report()
    
    for agent_name, stats in report["agents"].items():
        print(f"\n🤖 {agent_name}:")
        print(f"   Success Rate: {stats['success_rate']:.3f}")
        print(f"   Avg Response Time: {stats['avg_response_time']:.3f}s")
        print(f"   User Satisfaction: {stats['user_satisfaction']:.3f}")
        print(f"   Total Requests: {stats['total_requests']}")
        print(f"   Weight: {stats['weight']:.3f}")
    
    print(f"\n📊 Overall Stats:")
    overall = report["overall_stats"]
    print(f"   Total Requests: {overall['total_requests']}")
    print(f"   Avg Success Rate: {overall['avg_success_rate']:.3f}")
    print(f"   Avg Response Time: {overall['avg_response_time']:.3f}s")


def test_context_awareness():
    """Test context awareness and user preferences."""
    print("\n🎯 Testing Context Awareness")
    print("=" * 50)
    
    router = get_prompt_router()
    
    # Create context with user preferences
    context = PromptContext(
        user_id="test_user_123",
        session_id="session_456",
        user_preferences={
            "risk_tolerance": "moderate",
            "preferred_sectors": ["technology", "healthcare"],
            "investment_horizon": "long_term"
        },
        system_state={
            "market_condition": "bull",
            "volatility": "medium"
        }
    )
    
    test_prompt = "What should I invest in?"
    
    print(f"📝 Prompt: {test_prompt}")
    print(f"👤 User Preferences: {context.user_preferences}")
    print(f"🖥️  System State: {context.system_state}")
    
    result = router.handle_prompt(test_prompt, context)
    
    print(f"   ✅ Success: {result.get('success', False)}")
    print(f"   🤖 Agent: {result.get('agent_used', 'Unknown')}")
    print(f"   💬 Message: {result.get('message', 'No message')[:100]}...")


def test_agent_selection():
    """Test intelligent agent selection."""
    print("\n🎯 Testing Agent Selection")
    print("=" * 50)
    
    router = get_prompt_router()
    
    # Test prompts that should route to different agents
    test_cases = [
        ("investment", "What stocks should I buy today?", "InvestmentAgent"),
        ("forecast", "Forecast AAPL for next week", "ForecastAgent"),
        ("strategy", "What's the best RSI strategy?", "StrategyAgent"),
        ("general", "How is the weather?", "GeneralAgent"),
    ]
    
    for category, prompt, expected_agent in test_cases:
        print(f"\n📝 {category.title()} Prompt: {prompt}")
        result = router.handle_prompt(prompt)
        
        actual_agent = result.get('agent_used', 'Unknown')
        print(f"   Expected: {expected_agent}")
        print(f"   Actual: {actual_agent}")
        print(f"   ✅ Correct: {actual_agent == expected_agent}")


def main():
    """Run all tests."""
    print("🚀 Refactored Prompt Router Test Suite")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run all tests
        test_basic_functionality()
        test_memory_functionality()
        test_performance_tracking()
        test_context_awareness()
        test_agent_selection()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 