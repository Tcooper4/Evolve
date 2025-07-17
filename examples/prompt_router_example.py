#!/usr/bin/env python3
"""
Example script demonstrating how to use the new prompt router module.

This script shows how to integrate the prompt router into your application
and handle different types of prompts.
"""

import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing.prompt_router import route_prompt, get_prompt_router


def example_basic_usage():
    """Example of basic prompt routing usage."""
    print("=== Basic Prompt Routing Example ===")
    
    # Simple prompt routing
    result = route_prompt("Forecast SPY using the most accurate model")
    
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    print(f"Navigation: {result['navigation_info']}")
    print(f"Processing time: {result['processing_time']:.3f}s")
    print()


def example_with_llm_type():
    """Example of prompt routing with specific LLM type."""
    print("=== LLM Type Example ===")
    
    # Route with specific LLM type
    result = route_prompt("Create a new LSTM model for AAPL", llm_type="gpt4")
    
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    print(f"Navigation: {result['navigation_info']}")
    print()


def example_error_handling():
    """Example of error handling in prompt routing."""
    print("=== Error Handling Example ===")
    
    # Test with empty prompt
    result = route_prompt("")
    
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    print(f"Error: {result.get('error', False)}")
    print()


def example_different_prompt_types():
    """Example of different types of prompts."""
    print("=== Different Prompt Types Example ===")
    
    prompts = [
        "Show me RSI strategy with Bollinger Bands",
        "What's the current market sentiment for TSLA?",
        "Generate a performance report for my portfolio",
        "Switch to momentum strategy and optimize it",
        "What are the best stocks to buy today?",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Prompt {i}: {prompt} ---")
        
        result = route_prompt(prompt)
        
        print(f"Success: {result['success']}")
        print(f"Message: {result['message'][:100]}...")
        print(f"Navigation: {result['navigation_info']}")
        
        if result.get('strategy_name'):
            print(f"Strategy: {result['strategy_name']}")
        if result.get('model_used'):
            print(f"Model: {result['model_used']}")
        if result.get('confidence'):
            print(f"Confidence: {result['confidence']:.2%}")


def example_direct_router_usage():
    """Example of using the router directly."""
    print("\n=== Direct Router Usage Example ===")
    
    # Get router instance
    router = get_prompt_router()
    
    # Use router directly
    result = router.route_prompt("Analyze the market trends for QQQ")
    
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    print(f"Navigation: {result['navigation_info']}")
    print(f"Processing time: {result['processing_time']:.3f}s")


def main():
    """Run all examples."""
    print("🚀 Prompt Router Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_with_llm_type()
        example_error_handling()
        example_different_prompt_types()
        example_direct_router_usage()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 