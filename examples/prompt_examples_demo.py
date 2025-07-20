"""
Prompt Examples System Demo

This script demonstrates the prompt examples functionality that uses semantic
similarity to find relevant examples for new prompts.
"""

import json
from pathlib import Path

from agents.llm.agent import PromptAgent


def demo_prompt_examples():
    """Demonstrate the prompt examples system."""
    print("ðŸš€ Prompt Examples System Demo")
    print("=" * 50)
    
    # Initialize the prompt agent
    print("1. Initializing Prompt Agent...")
    agent = PromptAgent()
    
    # Show initial statistics
    print("\n2. Initial Prompt Examples Statistics:")
    stats = agent.get_prompt_examples_stats()
    if "error" in stats:
        print(f"   âŒ {stats['error']}")
    else:
        print(f"   ðŸ“Š Total examples: {stats['total_examples']}")
        print(f"   ðŸ“‚ Categories: {list(stats['categories'].keys())}")
        print(f"   ðŸ·ï¸  Unique symbols: {stats['unique_symbols']}")
        print(f"   ðŸ“ˆ Average performance score: {stats['average_performance_score']:.3f}")
        print(f"   ðŸ” Embeddings available: {stats['embeddings_available']}")
        print(f"   ðŸ¤– Sentence transformer available: {stats['sentence_transformer_available']}")
    
    # Test finding similar examples
    print("\n3. Testing Similar Example Finding:")
    test_prompts = [
        "Forecast AAPL stock price for the next 30 days using technical analysis",
        "Create a bullish RSI strategy for TSLA with 14-period lookback",
        "Backtest MACD strategy on SPY for the last 6 months",
        "Analyze volatility patterns in NVDA stock over the past year",
        "Optimize portfolio weights for AAPL, GOOGL, MSFT with risk parity approach"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n   Test {i}: {prompt}")
        
        # Find similar examples
        similar_examples = agent._find_similar_examples(prompt, top_k=2)
        
        if similar_examples:
            print(f"   âœ… Found {len(similar_examples)} similar examples:")
            for j, example_data in enumerate(similar_examples, 1):
                example = example_data['example']
                similarity = example_data['similarity_score']
                print(f"      {j}. {example['prompt']}")
                print(f"         Similarity: {similarity:.3f}")
                print(f"         Category: {example.get('category', 'unknown')}")
                print(f"         Performance: {example.get('performance_score', 0.0):.2f}")
        else:
            print("   âŒ No similar examples found")
    
    # Test few-shot prompt creation
    print("\n4. Testing Few-Shot Prompt Creation:")
    test_prompt = "Forecast GOOGL stock price for the next 60 days"
    similar_examples = agent._find_similar_examples(test_prompt, top_k=2)
    
    if similar_examples:
        enhanced_prompt = agent._create_few_shot_prompt(test_prompt, similar_examples)
        print(f"   Original prompt: {test_prompt}")
        print(f"   Enhanced prompt length: {len(enhanced_prompt)} characters")
        print(f"   Enhanced prompt preview: {enhanced_prompt[:200]}...")
    else:
        print("   âŒ No examples available for few-shot prompt creation")
    
    # Test symbol and timeframe extraction
    print("\n5. Testing Prompt Parsing:")
    test_prompts_parsing = [
        "Forecast AAPL price for next 30 days",
        "Create RSI strategy for TSLA and GOOGL",
        "Backtest MACD on SPY for last 6 months",
        "Analyze Bollinger Bands for MSFT"
    ]
    
    for prompt in test_prompts_parsing:
        symbols = agent._extract_symbols_from_prompt(prompt)
        timeframe = agent._extract_timeframe_from_prompt(prompt)
        strategy_type = agent._extract_strategy_type_from_prompt(prompt)
        
        print(f"   Prompt: {prompt}")
        print(f"     Symbols: {symbols}")
        print(f"     Timeframe: {timeframe}")
        print(f"     Strategy: {strategy_type}")
    
    # Test saving a new example
    print("\n6. Testing Example Saving:")
    try:
        new_parsed_output = {
            "action": "forecast",
            "symbol": "DEMO",
            "timeframe": "7 days",
            "method": "demo_test",
            "confidence": 0.95
        }
        
        agent._save_successful_example(
            "Forecast DEMO stock for next week",
            new_parsed_output,
            "demo_test",
            0.95
        )
        
        # Check updated statistics
        updated_stats = agent.get_prompt_examples_stats()
        print(f"   âœ… Successfully saved new example")
        print(f"   ðŸ“Š Updated total examples: {updated_stats['total_examples']}")
        
    except Exception as e:
        print(f"   âŒ Error saving example: {e}")
    
    print("\n" + "=" * 50)
    print("ðŸŽ‰ Demo completed!")


def demo_similarity_search():
    """Demonstrate similarity search functionality."""
    print("\nðŸ” Similarity Search Demo")
    print("=" * 30)
    
    agent = PromptAgent()
    
    # Test queries with different similarity levels
    test_queries = [
        ("Forecast AAPL price", "Should find forecasting examples"),
        ("Create RSI strategy", "Should find strategy creation examples"),
        ("Backtest MACD", "Should find backtesting examples"),
        ("Analyze volatility", "Should find analysis examples"),
        ("Random unrelated query", "Should find low similarity examples")
    ]
    
    for query, description in test_queries:
        print(f"\nQuery: {query}")
        print(f"Description: {description}")
        
        similar_examples = agent._find_similar_examples(query, top_k=3)
        
        if similar_examples:
            print("Top matches:")
            for i, example_data in enumerate(similar_examples, 1):
                example = example_data['example']
                similarity = example_data['similarity_score']
                print(f"  {i}. Similarity: {similarity:.3f}")
                print(f"     Prompt: {example['prompt']}")
                print(f"     Category: {example.get('category', 'unknown')}")
        else:
            print("  No similar examples found")


def demo_performance_tracking():
    """Demonstrate performance tracking functionality."""
    print("\nðŸ“ˆ Performance Tracking Demo")
    print("=" * 30)
    
    agent = PromptAgent()
    
    # Simulate processing some prompts and tracking performance
    test_cases = [
        {
            "prompt": "Forecast TSLA price for next 30 days",
            "success": True,
            "expected_score": 0.9
        },
        {
            "prompt": "Create MACD strategy for AAPL",
            "success": True,
            "expected_score": 0.85
        },
        {
            "prompt": "Invalid prompt that should fail",
            "success": False,
            "expected_score": 0.0
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['prompt']}")
        
        # Simulate processing
        if test_case['success']:
            # Find similar examples to boost score
            similar_examples = agent._find_similar_examples(test_case['prompt'], top_k=2)
            
            # Calculate performance score
            base_score = test_case['expected_score']
            if similar_examples:
                avg_similarity = sum(ex['similarity_score'] for ex in similar_examples) / len(similar_examples)
                final_score = min(1.0, base_score + avg_similarity * 0.1)
                print(f"  âœ… Success with {len(similar_examples)} similar examples")
                print(f"  ðŸ“Š Base score: {base_score:.2f}")
                print(f"  ðŸ“ˆ Similarity boost: {avg_similarity:.3f}")
                print(f"  ðŸŽ¯ Final score: {final_score:.2f}")
            else:
                print(f"  âœ… Success (no similar examples)")
                print(f"  ðŸ“Š Score: {base_score:.2f}")
        else:
            print(f"  âŒ Failed")
            print(f"  ðŸ“Š Score: 0.0")


def main():
    """Run all demos."""
    try:
        # Run main demo
        demo_prompt_examples()
        
        # Run additional demos
        demo_similarity_search()
        demo_performance_tracking()
        
        print("\n" + "=" * 50)
        print("ðŸŽ¯ All demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("âœ… Loading prompt examples from JSON")
        print("âœ… Semantic similarity search using SentenceTransformers")
        print("âœ… Few-shot prompt creation")
        print("âœ… Symbol and timeframe extraction")
        print("âœ… Performance tracking and scoring")
        print("âœ… Automatic example saving")
        
    except Exception as e:
        print(f"âŒ Error running demos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
