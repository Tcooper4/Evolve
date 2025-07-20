"""
StrategyResearchAgent Example

This example demonstrates how to use the StrategyResearchAgent to:
1. Scan multiple sources for new trading strategies
2. Extract and analyze discovered strategies
3. Generate executable strategy code
4. Test strategies with the backtester
5. Schedule periodic scans
"""

import asyncio
import time
import json
from datetime import datetime
from pathlib import Path

from agents.strategy_research_agent import StrategyResearchAgent
from utils.common_helpers import safe_json_save


def main():
    """Main example function"""
    print("ðŸš€ StrategyResearchAgent Example")
    print("=" * 50)
    
    # Initialize the agent
    print("Initializing StrategyResearchAgent...")
    agent = StrategyResearchAgent()
    
    # Example 1: Run a single research scan
    print("\nðŸ“Š Running research scan...")
    results = agent.run()
    
    print(f"Scan Results:")
    print(f"  - Discoveries found: {results['discoveries_found']}")
    print(f"  - Strategies tested: {results['strategies_tested']}")
    print(f"  - Status: {results['status']}")
    
    if results['status'] == 'success':
        print("\nðŸ“ˆ Discovery Summary:")
        summary = results['summary']
        print(f"  - Total discoveries: {summary['total_discoveries']}")
        print(f"  - By source: {summary['by_source']}")
        print(f"  - By type: {summary['by_type']}")
        print(f"  - Confidence distribution: {summary['confidence_distribution']}")
        
        # Show recent discoveries
        if summary['recent_discoveries']:
            print("\nðŸ” Recent Discoveries:")
            for discovery in summary['recent_discoveries'][:5]:
                print(f"  - {discovery['title']} ({discovery['source']})")
                print(f"    Type: {discovery['type']}, Confidence: {discovery['confidence']:.2f}")
    
    # Example 2: Manual search of specific sources
    print("\nðŸ” Manual source search...")
    
    # Search arXiv
    print("Searching arXiv...")
    arxiv_discoveries = agent.search_arxiv("trading strategy", max_results=5)
    print(f"Found {len(arxiv_discoveries)} strategies from arXiv")
    
    for discovery in arxiv_discoveries[:3]:
        print(f"  - {discovery.title}")
        print(f"    Type: {discovery.strategy_type}, Confidence: {discovery.confidence_score:.2f}")
        print(f"    Authors: {', '.join(discovery.authors[:2])}")
    
    # Search GitHub
    print("\nSearching GitHub...")
    github_discoveries = agent.search_github("trading strategy", max_results=5)
    print(f"Found {len(github_discoveries)} strategies from GitHub")
    
    for discovery in github_discoveries[:3]:
        print(f"  - {discovery.title}")
        print(f"    Type: {discovery.strategy_type}, Confidence: {discovery.confidence_score:.2f}")
        print(f"    Stars: {discovery.url}")
    
    # Example 3: Generate and test a specific strategy
    if arxiv_discoveries:
        print("\nðŸ§ª Testing discovered strategy...")
        test_discovery = arxiv_discoveries[0]
        
        print(f"Testing: {test_discovery.title}")
        print(f"Source: {test_discovery.source}")
        print(f"Type: {test_discovery.strategy_type}")
        
        # Generate strategy code
        print("Generating strategy code...")
        strategy_code = agent.generate_strategy_code(test_discovery)
        
        if strategy_code:
            print("âœ… Strategy code generated successfully")
            
            # Save strategy code to file
            strategy_filename = f"discovered_{test_discovery.title.replace(' ', '_')[:30]}.py"
            strategy_path = Path("strategies/discovered") / test_discovery.source / strategy_filename
            strategy_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(strategy_path, 'w') as f:
                f.write(strategy_code)
            
            print(f"Strategy code saved to: {strategy_path}")
            
            # Test the strategy
            print("Running backtest...")
            test_results = agent.test_discovered_strategy(test_discovery)
            
            if 'error' not in test_results:
                print("âœ… Backtest completed successfully")
                print(f"Results: {json.dumps(test_results, indent=2)}")
            else:
                print(f"âŒ Backtest failed: {test_results['error']}")
        else:
            print("âŒ Failed to generate strategy code")
    
    # Example 4: Strategy analysis and filtering
    print("\nðŸ“Š Strategy Analysis...")
    
    # Filter by confidence
    high_confidence = [d for d in agent.discovered_strategies if d.confidence_score > 0.7]
    medium_confidence = [d for d in agent.discovered_strategies if 0.4 <= d.confidence_score <= 0.7]
    low_confidence = [d for d in agent.discovered_strategies if d.confidence_score < 0.4]
    
    print(f"High confidence strategies: {len(high_confidence)}")
    print(f"Medium confidence strategies: {len(medium_confidence)}")
    print(f"Low confidence strategies: {len(low_confidence)}")
    
    # Filter by strategy type
    momentum_strategies = [d for d in agent.discovered_strategies if d.strategy_type == "momentum"]
    mean_reversion_strategies = [d for d in agent.discovered_strategies if d.strategy_type == "mean_reversion"]
    ml_strategies = [d for d in agent.discovered_strategies if d.strategy_type == "ml"]
    
    print(f"Momentum strategies: {len(momentum_strategies)}")
    print(f"Mean reversion strategies: {len(mean_reversion_strategies)}")
    print(f"ML strategies: {len(ml_strategies)}")
    
    # Example 5: Save discoveries to different formats
    print("\nðŸ’¾ Saving discoveries...")
    
    # Save as JSON
    discoveries_data = []
    for discovery in agent.discovered_strategies:
        discovery_dict = {
            'title': discovery.title,
            'source': discovery.source,
            'strategy_type': discovery.strategy_type,
            'confidence_score': discovery.confidence_score,
            'authors': discovery.authors,
            'url': discovery.url,
            'discovered_date': discovery.discovered_date,
            'parameters': discovery.parameters,
            'tags': discovery.tags
        }
        discoveries_data.append(discovery_dict)
    
    safe_json_save("examples/discovered_strategies.json", discoveries_data)
    print("Discoveries saved to: examples/discovered_strategies.json")
    
    # Save summary report
    summary_report = {
        'scan_date': datetime.now().isoformat(),
        'total_discoveries': len(agent.discovered_strategies),
        'summary': agent.get_discovery_summary(),
        'test_results': agent.test_results
    }
    
    safe_json_save("examples/strategy_research_report.json", summary_report)
    print("Summary report saved to: examples/strategy_research_report.json")
    
    # Example 6: Schedule periodic scans
    print("\nâ° Scheduling periodic scans...")
    
    # Schedule scans every 12 hours
    agent.schedule_periodic_scans(interval_hours=12)
    print("Periodic scans scheduled every 12 hours")
    print("Agent will continue running in background...")
    
    # Example 7: Interactive strategy exploration
    print("\nðŸ” Interactive Strategy Exploration")
    print("Available commands:")
    print("  - 'list': List all discovered strategies")
    print("  - 'test <index>': Test strategy by index")
    print("  - 'details <index>': Show strategy details")
    print("  - 'scan': Run new scan")
    print("  - 'quit': Exit")
    
    while True:
        try:
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'list':
                print("\nDiscovered Strategies:")
                for i, discovery in enumerate(agent.discovered_strategies):
                    print(f"  {i}: {discovery.title} ({discovery.source}) - {discovery.strategy_type}")
            
            elif command.startswith('test '):
                try:
                    index = int(command.split()[1])
                    if 0 <= index < len(agent.discovered_strategies):
                        discovery = agent.discovered_strategies[index]
                        print(f"Testing: {discovery.title}")
                        results = agent.test_discovered_strategy(discovery)
                        print(f"Results: {json.dumps(results, indent=2)}")
                    else:
                        print("Invalid index")
                except (ValueError, IndexError):
                    print("Invalid index")
            
            elif command.startswith('details '):
                try:
                    index = int(command.split()[1])
                    if 0 <= index < len(agent.discovered_strategies):
                        discovery = agent.discovered_strategies[index]
                        print(f"\nStrategy Details:")
                        print(f"  Title: {discovery.title}")
                        print(f"  Source: {discovery.source}")
                        print(f"  Type: {discovery.strategy_type}")
                        print(f"  Confidence: {discovery.confidence_score:.2f}")
                        print(f"  Authors: {', '.join(discovery.authors)}")
                        print(f"  URL: {discovery.url}")
                        print(f"  Parameters: {discovery.parameters}")
                        print(f"  Tags: {', '.join(discovery.tags)}")
                    else:
                        print("Invalid index")
                except (ValueError, IndexError):
                    print("Invalid index")
            
            elif command == 'scan':
                print("Running new scan...")
                results = agent.run()
                print(f"Found {results['discoveries_found']} new strategies")
            
            else:
                print("Unknown command")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
    
    print("\nâœ… StrategyResearchAgent example completed!")


def example_arxiv_search():
    """Example of searching arXiv specifically"""
    print("\nðŸ“š arXiv Search Example")
    
    agent = StrategyResearchAgent()
    
    # Search for specific strategy types
    queries = [
        "momentum trading strategy",
        "mean reversion trading",
        "machine learning trading",
        "options trading strategy"
    ]
    
    for query in queries:
        print(f"\nSearching for: {query}")
        discoveries = agent.search_arxiv(query, max_results=3)
        
        print(f"Found {len(discoveries)} strategies:")
        for discovery in discoveries:
            print(f"  - {discovery.title}")
            print(f"    Confidence: {discovery.confidence_score:.2f}")
            print(f"    Type: {discovery.strategy_type}")


def example_github_search():
    """Example of searching GitHub specifically"""
    print("\nðŸ™ GitHub Search Example")
    
    agent = StrategyResearchAgent()
    
    # Search for trading repositories
    queries = [
        "trading strategy python",
        "quantitative trading",
        "algorithmic trading",
        "backtesting framework"
    ]
    
    for query in queries:
        print(f"\nSearching for: {query}")
        discoveries = agent.search_github(query, max_results=3)
        
        print(f"Found {len(discoveries)} repositories:")
        for discovery in discoveries:
            print(f"  - {discovery.title}")
            print(f"    Confidence: {discovery.confidence_score:.2f}")
            print(f"    Type: {discovery.strategy_type}")


def example_strategy_generation():
    """Example of generating strategy code"""
    print("\nâš™ï¸ Strategy Code Generation Example")
    
    agent = StrategyResearchAgent()
    
    # Create a sample discovery
    sample_discovery = agent.StrategyDiscovery(
        source="example",
        title="Sample Momentum Strategy",
        description="A sample momentum trading strategy for demonstration",
        authors=["Example Author"],
        url="https://example.com",
        discovered_date=datetime.now().isoformat(),
        strategy_type="momentum",
        confidence_score=0.8,
        code_snippets=["def calculate_momentum(prices): return prices.pct_change()"],
        parameters={"lookback": 20, "threshold": 0.5, "window": 14},
        requirements=["pandas", "numpy"],
        tags=["python", "momentum", "trading"]
    )
    
    # Generate strategy code
    print("Generating strategy code...")
    strategy_code = agent.generate_strategy_code(sample_discovery)
    
    if strategy_code:
        print("âœ… Strategy code generated:")
        print("-" * 40)
        print(strategy_code[:500] + "..." if len(strategy_code) > 500 else strategy_code)
        print("-" * 40)
        
        # Save to file
        with open("examples/generated_strategy.py", 'w') as f:
            f.write(strategy_code)
        print("Strategy code saved to: examples/generated_strategy.py")
    else:
        print("âŒ Failed to generate strategy code")


if __name__ == "__main__":
    # Run main example
    main()
    
    # Run specific examples
    example_arxiv_search()
    example_github_search()
    example_strategy_generation()
    
    print("\nðŸŽ‰ All examples completed!")
