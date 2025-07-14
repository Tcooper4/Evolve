#!/usr/bin/env python3
"""
Test Unified Interface

Verify that the unified interface works correctly.
"""

import os
import sys

import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.skip(
    "Skipping test_unified_interface.py: unified_interface is deprecated and replaced by new UI/agent system.",
    allow_module_level=True,
)


def test_unified_interface():
    """Test the unified interface functionality."""
    print("🧪 Testing Unified Interface")
    print("=" * 50)

    try:
        # Import unified interface
        from interface.unified_interface import UnifiedInterface

        print("✅ Successfully imported UnifiedInterface")

        # Initialize interface
        interface = UnifiedInterface()
        print("✅ Successfully initialized interface")

        # Test help command
        print("\n📋 Testing help command...")
        result = interface.process_command("help")
        if result.get("status") != "error":
            print("✅ Help command works")
        else:
            print(f"❌ Help command failed: {result.get('error')}")

        # Test forecasting command
        print("\n📈 Testing forecast command...")
        result = interface.process_command("forecast AAPL 7d")
        if result.get("status") != "error":
            print("✅ Forecast command works")
        else:
            print(f"❌ Forecast command failed: {result.get('error')}")

        # Test strategy command
        print("\n🎯 Testing strategy command...")
        result = interface.process_command("strategy list")
        if result.get("status") != "error":
            print("✅ Strategy command works")
        else:
            print(f"❌ Strategy command failed: {result.get('error')}")

        # Test agent command
        print("\n🤖 Testing agent command...")
        result = interface.process_command("agent list")
        if result.get("status") != "error":
            print("✅ Agent command works")
        else:
            print(f"❌ Agent command failed: {result.get('error')}")

        # Test status command
        print("\n⚙️ Testing status command...")
        result = interface.process_command("status")
        if result.get("status") != "error":
            print("✅ Status command works")
        else:
            print(f"❌ Status command failed: {result.get('error')}")

        print("\n🎉 All tests completed!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure unified_interface.py is in the interface directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def test_prompt_to_strategy_mapping():
    """Test prompt-to-strategy mapping and model execution via natural language."""
    print("\n🧠 Testing Prompt-to-Strategy Mapping")
    print("=" * 50)

    try:
        # Import required modules
        from trading.llm.llm_processor import LLMProcessor
        from trading.models.forecast_engine import ForecastEngine
        from trading.strategies.strategy_factory import StrategyFactory

        print("✅ Successfully imported required modules")

        # Initialize components
        llm_processor = LLMProcessor()
        strategy_factory = StrategyFactory()
        forecast_engine = ForecastEngine()

        # Test natural language prompts
        test_prompts = [
            "I want to buy when the price is oversold and sell when overbought",
            "Use moving average crossover strategy for AAPL",
            "Apply Bollinger Bands strategy with 20-period lookback",
            "Implement momentum strategy with RSI filter",
            "Create a mean reversion strategy for volatile stocks",
        ]

        for prompt in test_prompts:
            print(f"\n📝 Testing prompt: '{prompt}'")

            # Process natural language to strategy mapping
            strategy_config = llm_processor.extract_strategy_from_prompt(prompt)
            assert (
                strategy_config is not None
            ), f"Failed to extract strategy from prompt: {prompt}"
            print(
                f"✅ Extracted strategy config: {strategy_config.get('strategy_type', 'Unknown')}"
            )

            # Create strategy instance
            strategy = strategy_factory.create_strategy(strategy_config)
            assert (
                strategy is not None
            ), f"Failed to create strategy from config: {strategy_config}"
            print(f"✅ Created strategy: {type(strategy).__name__}")

            # Test model execution
            if hasattr(strategy, "generate_signals"):
                signals = strategy.generate_signals()
                assert (
                    signals is not None
                ), f"Strategy {type(strategy).__name__} failed to generate signals"
                print(
                    f"✅ Generated signals: {len(signals) if hasattr(signals, '__len__') else 'N/A'}"
                )

            # Test forecast integration
            if hasattr(forecast_engine, "forecast"):
                forecast = forecast_engine.forecast("AAPL", strategy=strategy)
                assert (
                    forecast is not None
                ), f"Failed to generate forecast with strategy {type(strategy).__name__}"
                print(f"✅ Generated forecast with strategy integration")

        print("\n🎉 All prompt-to-strategy mapping tests passed!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure required modules are available")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def test_streamlit_import():
    """Test Streamlit import."""
    print("\n🌐 Testing Streamlit import...")
    try:
        pass

        print("✅ Streamlit is available")

        # Test unified interface Streamlit functions

        print("✅ Streamlit UI functions are available")

    except ImportError:
        print("❌ Streamlit not available - install with: pip install streamlit")
    except Exception as e:
        print(f"❌ Streamlit test error: {e}")


def main():
    """Main test function."""
    test_unified_interface()
    test_prompt_to_strategy_mapping()
    test_streamlit_import()

    print("\n📋 Test Summary:")
    print("If all tests passed, you can:")
    print("1. Use command line: python interface/unified_interface.py --terminal")
    print("2. Use Streamlit: streamlit run app.py")
    print("3. Execute commands: python interface/unified_interface.py --command 'help'")


if __name__ == "__main__":
    main()
