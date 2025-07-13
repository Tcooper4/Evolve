#!/usr/bin/env python3
"""
Simple Unified Interface Launcher

Quick access to all Evolve features through unified interface.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))


def main():
    """Main launcher function."""
    print("🔮 Evolve Unified Interface Launcher")
    print("=" * 50)

    if len(sys.argv) > 1:
        # Command line mode
        command = " ".join(sys.argv[1:])
        print(f"Executing: {command}")

        try:
            from unified_interface import UnifiedInterface

            interface = UnifiedInterface()
            result = interface.process_command(command)

            if result.get("status") == "error":
                print(f"❌ Error: {result.get('error')}")
            else:
                print("✅ Success!")
                print(f"Result: {result}")

        except ImportError as e:
            print(f"❌ Error: {e}")
            print("Make sure unified_interface.py is in the current directory")
    else:
        # Interactive mode
        print("Usage:")
        print("  python run_unified.py <command>")
        print("")
        print("Examples:")
        print("  python run_unified.py help")
        print("  python run_unified.py forecast AAPL 7d")
        print('  python run_unified.py "What\'s the best model for TSLA?"')
        print("")
        print("Or launch Streamlit interface:")
        print("  streamlit run app.py")
        print("  streamlit run unified_interface.py")


if __name__ == "__main__":
    main()
