#!/usr/bin/env python3
"""
Test Unified Interface

Verify that the unified interface works correctly.
"""

import sys
import os
import pytest
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.skip("Skipping test_unified_interface.py: unified_interface is deprecated and replaced by new UI/agent system.", allow_module_level=True)

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
        if result.get('status') != 'error':
            print("✅ Help command works")
        else:
            print(f"❌ Help command failed: {result.get('error')}")
        
        # Test forecasting command
        print("\n📈 Testing forecast command...")
        result = interface.process_command("forecast AAPL 7d")
        if result.get('status') != 'error':
            print("✅ Forecast command works")
        else:
            print(f"❌ Forecast command failed: {result.get('error')}")
        
        # Test strategy command
        print("\n🎯 Testing strategy command...")
        result = interface.process_command("strategy list")
        if result.get('status') != 'error':
            print("✅ Strategy command works")
        else:
            print(f"❌ Strategy command failed: {result.get('error')}")
        
        # Test agent command
        print("\n🤖 Testing agent command...")
        result = interface.process_command("agent list")
        if result.get('status') != 'error':
            print("✅ Agent command works")
        else:
            print(f"❌ Agent command failed: {result.get('error')}")
        
        # Test status command
        print("\n⚙️ Testing status command...")
        result = interface.process_command("status")
        if result.get('status') != 'error':
            print("✅ Status command works")
        else:
            print(f"❌ Status command failed: {result.get('error')}")
        
        print("\n🎉 All tests completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure unified_interface.py is in the interface directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_streamlit_import():
    """Test Streamlit import."""
    print("\n🌐 Testing Streamlit import...")
    try:
        import streamlit as st
        print("✅ Streamlit is available")
        
        # Test unified interface Streamlit functions
        from interface.unified_interface import streamlit_ui, render_main_interface
        print("✅ Streamlit UI functions are available")
        
    except ImportError:
        print("❌ Streamlit not available - install with: pip install streamlit")
    except Exception as e:
        print(f"❌ Streamlit test error: {e}")

def main():
    """Main test function."""
    test_unified_interface()
    test_streamlit_import()
    
    print("\n📋 Test Summary:")
    print("If all tests passed, you can:")
    print("1. Use command line: python interface/unified_interface.py --terminal")
    print("2. Use Streamlit: streamlit run app.py")
    print("3. Execute commands: python interface/unified_interface.py --command 'help'")

if __name__ == "__main__":
    main() 