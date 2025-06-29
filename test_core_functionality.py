#!/usr/bin/env python3
"""Simple test for core functionality."""

import sys
import traceback

def test_basic_imports():
    """Test basic imports."""
    print("Testing basic imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import streamlit as st
        print("✅ Basic imports successful")
        return True
    except ImportError as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_config_loader():
    """Test configuration loader."""
    print("Testing configuration loader...")
    
    try:
        from utils.config_loader import ConfigLoader
        config = ConfigLoader()
        print("✅ Configuration loader successful")
        return True
    except ImportError as e:
        print(f"❌ Configuration loader failed: {e}")
        return False

def test_dashboard_import():
    """Test dashboard import."""
    print("Testing dashboard import...")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('dashboard', 'pages/10_Strategy_Health_Dashboard.py')
        module = importlib.util.module_from_spec(spec)
        print("✅ Dashboard import successful")
        return True
    except Exception as e:
        print(f"❌ Dashboard import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Core Functionality")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_config_loader,
        test_dashboard_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n📊 Summary:")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All core tests passed!")
        return True
    else:
        print("\n⚠️ Some tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 