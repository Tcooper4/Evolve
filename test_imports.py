#!/usr/bin/env python3
"""
Test script to check all imports and identify any remaining issues.
"""

import sys
import traceback

def test_import(module_name):
    """Test importing a module and return success status."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} imported successfully")
        return True
    except Exception as e:
        print(f"❌ {module_name} failed to import: {e}")
        traceback.print_exc()
        return None

def main():
    """Test all critical imports."""
    print("Testing critical imports...")
    print("=" * 50)
    
    modules_to_test = [
        "trading.memory.model_monitor",
        "llm.llm_summary", 
        "trading.portfolio.portfolio_manager",
        "trading.risk.risk_analyzer",
        "agents.strategy_switcher",
        "trading.utils",
        "fpdf"
    ]
    
    success_count = 0
    total_count = len(modules_to_test)
    
    for module in modules_to_test:
        if test_import(module):
            success_count += 1
        print()
    
    print("=" * 50)
    print(f"Results: {success_count}/{total_count} modules imported successfully")
    
    if success_count == total_count:
        print("🎉 All imports successful! The app should work now.")
    else:
        print("⚠️  Some imports failed. Check the errors above.")

    return None
if __name__ == "__main__":
    main() 