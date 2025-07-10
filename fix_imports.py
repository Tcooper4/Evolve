#!/usr/bin/env python3
"""
Fix import issues in trading/__init__.py and validate the fixes
"""

import re
import importlib
import sys
import os

def test_imports():
    """Test that the trading module can be imported successfully."""
    try:
        # Add the current directory to Python path if not already there
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        
        # Test importing the trading module
        import trading
        print("✅ Successfully imported trading module")
        
        # Test specific imports that were fixed
        test_imports = [
            ('trading.optimization', ['StrategyOptimizer', 'BaseOptimizer', 'OptimizationVisualizer']),
            ('trading.risk', ['RiskManager']),
            ('trading.portfolio', ['PortfolioManager']),
            ('trading.agents', ['get_prompt_router_agent', 'get_model_builder_agent']),
            ('trading.utils', ['LoggingManager', 'DataValidator', 'ConfigManager'])
        ]
        
        all_imports_working = True
        for module_name, expected_classes in test_imports:
            try:
                module = importlib.import_module(module_name)
                for class_name in expected_classes:
                    if hasattr(module, class_name):
                        print(f"✅ {module_name}.{class_name} imported successfully")
                    else:
                        print(f"❌ {module_name}.{class_name} not found")
                        all_imports_working = False
            except ImportError as e:
                print(f"❌ Failed to import {module_name}: {e}")
                all_imports_working = False
        
        return all_imports_working
        
    except ImportError as e:
        print(f"❌ Failed to import trading module: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import test: {e}")
        return False

def fix_trading_init():
    """Fix the import statements in trading/__init__.py and validate the fixes"""
    
    print("🔧 Fixing import statements in trading/__init__.py...")
    
    # Read the current file
    try:
        with open('trading/__init__.py', 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ trading/__init__.py not found")
        return False
    except Exception as e:
        print(f"❌ Error reading trading/__init__.py: {e}")
        return False
    
    # Store original content for comparison
    original_content = content
    
    # Fix optimization imports
    content = re.sub(
        r'from \.optimization import \(\s*StrategyOptimizer, PortfolioOptimizer, OptunaOptimizer,\s*BaseOptimizer, OptimizationVisualizer\s*\)',
        'from .optimization import (\n        StrategyOptimizer, BaseOptimizer, OptimizationVisualizer\n    )',
        content
    )
    
    # Fix risk imports
    content = re.sub(
        r'from \.risk import \(\s*RiskManager, PositionSizingEngine, RiskAnalyzer,\s*RiskAdjustedStrategy, RiskMetrics\s*\)',
        'from .risk import (\n        RiskManager\n    )',
        content
    )
    
    # Fix portfolio imports
    content = re.sub(
        r'from \.portfolio import \(\s*PortfolioManager, PortfolioSimulator, PositionSizer,\s*LLMUtils\s*\)',
        'from .portfolio import (\n        PortfolioManager\n    )',
        content
    )
    
    # Fix agents imports
    content = re.sub(
        r'from \.agents import \(\s*PromptRouterAgent, ExecutionAgent, ModelBuilderAgent,\s*StrategySelectorAgent, MarketRegimeAgent, AgentRegistry\s*\)',
        'from .agents import (\n        get_prompt_router_agent, get_model_builder_agent\n    )',
        content
    )
    
    # Fix utils imports
    content = re.sub(
        r'from \.utils import \(\s*SafeExecutor, ReasoningLogger, PerformanceLogger,\s*ErrorHandler, ConfigUtils, DataUtils\s*\)',
        'from .utils import (\n        LoggingManager, DataValidator, ConfigManager, PerformanceLogger\n    )',
        content
    )
    
    # Check if any changes were made
    if content == original_content:
        print("ℹ️ No changes were needed - imports already appear to be correct")
    else:
        # Write the fixed content back
        try:
            with open('trading/__init__.py', 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ Fixed import statements in trading/__init__.py")
        except Exception as e:
            print(f"❌ Error writing to trading/__init__.py: {e}")
            return False
    
    # Validate the fixes
    print("\n🧪 Validating import fixes...")
    if test_imports():
        print("✅ All imports validated successfully!")
        return True
    else:
        print("❌ Some imports still have issues")
        return False

if __name__ == "__main__":
    success = fix_trading_init()
    sys.exit(0 if success else 1) 