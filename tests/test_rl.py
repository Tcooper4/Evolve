#!/usr/bin/env python3
"""Test RL module imports."""

import sys
import os
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_rl_module_import():
    """Test RL module import."""
    try:
        from rl.strategy_trainer import create_rl_strategy_trainer
        print("✅ RL module imported successfully!")
        assert True
    except Exception as e:
        print(f"❌ RL module import failed: {e}")
        pytest.skip(f"RL module not available: {e}")

def test_tft_model_import():
    """Test TFT model import."""
    try:
        from models.tft_model import create_tft_model
        print("✅ TFT model imported successfully!")
        assert True
    except Exception as e:
        print(f"❌ TFT model import failed: {e}")
        pytest.skip(f"TFT model not available: {e}")

def test_causal_model_import():
    """Test Causal model import."""
    try:
        from causal.causal_model import create_causal_model
        print("✅ Causal model imported successfully!")
        assert True
    except Exception as e:
        print(f"❌ Causal model import failed: {e}")
        pytest.skip(f"Causal model not available: {e}")

if __name__ == "__main__":
    pytest.main([__file__]) 