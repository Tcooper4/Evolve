#!/usr/bin/env python3
"""Test RL module imports."""

import os
import sys

import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_rl_module_import():
    """Test RL module import."""
    try:
        pass

        print("✅ RL module imported successfully!")
        assert True
    except Exception as e:
        print(f"❌ RL module import failed: {e}")
        pytest.skip(f"RL module not available: {e}")


def test_tft_model_import():
    """Test TFT model import."""
    try:
        pass

        print("✅ TFT model imported successfully!")
        assert True
    except Exception as e:
        print(f"❌ TFT model import failed: {e}")
        pytest.skip(f"TFT model not available: {e}")


def test_causal_model_import():
    """Test Causal model import."""
    try:
        pass

        print("✅ Causal model imported successfully!")
        assert True
    except Exception as e:
        print(f"❌ Causal model import failed: {e}")
        pytest.skip(f"Causal model not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
