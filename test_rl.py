#!/usr/bin/env python3
"""Test RL module imports."""

try:
    from rl.strategy_trainer import create_rl_strategy_trainer
    print("✅ RL module imported successfully!")
except Exception as e:
    print(f"❌ RL module import failed: {e}")

try:
    from models.tft_model import create_tft_model
    print("✅ TFT model imported successfully!")
except Exception as e:
    print(f"❌ TFT model import failed: {e}")

try:
    from causal.causal_model import create_causal_model
    print("✅ Causal model imported successfully!")
except Exception as e:
    print(f"❌ Causal model import failed: {e}")

print("Test completed!") 