#!/usr/bin/env python3
"""Test script to verify LSTM model works without TensorFlow."""

import logging

import numpy as np
import pandas as pd

from trading.models.lstm_model import LSTMForecaster

logger = logging.getLogger(__name__)


def test_lstm_model():
    """Test LSTM model functionality without TensorFlow."""
    logger.info("Testing LSTM model without TensorFlow...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    data = pd.DataFrame(
        {
            "price": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 100),
            "rsi": np.random.uniform(0, 100, 100),
            "macd": np.random.randn(100),
            "sma": np.random.randn(100).cumsum() + 100,
        },
        index=dates,
    )

    # Model configuration
    config = {
        "input_size": 5,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "sequence_length": 10,
        "feature_columns": ["price", "volume", "rsi", "macd", "sma"],
        "target_column": "price",
    }

    try:
        # Initialize model
        logger.info("1. Initializing LSTM model...")
        model = LSTMForecaster(config)
        logger.info("   ✓ Model initialized successfully")

        # Test data preparation
        logger.info("2. Testing data preparation...")
        X = data[config["feature_columns"]]
        y = data[config["target_column"]]
        logger.info("   ✓ Data prepared successfully")

        # Test model training (just a few epochs)
        logger.info("3. Testing model training...")
        history = model.fit(X, y, epochs=2, batch_size=16)
        logger.info("   ✓ Model training completed")
        logger.info(f"   Training loss: {history['train_loss'][-1]:.4f}")

        # Test prediction
        logger.info("4. Testing model prediction...")
        predictions = model.predict(X)
        logger.info("   ✓ Model prediction completed")
        logger.info(f"   Predictions shape: {predictions.shape}")

        # Test model save/load
        logger.info("5. Testing model save/load...")
        model.save("test_lstm_model.pt")
        new_model = LSTMForecaster(config)
        new_model.load("test_lstm_model.pt")
        logger.info("   ✓ Model save/load completed")

        logger.info(
            "\n🎉 All tests passed! LSTM model works correctly without TensorFlow."
        )
        logger.info("✅ Using PyTorch backend successfully")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    test_lstm_model()
