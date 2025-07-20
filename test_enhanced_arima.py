#!/usr/bin/env python3
"""
Test script for enhanced ARIMA model with auto_arima optimization.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_sample_data(n_points: int = 100) -> pd.Series:
    """Generate sample time series data for testing."""
    np.random.seed(42)

    # Create time index
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='D')

    # Generate trend + seasonality + noise
    trend = np.linspace(100, 150, n_points)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(n_points) / 7)  # Weekly seasonality
    noise = np.random.normal(0, 5, n_points)

    data = trend + seasonality + noise
    return pd.Series(data, index=dates)

def test_enhanced_arima():
    """Test the enhanced ARIMA model with different configurations."""

    try:
        from trading.models.arima_model import ARIMAModel

        # Generate sample data
        logger.info("Generating sample time series data...")
        data = generate_sample_data(100)
        logger.info(f"Generated {len(data)} data points")

        # Test configurations
        configs = [
            {
                "name": "AIC Optimization (Seasonal)",
                "config": {
                    "use_auto_arima": True,
                    "seasonal": True,
                    "optimization_criterion": "aic",
                    "auto_arima_config": {
                        "max_p": 3, "max_q": 3, "max_d": 2,
                        "trace": True
                    }
                }
            },
            {
                "name": "BIC Optimization (Non-Seasonal)",
                "config": {
                    "use_auto_arima": True,
                    "seasonal": False,
                    "optimization_criterion": "bic",
                    "auto_arima_config": {
                        "max_p": 3, "max_q": 3, "max_d": 2,
                        "trace": True
                    }
                }
            },
            {
                "name": "MSE Optimization (Seasonal)",
                "config": {
                    "use_auto_arima": True,
                    "seasonal": True,
                    "optimization_criterion": "mse",
                    "backtest_steps": 10,
                    "auto_arima_config": {
                        "max_p": 2, "max_q": 2, "max_d": 1,
                        "trace": True
                    }
                }
            },
            {
                "name": "RMSE Optimization (Non-Seasonal)",
                "config": {
                    "use_auto_arima": True,
                    "seasonal": False,
                    "optimization_criterion": "rmse",
                    "backtest_steps": 10,
                    "auto_arima_config": {
                        "max_p": 2, "max_q": 2, "max_d": 1,
                        "trace": True
                    }
                }
            },
            {
                "name": "Manual ARIMA (Fallback)",
                "config": {
                    "use_auto_arima": False,
                    "order": (1, 1, 1),
                    "seasonal_order": None
                }
            }
        ]

        results = []

        for test_config in configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {test_config['name']}")
            logger.info(f"{'='*60}")

            try:
                # Create and fit model
                model = ARIMAModel(test_config['config'])
                fit_result = model.fit(data)

                if fit_result['success']:
                    logger.info(f"✅ Model fitted successfully")
                    logger.info(f"   Order: {fit_result.get('order', 'N/A')}")
                    logger.info(f"   Seasonal Order: {fit_result.get('seasonal_order', 'N/A')}")
                    logger.info(f"   AIC: {fit_result.get('aic', 'N/A')}")
                    logger.info(f"   BIC: {fit_result.get('bic', 'N/A')}")

                    # Make predictions
                    pred_result = model.predict(steps=10)
                    if pred_result['success']:
                        logger.info(f"✅ Predictions generated successfully")
                        logger.info(f"   Forecast shape: {len(pred_result['forecast'])}")

                        # Store results
                        results.append({
                            "name": test_config['name'],
                            "success": True,
                            "order": fit_result.get('order'),
                            "seasonal_order": fit_result.get('seasonal_order'),
                            "aic": fit_result.get('aic'),
                            "bic": fit_result.get('bic'),
                            "optimization_criterion": test_config['config'].get('optimization_criterion', 'manual')
                        })
                    else:
                        logger.error(f"❌ Prediction failed: {pred_result.get('error', 'Unknown error')}")
                        results.append({
                            "name": test_config['name'],
                            "success": False,
                            "error": pred_result.get('error', 'Unknown error')
                        })
                else:
                    logger.error(f"❌ Model fitting failed: {fit_result.get('error', 'Unknown error')}")
                    results.append({
                        "name": test_config['name'],
                        "success": False,
                        "error": fit_result.get('error', 'Unknown error')
                    })

            except Exception as e:
                logger.error(f"❌ Test failed with exception: {e}")
                results.append({
                    "name": test_config['name'],
                    "success": False,
                    "error": str(e)
                })

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*60}")

        successful_tests = [r for r in results if r['success']]
        failed_tests = [r for r in results if not r['success']]

        logger.info(f"✅ Successful tests: {len(successful_tests)}/{len(results)}")
        logger.info(f"❌ Failed tests: {len(failed_tests)}/{len(results)}")

        if successful_tests:
            logger.info(f"\nSuccessful configurations:")
            for result in successful_tests:
                logger.info(f"  - {result['name']}")
                logger.info(f"    Order: {result['order']}")
                if result['seasonal_order'] and result['seasonal_order'] != (0, 0, 0, 0):
                    logger.info(f"    Seasonal: {result['seasonal_order']}")
                logger.info(f"    AIC: {result['aic']:.2f}")
                logger.info(f"    BIC: {result['bic']:.2f}")
                logger.info(f"    Optimization: {result['optimization_criterion'].upper()}")
                logger.info("")

        if failed_tests:
            logger.info(f"\nFailed configurations:")
            for result in failed_tests:
                logger.info(f"  - {result['name']}: {result['error']}")

        return len(successful_tests) == len(results)

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("Make sure you have pmdarima installed: pip install pmdarima")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting Enhanced ARIMA Model Tests")
    logger.info("=" * 60)

    success = test_enhanced_arima()

    if success:
        logger.info("\n🎉 ALL TESTS PASSED!")
    else:
        logger.info("\n❌ SOME TESTS FAILED!")

    logger.info("\nEnhanced ARIMA Features:")
    logger.info("✅ Automatic parameter selection with pmdarima.auto_arima")
    logger.info("✅ Seasonal component control (seasonal=True/False)")
    logger.info("✅ Multiple optimization criteria (AIC, BIC, MSE, RMSE)")
    logger.info("✅ Backtesting for MSE/RMSE optimization")
    logger.info("✅ Fallback to manual ARIMA if auto_arima fails")
    logger.info("✅ Comprehensive logging and error handling")

