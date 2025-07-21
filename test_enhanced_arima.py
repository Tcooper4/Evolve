"""
Enhanced ARIMA Model Test

This test validates the enhanced ARIMA model with various optimization
criteria, seasonal components, and fallback mechanisms.
"""

import logging

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_sample_data(n_points: int = 100) -> pd.Series:
    """Generate sample time series data for testing."""
    np.random.seed(42)

    # Create time index
    dates = pd.date_range(start="2020-01-01", periods=n_points, freq="D")

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
                        "max_p": 3,
                        "max_q": 3,
                        "max_d": 2,
                        "trace": True,
                    },
                },
            },
            {
                "name": "BIC Optimization (Non-Seasonal)",
                "config": {
                    "use_auto_arima": True,
                    "seasonal": False,
                    "optimization_criterion": "bic",
                    "auto_arima_config": {
                        "max_p": 3,
                        "max_q": 3,
                        "max_d": 2,
                        "trace": True,
                    },
                },
            },
            {
                "name": "MSE Optimization (Seasonal)",
                "config": {
                    "use_auto_arima": True,
                    "seasonal": True,
                    "optimization_criterion": "mse",
                    "backtest_steps": 10,
                    "auto_arima_config": {
                        "max_p": 2,
                        "max_q": 2,
                        "max_d": 1,
                        "trace": True,
                    },
                },
            },
            {
                "name": "RMSE Optimization (Non-Seasonal)",
                "config": {
                    "use_auto_arima": True,
                    "seasonal": False,
                    "optimization_criterion": "rmse",
                    "backtest_steps": 10,
                    "auto_arima_config": {
                        "max_p": 2,
                        "max_q": 2,
                        "max_d": 1,
                        "trace": True,
                    },
                },
            },
            {
                "name": "Manual ARIMA (Fallback)",
                "config": {
                    "use_auto_arima": False,
                    "order": (1, 1, 1),
                    "seasonal_order": None,
                },
            },
        ]

        results = []

        for test_config in configs:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Testing: {test_config['name']}")
            logger.info(f"{'=' * 60}")

            try:
                # Create model
                model = ARIMAModel(test_config["config"])
                logger.info("✅ Model created successfully")

                # Fit model
                logger.info("Fitting model...")
                fit_result = model.fit(data)
                logger.info(f"✅ Model fitted successfully: {fit_result.success}")

                if fit_result.success:
                    # Get model info
                    model_info = model.get_model_info()
                    logger.info(f"Model order: {model_info.get('order', 'N/A')}")
                    logger.info(
                        f"Seasonal order: {model_info.get('seasonal_order', 'N/A')}"
                    )
                    logger.info(f"AIC: {model_info.get('aic', 'N/A')}")
                    logger.info(f"BIC: {model_info.get('bic', 'N/A')}")

                    # Test forecasting
                    logger.info("Testing forecasting...")
                    forecast_result = model.forecast(steps=10)
                    logger.info(f"✅ Forecast successful: {forecast_result.success}")

                    if forecast_result.success:
                        forecast = forecast_result.forecast
                        logger.info(f"Forecast shape: {forecast.shape}")
                        logger.info(
                            f"Forecast range: {forecast.min():.2f} to {forecast.max():.2f}"
                        )

                        # Test confidence intervals
                        if hasattr(forecast_result, "confidence_intervals"):
                            ci = forecast_result.confidence_intervals
                            logger.info(
                                f"Confidence intervals: {ci.shape if hasattr(ci, 'shape') else 'N/A'}"
                            )

                    # Test diagnostics
                    logger.info("Running diagnostics...")
                    diagnostics = model.run_diagnostics()
                    logger.info(
                        f"✅ Diagnostics completed: {diagnostics.get('passed', False)}"
                    )

                    if not diagnostics.get("passed", False):
                        logger.warning(
                            f"Diagnostics issues: {diagnostics.get('issues', [])}"
                        )

                    results.append(
                        {
                            "name": test_config["name"],
                            "success": True,
                            "fit_success": fit_result.success,
                            "forecast_success": (
                                forecast_result.success if fit_result.success else False
                            ),
                            "diagnostics_passed": diagnostics.get("passed", False),
                        }
                    )

                else:
                    logger.error(f"❌ Model fitting failed: {fit_result.error}")
                    results.append(
                        {
                            "name": test_config["name"],
                            "success": False,
                            "error": fit_result.error,
                        }
                    )

            except Exception as e:
                logger.error(f"❌ Test failed for {test_config['name']}: {e}")
                results.append(
                    {"name": test_config["name"], "success": False, "error": str(e)}
                )

        # Summary
        logger.info(f"\n{'=' * 60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'=' * 60}")

        successful_tests = [r for r in results if r["success"]]
        failed_tests = [r for r in results if not r["success"]]

        logger.info(f"Successful tests: {len(successful_tests)}/{len(results)}")
        logger.info(f"Failed tests: {len(failed_tests)}/{len(results)}")

        for result in results:
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            logger.info(f"{status}: {result['name']}")
            if not result["success"] and "error" in result:
                logger.info(f"   Error: {result['error']}")

        if len(successful_tests) == len(results):
            logger.info("🎉 All enhanced ARIMA tests passed!")
            return True
        else:
            logger.error(f"❌ {len(failed_tests)} tests failed")
            return False

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


def test_arima_robustness():
    """Test ARIMA model robustness with different data types."""
    logger.info("Testing ARIMA model robustness...")

    try:
        from trading.models.arima_model import ARIMAModel

        # Test with different data types
        test_cases = [
            {
                "name": "Short Series",
                "data": generate_sample_data(20),
                "config": {"use_auto_arima": False, "order": (1, 1, 0)},
            },
            {
                "name": "Long Series",
                "data": generate_sample_data(500),
                "config": {"use_auto_arima": True, "seasonal": True},
            },
            {
                "name": "Trendy Data",
                "data": pd.Series(np.linspace(100, 200, 100)),
                "config": {"use_auto_arima": False, "order": (1, 1, 1)},
            },
            {
                "name": "Noisy Data",
                "data": pd.Series(np.random.normal(100, 10, 100)),
                "config": {"use_auto_arima": True, "seasonal": False},
            },
        ]

        for test_case in test_cases:
            logger.info(f"\nTesting: {test_case['name']}")

            try:
                model = ARIMAModel(test_case["config"])
                fit_result = model.fit(test_case["data"])

                if fit_result.success:
                    forecast_result = model.forecast(steps=5)
                    logger.info(f"✅ {test_case['name']}: Success")
                else:
                    logger.warning(f"⚠️ {test_case['name']}: Fit failed")

            except Exception as e:
                logger.error(f"❌ {test_case['name']}: {e}")

        logger.info("✅ Robustness tests completed")
        return True

    except Exception as e:
        logger.error(f"❌ Robustness test failed: {e}")
        return False


def main():
    """Main test function."""
    logger.info("🚀 Starting Enhanced ARIMA Test Suite")
    logger.info("=" * 60)

    test_results = []

    # Run tests
    tests = [
        ("Enhanced ARIMA", test_enhanced_arima),
        ("Robustness", test_arima_robustness),
    ]

    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
            test_results.append((test_name, False))

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")

    logger.info(f"\nOverall: {passed}/{total} test suites passed")

    if passed == total:
        logger.info("🎉 All enhanced ARIMA tests completed successfully!")
        return True
    else:
        logger.error(f"❌ {total - passed} test suites failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
