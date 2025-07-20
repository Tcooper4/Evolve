#!/usr/bin/env python3
"""
Example usage of enhanced ARIMA model with auto_arima optimization.
"""

import pandas as pd
import numpy as np
import logging
from trading.models.arima_model import ARIMAModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Demonstrate enhanced ARIMA model usage."""
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    trend = np.linspace(100, 150, 100)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(100) / 7)
    noise = np.random.normal(0, 5, 100)
    data = pd.Series(trend + seasonality + noise, index=dates)
    
    logger.info("Enhanced ARIMA Model Examples")
    logger.info("=" * 50)
    
    # Example 1: AIC Optimization with Seasonal Components
    logger.info("\n1. AIC Optimization (Seasonal)")
    logger.info("-" * 30)
    
    config_aic = {
        "use_auto_arima": True,
        "seasonal": True,
        "optimization_criterion": "aic",
        "auto_arima_config": {
            "max_p": 3, "max_q": 3, "max_d": 2,
            "trace": True
        }
    }
    
    model_aic = ARIMAModel(config_aic)
    result_aic = model_aic.fit(data)
    
    if result_aic['success']:
        logger.info(f"âœ… Model fitted successfully")
        logger.info(f"   Order: {result_aic['order']}")
        logger.info(f"   Seasonal Order: {result_aic['seasonal_order']}")
        logger.info(f"   AIC: {result_aic['aic']:.2f}")
        
        # Make predictions
        pred_aic = model_aic.predict(steps=10)
        if pred_aic['success']:
            logger.info(f"   Forecast: {pred_aic['forecast'][:5]}...")
    
    # Example 2: BIC Optimization without Seasonal Components
    logger.info("\n2. BIC Optimization (Non-Seasonal)")
    logger.info("-" * 30)
    
    config_bic = {
        "use_auto_arima": True,
        "seasonal": False,
        "optimization_criterion": "bic",
        "auto_arima_config": {
            "max_p": 3, "max_q": 3, "max_d": 2,
            "trace": True
        }
    }
    
    model_bic = ARIMAModel(config_bic)
    result_bic = model_bic.fit(data)
    
    if result_bic['success']:
        logger.info(f"âœ… Model fitted successfully")
        logger.info(f"   Order: {result_bic['order']}")
        logger.info(f"   Seasonal Order: {result_bic['seasonal_order']}")
        logger.info(f"   BIC: {result_bic['bic']:.2f}")
        
        # Make predictions
        pred_bic = model_bic.predict(steps=10)
        if pred_bic['success']:
            logger.info(f"   Forecast: {pred_bic['forecast'][:5]}...")
    
    # Example 3: MSE Optimization with Backtesting
    logger.info("\n3. MSE Optimization (Backtesting)")
    logger.info("-" * 30)
    
    config_mse = {
        "use_auto_arima": True,
        "seasonal": True,
        "optimization_criterion": "mse",
        "backtest_steps": 10,
        "auto_arima_config": {
            "max_p": 2, "max_q": 2, "max_d": 1,
            "trace": True
        }
    }
    
    model_mse = ARIMAModel(config_mse)
    result_mse = model_mse.fit(data)
    
    if result_mse['success']:
        logger.info(f"âœ… Model fitted successfully")
        logger.info(f"   Order: {result_mse['order']}")
        logger.info(f"   Seasonal Order: {result_mse['seasonal_order']}")
        logger.info(f"   AIC: {result_mse['aic']:.2f}")
        
        # Make predictions
        pred_mse = model_mse.predict(steps=10)
        if pred_mse['success']:
            logger.info(f"   Forecast: {pred_mse['forecast'][:5]}...")
    
    # Example 4: RMSE Optimization without Seasonal Components
    logger.info("\n4. RMSE Optimization (Non-Seasonal)")
    logger.info("-" * 30)
    
    config_rmse = {
        "use_auto_arima": True,
        "seasonal": False,
        "optimization_criterion": "rmse",
        "backtest_steps": 10,
        "auto_arima_config": {
            "max_p": 2, "max_q": 2, "max_d": 1,
            "trace": True
        }
    }
    
    model_rmse = ARIMAModel(config_rmse)
    result_rmse = model_rmse.fit(data)
    
    if result_rmse['success']:
        logger.info(f"âœ… Model fitted successfully")
        logger.info(f"   Order: {result_rmse['order']}")
        logger.info(f"   Seasonal Order: {result_rmse['seasonal_order']}")
        logger.info(f"   AIC: {result_rmse['aic']:.2f}")
        
        # Make predictions
        pred_rmse = model_rmse.predict(steps=10)
        if pred_rmse['success']:
            logger.info(f"   Forecast: {pred_rmse['forecast'][:5]}...")
    
    # Example 5: Manual ARIMA (Fallback)
    logger.info("\n5. Manual ARIMA (Fallback)")
    logger.info("-" * 30)
    
    config_manual = {
        "use_auto_arima": False,
        "order": (1, 1, 1),
        "seasonal_order": None
    }
    
    model_manual = ARIMAModel(config_manual)
    result_manual = model_manual.fit(data)
    
    if result_manual['success']:
        logger.info(f"âœ… Model fitted successfully")
        logger.info(f"   Order: {result_manual['order']}")
        logger.info(f"   Seasonal Order: {result_manual['seasonal_order']}")
        logger.info(f"   AIC: {result_manual['aic']:.2f}")
        
        # Make predictions
        pred_manual = model_manual.predict(steps=10)
        if pred_manual['success']:
            logger.info(f"   Forecast: {pred_manual['forecast'][:5]}...")
    
    logger.info("\n" + "=" * 50)
    logger.info("Enhanced ARIMA Features Summary:")
    logger.info("âœ… Automatic parameter selection with pmdarima.auto_arima")
    logger.info("âœ… Seasonal component control (seasonal=True/False)")
    logger.info("âœ… Multiple optimization criteria (AIC, BIC, MSE, RMSE)")
    logger.info("âœ… Backtesting for MSE/RMSE optimization")
    logger.info("âœ… Fallback to manual ARIMA if auto_arima fails")
    logger.info("âœ… Comprehensive logging and error handling")

if __name__ == "__main__":
    main()
