# -*- coding: utf-8 -*-
"""Resilient forecasting-backend loader shared by the Analyze and Backtest
pages.

Both pages previously defined their own ``_get_forecasting_backend`` with a
single try/except around ELEVEN imports: one missing optional dependency
(verified: statsmodels -> ARIMAModel) returned None and the page called
``st.stop()`` - the whole page bricked because one of four forecast models
couldn't import. This loader imports each component individually, reports
what's missing, and lets a page run with whatever subset is available.

A backend is considered usable when the data layer loads and at least one
forecast model is present.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# name -> (module path, attribute)
_COMPONENTS: Dict[str, Tuple[str, str]] = {
    "DataLoader": ("trading.data.data_loader", "DataLoader"),
    "DataLoadRequest": ("trading.data.data_loader", "DataLoadRequest"),
    "YFinanceProvider": (
        "trading.data.providers.yfinance_provider", "YFinanceProvider"
    ),
    "LSTMForecaster": ("trading.models.lstm_model", "LSTMForecaster"),
    "XGBoostModel": ("trading.models.xgboost_model", "XGBoostModel"),
    "ProphetModel": ("trading.models.prophet_model", "ProphetModel"),
    "ARIMAModel": ("trading.models.arima_model", "ARIMAModel"),
    "FeatureEngineering": ("trading.data.preprocessing", "FeatureEngineering"),
    "DataPreprocessor": ("trading.data.preprocessing", "DataPreprocessor"),
    "ModelSelectorAgent": (
        "trading.agents.model_selector_agent", "ModelSelectorAgent"
    ),
    "MarketAnalyzer": ("trading.market.market_analyzer", "MarketAnalyzer"),
}

_CORE = ("DataLoader", "DataLoadRequest", "YFinanceProvider")
_MODELS = ("LSTMForecaster", "XGBoostModel", "ProphetModel", "ARIMAModel")


def load_forecasting_backend() -> Tuple[Dict[str, Any], List[str]]:
    """Import every backend component individually.

    Returns:
        (backend, missing): ``backend`` maps component name -> class for
        everything that imported; ``missing`` lists human-readable
        "Name (reason)" entries for everything that didn't. ``backend`` is
        ``{}`` when the result is unusable (core data layer absent or no
        forecast model at all).
    """
    backend: Dict[str, Any] = {}
    missing: List[str] = []
    for name, (module_path, attr) in _COMPONENTS.items():
        try:
            module = importlib.import_module(module_path)
            backend[name] = getattr(module, attr)
        except Exception as e:  # noqa: BLE001 - report every failure kind
            missing.append(f"{name} ({type(e).__name__}: {e})")
            logger.warning("Forecasting backend: %s unavailable: %s", name, e)

    core_ok = all(k in backend for k in _CORE)
    any_model = any(k in backend for k in _MODELS)
    if not (core_ok and any_model):
        return {}, missing
    return backend, missing
