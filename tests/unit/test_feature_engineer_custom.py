import pandas as pd
import numpy as np
from trading.feature_engineering import FeatureEngineer


def create_data(rows: int = 30) -> pd.DataFrame:
    dates = pd.date_range(start="2023-01-01", periods=rows, freq="D")
    np.random.seed(0)
    data = pd.DataFrame({
        "open": np.random.uniform(100, 110, size=rows),
        "high": np.random.uniform(110, 120, size=rows),
        "low": np.random.uniform(90, 100, size=rows),
        "close": np.random.uniform(100, 110, size=rows),
        "volume": np.random.uniform(1e5, 1e6, size=rows),
    }, index=dates)
    return data


def test_default_custom_indicators_registered():
    fe = FeatureEngineer()
    assert "ROLLING_ZSCORE" in fe.custom_indicators
    assert "PRICE_RATIOS" in fe.custom_indicators


def test_engineer_features_contains_custom_columns():
    fe = FeatureEngineer()
    data = create_data()
    features = fe.engineer_features(data)
    assert "ROLLING_ZSCORE" in features.columns
    assert "PRICE_RATIOS_HL_RATIO" in features.columns
    assert "PRICE_RATIOS_CO_RATIO" in features.columns
