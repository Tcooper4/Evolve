"""
Macroeconomic Feature Engineering

Enriches trading data with macroeconomic indicators from FRED and World Bank.
Provides inflation, GDP, unemployment, and interest rate features for enhanced forecasting.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Import data providers with fallback handling
try:
    import fredapi

    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    logging.warning("FRED API not available. Install with: pip install fredapi")

try:
    import wbdata

    WBDATA_AVAILABLE = True
except ImportError:
    WBDATA_AVAILABLE = False
    logging.warning("World Bank Data not available. Install with: pip install wbdata")

from trading.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class MacroFeatureEngineer:
    """Macroeconomic feature engineering for trading data enrichment."""

    def __init__(self, fred_api_key: Optional[str] = None):
        """Initialize the macro feature engineer.

        Args:
            fred_api_key: Optional FRED API key for enhanced data access
        """
        self.fred_api_key = fred_api_key or os.getenv("FRED_API_KEY")
        self.cache_dir = Path("cache/macro_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize FRED client if available
        self.fred_client = None
        if FRED_AVAILABLE and self.fred_api_key:
            try:
                self.fred_client = fredapi.Fred(api_key=self.fred_api_key)
                logger.info("FRED API client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize FRED client: {e}")

        # Define FRED series IDs for key indicators
        self.fred_series = {
            "inflation": "CPIAUCSL",  # Consumer Price Index
            "gdp": "GDP",  # Gross Domestic Product
            "unemployment": "UNRATE",  # Unemployment Rate
            "fed_funds": "FEDFUNDS",  # Federal Funds Rate
            "treasury_10y": "GS10",  # 10-Year Treasury Rate
            "industrial_production": "INDPRO",  # Industrial Production
            "retail_sales": "RSAFS",  # Retail Sales
            "housing_starts": "HOUST",  # Housing Starts
            "consumer_confidence": "UMCSENT",  # Consumer Sentiment
            "vix": "VIXCLS",  # VIX Volatility Index
        }

        # Define World Bank indicators
        self.wb_indicators = {
            "gdp_growth": "NY.GDP.MKTP.KD.ZG",  # GDP growth
            "inflation_wb": "FP.CPI.TOTL.ZG",  # Inflation
            "interest_rate": "FR.INR.RINR",  # Interest rate
            "unemployment_wb": "SL.UEM.TOTL.ZS",  # Unemployment
        }

        logger.info("Macro feature engineer initialized")

    def get_fred_data(self, series_ids: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from FRED API.

        Args:
            series_ids: List of FRED series IDs
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with FRED data
        """
        if not self.fred_client:
            logger.warning("FRED client not available, using cached data")
            return self._get_cached_macro_data("fred", start_date, end_date)

        try:
            data = {}
            for series_id in series_ids:
                try:
                    series_data = self.fred_client.get_series(
                        series_id, observation_start=start_date, observation_end=end_date
                    )
                    data[series_id] = series_data
                    logger.debug(f"Fetched FRED data for {series_id}")
                except Exception as e:
                    logger.warning(f"Failed to fetch {series_id}: {e}")

            # Combine into DataFrame
            df = pd.DataFrame(data)
            df.index.name = "date"

            # Cache the data
            self._cache_macro_data("fred", df, start_date, end_date)

            return df

        except Exception as e:
            logger.error(f"Error fetching FRED data: {e}")
            return self._get_cached_macro_data("fred", start_date, end_date)

    def get_worldbank_data(self, country: str = "US", start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch data from World Bank API.

        Args:
            country: Country code (default: US)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with World Bank data
        """
        if not WBDATA_AVAILABLE:
            logger.warning("World Bank Data not available, using cached data")
            return self._get_cached_macro_data("worldbank", start_date, end_date)

        try:
            # Convert dates to datetime
            start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime(2010, 1, 1)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

            # Fetch data
            data = wbdata.get_dataframe(self.wb_indicators, country=country, data_date=(start_dt, end_dt))

            # Clean column names
            data.columns = [col.split(".")[-1] for col in data.columns]

            # Cache the data
            self._cache_macro_data("worldbank", data, start_date, end_date)

            return data

        except Exception as e:
            logger.error(f"Error fetching World Bank data: {e}")
            return self._get_cached_macro_data("worldbank", start_date, end_date)

    def enrich_trading_data(self, df: pd.DataFrame, include_sentiment: bool = True) -> pd.DataFrame:
        """Enrich trading data with macroeconomic features.

        Args:
            df: Trading data DataFrame with datetime index
            include_sentiment: Whether to include sentiment indicators

        Returns:
            Enriched DataFrame with macro features
        """
        try:
            enriched_df = df.copy()

            # Get date range from trading data
            start_date = df.index.min().strftime("%Y-%m-%d")
            end_date = df.index.max().strftime("%Y-%m-%d")

            # Fetch FRED data
            fred_series = list(self.fred_series.values())
            if include_sentiment:
                fred_series.extend(["UMCSENT", "VIXCLS"])  # Add sentiment indicators

            fred_data = self.get_fred_data(fred_series, start_date, end_date)

            # Fetch World Bank data
            wb_data = self.get_worldbank_data("US", start_date, end_date)

            # Merge FRED data
            if not fred_data.empty:
                # Resample to match trading data frequency
                fred_resampled = fred_data.resample("D").ffill()
                enriched_df = enriched_df.join(fred_resampled, how="left")

                # Forward fill missing values
                enriched_df = enriched_df.fillna(method="ffill")

                # Add lagged features
                enriched_df = self._add_lagged_features(enriched_df, fred_data.columns)

                # Add rolling statistics
                enriched_df = self._add_rolling_features(enriched_df, fred_data.columns)

            # Merge World Bank data
            if not wb_data.empty:
                wb_resampled = wb_data.resample("D").ffill()
                enriched_df = enriched_df.join(wb_resampled, how="left")
                enriched_df = enriched_df.fillna(method="ffill")

            # Add derived features
            enriched_df = self._add_derived_features(enriched_df)

            # Add market regime features
            enriched_df = self._add_market_regime_features(enriched_df)

            logger.info(f"Enriched trading data with {len(enriched_df.columns) - len(df.columns)} macro features")
            return enriched_df

        except Exception as e:
            logger.error(f"Error enriching trading data: {e}")
            return df

    def _add_lagged_features(self, df: pd.DataFrame, macro_columns: List[str]) -> pd.DataFrame:
        """Add lagged versions of macroeconomic features."""
        for col in macro_columns:
            if col in df.columns:
                # Add various lags
                for lag in [1, 3, 7, 30, 90]:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)

                # Add percentage changes
                df[f"{col}_pct_change"] = df[col].pct_change()
                df[f"{col}_pct_change_3m"] = df[col].pct_change(periods=90)

        return df

    def _add_rolling_features(self, df: pd.DataFrame, macro_columns: List[str]) -> pd.DataFrame:
        """Add rolling statistics for macroeconomic features."""
        for col in macro_columns:
            if col in df.columns:
                # Rolling means
                for window in [7, 30, 90]:
                    df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window=window).mean()

                # Rolling standard deviations
                for window in [30, 90]:
                    df[f"{col}_rolling_std_{window}"] = df[col].rolling(window=window).std()

                # Z-score (deviation from rolling mean)
                df[f"{col}_zscore_30"] = (df[col] - df[col].rolling(window=30).mean()) / df[col].rolling(
                    window=30
                ).std()

        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived macroeconomic features."""
        try:
            # Real interest rate (nominal - inflation)
            if "FEDFUNDS" in df.columns and "CPIAUCSL" in df.columns:
                df["real_interest_rate"] = df["FEDFUNDS"] - df["CPIAUCSL"].pct_change(periods=12) * 100

            # Yield curve slope (10Y - 3M)
            if "GS10" in df.columns and "FEDFUNDS" in df.columns:
                df["yield_curve_slope"] = df["GS10"] - df["FEDFUNDS"]

            # Economic stress index (composite of multiple indicators)
            stress_components = []
            if "UNRATE" in df.columns:
                stress_components.append(df["UNRATE"])
            if "VIXCLS" in df.columns:
                stress_components.append(df["VIXCLS"] / 100)  # Normalize VIX

            if stress_components:
                df["economic_stress_index"] = np.mean(stress_components, axis=0)

            # Business cycle indicator
            if "INDPRO" in df.columns and "UNRATE" in df.columns:
                # Simple business cycle indicator
                industrial_growth = df["INDPRO"].pct_change(periods=12)
                unemployment_change = df["UNRATE"].diff(periods=12)
                df["business_cycle_indicator"] = industrial_growth - unemployment_change

        except Exception as e:
            logger.warning(f"Error adding derived features: {e}")

        return df

    def _add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime classification features."""
        try:
            # Volatility regime
            if "VIXCLS" in df.columns:
                df["volatility_regime"] = pd.cut(
                    df["VIXCLS"],
                    bins=[0, 15, 25, 35, 100],
                    labels=["Low", "Normal", "High", "Extreme"],
                    include_lowest=True,
                )

            # Interest rate regime
            if "FEDFUNDS" in df.columns:
                df["interest_rate_regime"] = pd.cut(
                    df["FEDFUNDS"],
                    bins=[0, 2, 4, 6, 20],
                    labels=["Very Low", "Low", "Normal", "High"],
                    include_lowest=True,
                )

            # Economic growth regime
            if "GDP" in df.columns:
                gdp_growth = df["GDP"].pct_change(periods=4) * 100  # Quarterly growth
                df["growth_regime"] = pd.cut(
                    gdp_growth,
                    bins=[-10, 0, 2, 5, 20],
                    labels=["Recession", "Slow", "Normal", "Strong"],
                    include_lowest=True,
                )

        except Exception as e:
            logger.warning(f"Error adding market regime features: {e}")

        return df

    def _cache_macro_data(self, source: str, data: pd.DataFrame, start_date: str, end_date: str):
        """Cache macroeconomic data to avoid repeated API calls."""
        try:
            cache_file = self.cache_dir / f"{source}_{start_date}_{end_date}.parquet"
            data.to_parquet(cache_file)
            logger.debug(f"Cached {source} data to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache {source} data: {e}")

    def _get_cached_macro_data(self, source: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve cached macroeconomic data."""
        try:
            cache_file = self.cache_dir / f"{source}_{start_date}_{end_date}.parquet"
            if cache_file.exists():
                data = pd.read_parquet(cache_file)
                logger.debug(f"Loaded cached {source} data from {cache_file}")
                return data
        except Exception as e:
            logger.warning(f"Failed to load cached {source} data: {e}")

        # Return empty DataFrame if no cache available
        return pd.DataFrame()

    def get_feature_importance(self, df: pd.DataFrame, target_col: str = "returns") -> Dict[str, float]:
        """Calculate feature importance for macroeconomic features."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import LabelEncoder

            # Prepare features
            feature_cols = [
                col
                for col in df.columns
                if any(macro in col for macro in ["CPIAUCSL", "GDP", "UNRATE", "FEDFUNDS", "GS10", "VIXCLS"])
            ]

            if not feature_cols or target_col not in df.columns:
                return {}

            # Prepare data
            X = df[feature_cols].fillna(method="ffill").fillna(0)
            y = df[target_col].fillna(0)

            # Encode categorical features
            le = LabelEncoder()
            for col in X.select_dtypes(include=["object", "category"]).columns:
                X[col] = le.fit_transform(X[col].astype(str))

            # Train model
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)

            # Get feature importance
            importance = dict(zip(feature_cols, rf.feature_importances_))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}


# Global macro feature engineer instance
_macro_engineer = None


def get_macro_engineer() -> MacroFeatureEngineer:
    """Get the global macro feature engineer instance."""
    global _macro_engineer
    if _macro_engineer is None:
        _macro_engineer = MacroFeatureEngineer()
    return _macro_engineer
