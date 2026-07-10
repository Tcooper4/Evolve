"""
Macro and Earnings Data Integration

Pulls FRED, yield curve, inflation, and earnings data for market analysis.
Provides comprehensive macroeconomic data integration and analysis.
"""

import json
import logging
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Data sources for macro data."""

    FRED = "fred"
    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    EARNINGS = "earnings"
    CUSTOM = "custom"


@dataclass
class MacroIndicator:
    """Macroeconomic indicator data."""

    name: str
    value: float
    unit: str
    frequency: str
    last_updated: datetime
    source: DataSource
    metadata: Dict[str, Any]


@dataclass
class YieldCurveData:
    """Yield curve data."""

    date: datetime
    rates: Dict[str, float]  # maturity -> rate
    spread_10y_2y: float
    spread_10y_3m: float
    curve_slope: float
    metadata: Dict[str, Any]


@dataclass
class EarningsData:
    """Earnings data for companies."""

    symbol: str
    date: datetime
    actual_eps: float
    estimated_eps: float
    surprise: float
    surprise_percent: float
    revenue: float
    revenue_estimate: float
    revenue_surprise: float
    metadata: Dict[str, Any]


class MacroDataIntegration:
    """Advanced macro and earnings data integration system."""

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        alpha_vantage_api_key: Optional[str] = None,
        cache_dir: str = "cache/macro_data",
    ):
        """Initialize the macro data integration system.

        Args:
            fred_api_key: FRED API key for economic data
            alpha_vantage_api_key: Alpha Vantage API key for additional data
            cache_dir: Directory to cache data
        """
        self.fred_api_key = fred_api_key or os.getenv("FRED_API_KEY")
        self.alpha_vantage_api_key = alpha_vantage_api_key or os.getenv(
            "ALPHA_VANTAGE_API_KEY"
        )
        self.cache_dir = cache_dir

        # Create cache directory
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create cache_dir: {e}")

        # Initialize data sources
        self.fred_series = self._initialize_fred_series()
        self.yield_curve_maturities = self._initialize_yield_curve_maturities()
        self.earnings_calendar = {}

        # Data cache
        self.data_cache = {}
        self.cache_expiry = {}

        logger.info("Macro Data Integration initialized successfully")

    def _initialize_fred_series(self) -> Dict[str, Dict[str, Any]]:
        """Initialize FRED data series."""
        return {
            "GDP": {
                "series_id": "GDP",
                "name": "Gross Domestic Product",
                "frequency": "quarterly",
                "unit": "billions of dollars",
                "description": "Real GDP growth rate",
            },
            "UNRATE": {
                "series_id": "UNRATE",
                "name": "Unemployment Rate",
                "frequency": "monthly",
                "unit": "percent",
                "description": "Civilian unemployment rate",
            },
            "CPIAUCSL": {
                "series_id": "CPIAUCSL",
                "name": "Consumer Price Index",
                "frequency": "monthly",
                "unit": "index",
                "description": "CPI for all urban consumers",
            },
            "FEDFUNDS": {
                "series_id": "FEDFUNDS",
                "name": "Federal Funds Rate",
                "frequency": "monthly",
                "unit": "percent",
                "description": "Effective federal funds rate",
            },
            "DGS10": {
                "series_id": "DGS10",
                "name": "10-Year Treasury Rate",
                "frequency": "daily",
                "unit": "percent",
                "description": "10-year Treasury constant maturity rate",
            },
            "DGS2": {
                "series_id": "DGS2",
                "name": "2-Year Treasury Rate",
                "frequency": "daily",
                "unit": "percent",
                "description": "2-year Treasury constant maturity rate",
            },
            "DGS3MO": {
                "series_id": "DGS3MO",
                "name": "3-Month Treasury Rate",
                "frequency": "daily",
                "unit": "percent",
                "description": "3-month Treasury bill rate",
            },
            "VIXCLS": {
                "series_id": "VIXCLS",
                "name": "VIX Volatility Index",
                "frequency": "daily",
                "unit": "index",
                "description": "CBOE volatility index",
            },
            "DEXUSEU": {
                "series_id": "DEXUSEU",
                "name": "US/Euro Exchange Rate",
                "frequency": "daily",
                "unit": "dollars per euro",
                "description": "US dollar to euro exchange rate",
            },
            "DCOILWTICO": {
                "series_id": "DCOILWTICO",
                "name": "WTI Crude Oil Price",
                "frequency": "daily",
                "unit": "dollars per barrel",
                "description": "West Texas Intermediate crude oil price",
            },
        }

    def _initialize_yield_curve_maturities(self) -> List[str]:
        """Initialize yield curve maturities."""
        return ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

    def get_fred_data(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get data from FRED API."""
        try:
            # Validate inputs
            if not series_id or not isinstance(series_id, str):
                raise ValueError("series_id must be a non-empty string")

            if start_date and not isinstance(start_date, str):
                raise ValueError("start_date must be a string")

            if end_date and not isinstance(end_date, str):
                raise ValueError("end_date must be a string")

            # Log request
            logger.info(f"Fetching FRED data for series: {series_id}")
            start_time = datetime.now()

            if not self.fred_api_key:
                logger.warning("No FRED API key provided, using fallback data")
                return self._get_fallback_fred_data(series_id)

            # Check cache first
            cache_key = f"fred_{series_id}"
            if self._is_cache_valid(cache_key):
                logger.info(f"Using cached FRED data for {series_id}")
                return self.data_cache[cache_key]

            # Build API URL
            base_url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.fred_api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1000,
            }

            if start_date:
                params["observation_start"] = start_date
            if end_date:
                params["observation_end"] = end_date

            # Make API request
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()

            if "observations" not in data:
                raise ValueError("No observations found in FRED response")

            # Parse data
            observations = data["observations"]
            dates = []
            values = []

            for obs in observations:
                try:
                    date = pd.to_datetime(obs["date"])
                    value = float(obs["value"]) if obs["value"] != "." else np.nan
                    dates.append(date)
                    values.append(value)
                except (ValueError, KeyError):
                    continue

            # Create DataFrame
            df = (
                pd.DataFrame({"date": dates, "value": values})
                .set_index("date")
                .sort_index()
            )

            # Cache data
            self._cache_data(cache_key, df)

            # Log fetch latency
            fetch_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"FRED data fetch completed in {fetch_time:.2f}s for {series_id}: {len(df)} observations"
            )
            return df

        except Exception as e:
            logger.error(f"Error getting FRED data for {series_id}: {e}")
            return self._get_fallback_fred_data(series_id)

    def _get_fallback_fred_data(self, series_id: str) -> pd.DataFrame:
        """Generate fallback FRED data when API is unavailable."""
        try:
            # Generate synthetic data based on series type
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)

            dates = pd.date_range(start=start_date, end=end_date, freq="D")

            if series_id in ["DGS10", "DGS2", "DGS3MO"]:
                # Treasury rates - trending upward
                base_rate = {"DGS10": 4.0, "DGS2": 4.5, "DGS3MO": 5.0}.get(
                    series_id, 4.0
                )
                values = (
                    base_rate
                    + 0.5 * np.sin(np.linspace(0, 4 * np.pi, len(dates)))
                    + np.random.normal(0, 0.1, len(dates))
                )

            elif series_id == "VIXCLS":
                # VIX - mean-reverting around 20
                values = (
                    20
                    + 5 * np.sin(np.linspace(0, 6 * np.pi, len(dates)))
                    + np.random.normal(0, 2, len(dates))
                )
                values = np.maximum(values, 10)  # VIX can't go below ~10

            elif series_id == "UNRATE":
                # Unemployment rate - around 4%
                values = (
                    4.0
                    + 0.5 * np.sin(np.linspace(0, 2 * np.pi, len(dates)))
                    + np.random.normal(0, 0.1, len(dates))
                )

            elif series_id == "CPIAUCSL":
                # CPI - slowly increasing
                values = 300 + np.cumsum(np.random.normal(0.1, 0.05, len(dates)))

            else:
                # Generic series
                values = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))

            df = pd.DataFrame({"date": dates, "value": values}).set_index("date")

            logger.info(f"Generated fallback data for {series_id}")
            return df

        except Exception as e:
            logger.error(f"Error generating fallback data: {e}")
            return pd.DataFrame()

    def get_yield_curve_data(self, date: Optional[datetime] = None) -> YieldCurveData:
        """Get current yield curve data."""
        try:
            if date is None:
                date = datetime.now()

            # Get rates for different maturities
            rates = {}

            # Get 3M, 2Y, 10Y rates from FRED
            maturity_mapping = {"3M": "DGS3MO", "2Y": "DGS2", "10Y": "DGS10"}

            for maturity, series_id in maturity_mapping.items():
                df = self.get_fred_data(series_id)
                if not df.empty:
                    rates[maturity] = df["value"].iloc[-1]

            # Generate other maturities using interpolation
            if "3M" in rates and "2Y" in rates and "10Y" in rates:
                # Simple linear interpolation for demonstration
                rates["1M"] = rates["3M"] - 0.1
                rates["6M"] = rates["3M"] + 0.2
                rates["1Y"] = rates["2Y"] - 0.3
                rates["3Y"] = rates["2Y"] + 0.2
                rates["5Y"] = rates["2Y"] + 0.4
                rates["7Y"] = rates["10Y"] - 0.3
                rates["20Y"] = rates["10Y"] + 0.1
                rates["30Y"] = rates["10Y"] + 0.2

            # Calculate spreads
            spread_10y_2y = rates.get("10Y", 0) - rates.get("2Y", 0)
            spread_10y_3m = rates.get("10Y", 0) - rates.get("3M", 0)

            # Calculate curve slope (10Y - 2Y)
            curve_slope = spread_10y_2y

            return YieldCurveData(
                date=date,
                rates=rates,
                spread_10y_2y=spread_10y_2y,
                spread_10y_3m=spread_10y_3m,
                curve_slope=curve_slope,
                metadata={
                    "source": "FRED + interpolation",
                    "maturities_available": list(rates.keys()),
                },
            )

        except Exception as e:
            logger.error(f"Error getting yield curve data: {e}")
            return self._create_fallback_yield_curve(date)

    def _create_fallback_yield_curve(self, date: datetime) -> YieldCurveData:
        """Create fallback yield curve data."""
        rates = {
            "1M": 4.8,
            "3M": 5.0,
            "6M": 5.1,
            "1Y": 4.9,
            "2Y": 4.5,
            "3Y": 4.3,
            "5Y": 4.2,
            "7Y": 4.1,
            "10Y": 4.0,
            "20Y": 4.1,
            "30Y": 4.2,
        }

        return YieldCurveData(
            date=date,
            rates=rates,
            spread_10y_2y=-0.5,
            spread_10y_3m=-1.0,
            curve_slope=-0.5,
            metadata={"source": "fallback", "note": "synthetic data"},
        )

    def get_earnings_data(
        self, symbol: str, lookback_days: int = 365
    ) -> List[EarningsData]:
        """Get earnings data for a symbol."""
        try:
            # Validate inputs
            if not symbol or not isinstance(symbol, str):
                raise ValueError("symbol must be a non-empty string")

            if not isinstance(lookback_days, int) or lookback_days <= 0:
                raise ValueError("lookback_days must be a positive integer")

            # Log request
            logger.info(f"Fetching earnings data for symbol: {symbol}")
            start_time = datetime.now()

            # Check cache first
            cache_key = f"earnings_{symbol}"
            if self._is_cache_valid(cache_key):
                logger.info(f"Using cached earnings data for {symbol}")
                return self.data_cache[cache_key]

            # Get earnings calendar from Yahoo Finance
            ticker = yf.Ticker(symbol)
            earnings = ticker.earnings

            if earnings is None or earnings.empty:
                logger.warning(f"No earnings data found for {symbol}")
                return []

            # Convert to EarningsData objects
            earnings_data = []
            for date, row in earnings.iterrows():
                try:
                    actual_eps = row.get("Earnings", 0)
                    estimated_eps = row.get("Estimated_Earnings", actual_eps)
                    surprise = actual_eps - estimated_eps if estimated_eps != 0 else 0
                    surprise_percent = (
                        (surprise / estimated_eps * 100) if estimated_eps != 0 else 0
                    )

                    earnings_obj = EarningsData(
                        symbol=symbol,
                        date=date,
                        actual_eps=actual_eps,
                        estimated_eps=estimated_eps,
                        surprise=surprise,
                        surprise_percent=surprise_percent,
                        revenue=row.get("Revenue", 0),
                        revenue_estimate=row.get("Estimated_Revenue", 0),
                        revenue_surprise=row.get("Revenue_Surprise", 0),
                        metadata={"source": "yahoo_finance"},
                    )

                    earnings_data.append(earnings_obj)

                except Exception as e:
                    logger.warning(
                        f"Error parsing earnings data for {symbol} on {date}: {e}"
                    )
                    continue

            # Filter by lookback period
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            earnings_data = [e for e in earnings_data if e.date >= cutoff_date]

            # Cache data
            self._cache_data(cache_key, earnings_data)

            # Log fetch latency
            fetch_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Earnings data fetch completed in {fetch_time:.2f}s for {symbol}: {len(earnings_data)} records"
            )
            return earnings_data

        except Exception as e:
            logger.error(f"Error getting earnings data for {symbol}: {e}")
            return []

    def get_macro_indicators(self) -> List[MacroIndicator]:
        """Get current macro indicators."""
        try:
            indicators = []

            # Get key indicators from FRED
            key_series = ["UNRATE", "CPIAUCSL", "FEDFUNDS", "DGS10", "VIXCLS"]

            for series_id in key_series:
                try:
                    df = self.get_fred_data(series_id)
                    if not df.empty:
                        series_info = self.fred_series[series_id]

                        indicator = MacroIndicator(
                            name=series_info["name"],
                            value=df["value"].iloc[-1],
                            unit=series_info["unit"],
                            frequency=series_info["frequency"],
                            last_updated=df.index[-1],
                            source=DataSource.FRED,
                            metadata={
                                "series_id": series_id,
                                "description": series_info["description"],
                            },
                        )

                        indicators.append(indicator)

                except Exception as e:
                    logger.warning(f"Error getting indicator {series_id}: {e}")
                    continue

            logger.info(f"Retrieved {len(indicators)} macro indicators")
            return indicators

        except Exception as e:
            logger.error(f"Error getting macro indicators: {e}")
            return []

    def analyze_macro_environment(self) -> Dict[str, Any]:
        """Analyze current macro environment."""
        try:
            analysis = {}

            # Get yield curve data
            yield_curve = self.get_yield_curve_data()
            analysis["yield_curve"] = {
                "inverted": yield_curve.spread_10y_2y < 0,
                "spread_10y_2y": yield_curve.spread_10y_2y,
                "spread_10y_3m": yield_curve.spread_10y_3m,
                "curve_slope": yield_curve.curve_slope,
            }

            # Get macro indicators
            indicators = self.get_macro_indicators()
            indicator_dict = {ind.name: ind.value for ind in indicators}

            analysis["indicators"] = indicator_dict

            # Economic regime classification
            analysis["economic_regime"] = self._classify_economic_regime(analysis)

            # Market stress indicators
            analysis["stress_indicators"] = self._calculate_stress_indicators(analysis)

            # Investment implications
            analysis["implications"] = self._generate_investment_implications(analysis)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing macro environment: {e}")
            return {"error": str(e)}

    def _classify_economic_regime(self, analysis: Dict[str, Any]) -> str:
        """Classify current economic regime."""
        try:
            yield_curve = analysis.get("yield_curve", {})
            indicators = analysis.get("indicators", {})

            # Check for recession indicators
            inverted_yield = yield_curve.get("inverted", False)
            unemployment = indicators.get("Unemployment Rate", 4.0)
            fed_funds = indicators.get("Federal Funds Rate", 5.0)

            if inverted_yield and unemployment > 4.5:
                return "recession_risk"
            elif inverted_yield:
                return "growth_slowdown"
            elif fed_funds > 5.0:
                return "tight_monetary"
            elif unemployment < 4.0:
                return "strong_growth"
            else:
                return "moderate_growth"

        except Exception as e:
            logger.error(f"Error classifying economic regime: {e}")
            return "unknown"

    def _calculate_stress_indicators(
        self, analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate market stress indicators."""
        try:
            indicators = analysis.get("indicators", {})

            stress_indicators = {}

            # VIX-based stress
            vix = indicators.get("VIX Volatility Index", 20)
            stress_indicators["vix_stress"] = min(1.0, vix / 30)  # Normalize to 0-1

            # Yield curve stress
            yield_curve = analysis.get("yield_curve", {})
            spread_10y_2y = yield_curve.get("spread_10y_2y", 0)
            stress_indicators["curve_stress"] = max(
                0, -spread_10y_2y / 2
            )  # Inversion stress

            # Overall stress index
            stress_indicators["overall_stress"] = (
                stress_indicators.get("vix_stress", 0) * 0.6
                + stress_indicators.get("curve_stress", 0) * 0.4
            )

            return stress_indicators

        except Exception as e:
            logger.error(f"Error calculating stress indicators: {e}")
            return {"overall_stress": 0.5}

    def _generate_investment_implications(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate investment implications from macro analysis."""
        try:
            implications = []

            regime = analysis.get("economic_regime", "unknown")
            stress = analysis.get("stress_indicators", {}).get("overall_stress", 0.5)

            if regime == "recession_risk":
                implications.extend(
                    [
                        "Consider defensive positioning",
                        "Increase cash allocation",
                        "Focus on quality stocks",
                        "Consider bond allocation",
                    ]
                )
            elif regime == "growth_slowdown":
                implications.extend(
                    [
                        "Reduce risk exposure",
                        "Focus on defensive sectors",
                        "Consider dividend stocks",
                    ]
                )
            elif regime == "tight_monetary":
                implications.extend(
                    [
                        "Expect higher volatility",
                        "Consider value over growth",
                        "Monitor credit conditions",
                    ]
                )
            elif regime == "strong_growth":
                implications.extend(
                    [
                        "Favorable for equities",
                        "Consider cyclical sectors",
                        "Monitor inflation risks",
                    ]
                )

            if stress > 0.7:
                implications.append("High market stress - consider risk reduction")
            elif stress < 0.3:
                implications.append("Low market stress - favorable for risk assets")

            return implications

        except Exception as e:
            logger.error(f"Error generating implications: {e}")
            return ["Monitor macro conditions closely"]

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        try:
            if cache_key not in self.data_cache:
                return False

            if cache_key not in self.cache_expiry:
                return False

            return datetime.now() < self.cache_expiry[cache_key]

        except Exception as e:
            logger.error(f"Error checking cache validity: {e}")
            return False

    def _cache_data(self, cache_key: str, data: Any, expiry_hours: int = 24):
        """Cache data with expiration."""
        try:
            self.data_cache[cache_key] = data
            self.cache_expiry[cache_key] = datetime.now() + timedelta(
                hours=expiry_hours
            )

        except Exception as e:
            logger.error(f"Error caching data: {e}")

    def export_macro_data(self, filepath: str = "logs/macro_data_export.json"):
        """Export macro data analysis to file."""
        try:
            # Get current analysis
            analysis = self.analyze_macro_environment()

            # Get yield curve data
            yield_curve = self.get_yield_curve_data()

            # Get macro indicators
            indicators = self.get_macro_indicators()

            export_data = {
                "analysis": analysis,
                "yield_curve": {
                    "date": yield_curve.date.isoformat(),
                    "rates": yield_curve.rates,
                    "spread_10y_2y": yield_curve.spread_10y_2y,
                    "spread_10y_3m": yield_curve.spread_10y_3m,
                    "curve_slope": yield_curve.curve_slope,
                },
                "indicators": [
                    {
                        "name": ind.name,
                        "value": ind.value,
                        "unit": ind.unit,
                        "last_updated": ind.last_updated.isoformat(),
                        "source": ind.source.value,
                    }
                    for ind in indicators
                ],
                "export_date": datetime.now().isoformat(),
            }

            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Macro data exported to {filepath}")

        except Exception as e:
            logger.error(f"Error exporting macro data: {e}")


# Global macro data integration instance
macro_data_integration = MacroDataIntegration()


def get_macro_data_integration() -> MacroDataIntegration:
    """Get the global macro data integration instance."""
    return macro_data_integration


# Backward compatibility aliases
MacroDataIntegrator = MacroDataIntegration


class EconomicIndicatorLoader:
    """Economic indicator loader for backward compatibility.

    This is a compatibility wrapper around MacroDataIntegration for backward compatibility.
    """

    def __init__(self, **kwargs):
        """Initialize economic indicator loader.

        Args:
            **kwargs: Arguments passed to MacroDataIntegration
        """
        self.macro_integration = MacroDataIntegration(**kwargs)

    def get_indicators(self, **kwargs):
        """Get economic indicators.

        Args:
            **kwargs: Arguments passed to get_macro_indicators

        Returns:
            List of macro indicators
        """
        return self.macro_integration.get_macro_indicators(**kwargs)

    def analyze_environment(self, **kwargs):
        """Analyze macro environment.

        Args:
            **kwargs: Arguments passed to analyze_macro_environment

        Returns:
            Macro environment analysis
        """
        return self.macro_integration.analyze_macro_environment(**kwargs)
