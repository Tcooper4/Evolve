"""Live Data Feed with Fallback Pipeline for Evolve Trading Platform.

This module provides a robust data feed with automatic failover between
multiple data providers: Polygon → Finnhub → Alpha Vantage.
"""

import logging
import os
import warnings
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class DataProvider:
    """Base class for data providers."""

    def __init__(self, name: str, api_key: str = None):
        """Initialize data provider."""
        self.name = name
        self.api_key = api_key
        self.is_available = True
        self.last_check = None
        self.error_count = 0
        self.max_errors = 5
        self.last_successful_request = None

    def check_availability(self) -> bool:
        """Check if provider is available."""
        try:
            # Simple health check
            result = self._health_check()
            if result:
                self.last_check = datetime.now()
                self.last_successful_request = datetime.now()
            return result
        except Exception as e:
            logger.error(f"Health check failed for {self.name}: {e}")
            self.error_count += 1
            if self.error_count >= self.max_errors:
                self.is_available = False
            return False

    def _health_check(self) -> bool:
        """Implement health check in subclasses (stub)."""
        import logging

        logging.getLogger(__name__).warning(
            "_health_check() not implemented for base DataProvider; override in subclass."
        )
        return False

    def get_historical_data(
        self, symbol: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical data from provider (stub)."""
        import logging

        logging.getLogger(__name__).warning(
            "get_historical_data() not implemented for base DataProvider; override in subclass."
        )
        return None

    def get_live_data(self, symbol: str) -> Optional[Dict]:
        """Get live data from provider (stub)."""
        import logging

        logging.getLogger(__name__).warning(
            "get_live_data() not implemented for base DataProvider; override in subclass."
        )
        return None

    def get_provider_status(self) -> Dict[str, Any]:
        """Get provider status information."""
        return {
            "name": self.name,
            "available": self.is_available,
            "error_count": self.error_count,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_successful_request": self.last_successful_request.isoformat()
            if self.last_successful_request
            else None,
        }


class PolygonProvider(DataProvider):
    """Polygon.io data provider."""

    def __init__(self, api_key: str = None):
        """Initialize Polygon provider."""
        super().__init__("Polygon", api_key or os.getenv("POLYGON_API_KEY"))
        self.base_url = "https://api.polygon.io"

    def _health_check(self) -> bool:
        """Check Polygon API health."""
        try:
            url = f"{self.base_url}/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-01"
            params = {"apiKey": self.api_key}
            response = requests.get(url, params=params, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def get_historical_data(
        self, symbol: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical data from Polygon."""
        try:
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/{interval}/{start_date}/{end_date}"
            params = {"apiKey": self.api_key}

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            if data["status"] != "OK" or not data["results"]:
                return None

            df = pd.DataFrame(data["results"])
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
            df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume", "vw": "VWAP"})

            self.last_successful_request = datetime.now()
            return df[["timestamp", "Open", "High", "Low", "Close", "Volume", "VWAP"]]

        except Exception as e:
            logger.error(f"Polygon historical data error: {e}")
            return None

    def get_live_data(self, symbol: str) -> Optional[Dict]:
        """Get live data from Polygon."""
        try:
            url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
            params = {"apiKey": self.api_key}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if "results" not in data:
                return None

            result = data["results"]
            live_data = {
                "symbol": symbol,
                "price": result.get("lastTrade", {}).get("p", 0),
                "volume": result.get("lastTrade", {}).get("s", 0),
                "timestamp": datetime.now().isoformat(),
            }

            self.last_successful_request = datetime.now()
            return live_data

        except Exception as e:
            logger.error(f"Polygon live data error: {e}")
            return None


class FinnhubProvider(DataProvider):
    """Finnhub data provider."""

    def __init__(self, api_key: str = None):
        """Initialize Finnhub provider."""
        super().__init__("Finnhub", api_key or os.getenv("FINNHUB_API_KEY"))
        self.base_url = "https://finnhub.io/api/v1"

    def _health_check(self) -> bool:
        """Check Finnhub API health."""
        try:
            url = f"{self.base_url}/quote"
            params = {"symbol": "AAPL", "token": self.api_key}
            response = requests.get(url, params=params, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def get_historical_data(
        self, symbol: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical data from Finnhub."""
        try:
            start_ts = int(pd.to_datetime(start_date).timestamp())
            end_ts = int(pd.to_datetime(end_date).timestamp())

            url = f"{self.base_url}/stock/candle"
            params = {"symbol": symbol, "resolution": "D", "from": start_ts, "to": end_ts, "token": self.api_key}

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            if data["s"] != "ok" or not data["t"]:
                return None

            df = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(data["t"], unit="s"),
                    "Open": data["o"],
                    "High": data["h"],
                    "Low": data["l"],
                    "Close": data["c"],
                    "Volume": data["v"],
                }
            )

            self.last_successful_request = datetime.now()
            return df

        except Exception as e:
            logger.error(f"Finnhub historical data error: {e}")
            return None

    def get_live_data(self, symbol: str) -> Optional[Dict]:
        """Get live data from Finnhub."""
        try:
            url = f"{self.base_url}/quote"
            params = {"symbol": symbol, "token": self.api_key}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            live_data = {
                "symbol": symbol,
                "price": data.get("c", 0),
                "volume": data.get("v", 0),
                "timestamp": datetime.now().isoformat(),
            }

            self.last_successful_request = datetime.now()
            return live_data

        except Exception as e:
            logger.error(f"Finnhub live data error: {e}")
            return None


class AlphaVantageProvider(DataProvider):
    """Alpha Vantage data provider."""

    def __init__(self, api_key: str = None):
        """Initialize Alpha Vantage provider."""
        super().__init__("Alpha Vantage", api_key or os.getenv("ALPHA_VANTAGE_API_KEY"))
        self.base_url = "https://www.alphavantage.co/query"

    def _health_check(self) -> bool:
        """Check Alpha Vantage API health."""
        try:
            params = {"function": "TIME_SERIES_DAILY", "symbol": "AAPL", "apikey": self.api_key}
            response = requests.get(self.base_url, params=params, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def get_historical_data(
        self, symbol: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical data from Alpha Vantage."""
        try:
            params = {"function": "TIME_SERIES_DAILY", "symbol": symbol, "apikey": self.api_key, "outputsize": "full"}

            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            if "Time Series (Daily)" not in data:
                return None

            time_series = data["Time Series (Daily)"]
            records = []

            for date, values in time_series.items():
                if start_date <= date <= end_date:
                    records.append(
                        {
                            "timestamp": pd.to_datetime(date),
                            "Open": float(values["1. open"]),
                            "High": float(values["2. high"]),
                            "Low": float(values["3. low"]),
                            "Close": float(values["4. close"]),
                            "Volume": int(values["5. volume"]),
                        }
                    )

            if not records:
                return None

            df = pd.DataFrame(records)
            df = df.sort_values("timestamp")

            self.last_successful_request = datetime.now()
            return df

        except Exception as e:
            logger.error(f"Alpha Vantage historical data error: {e}")
            return None

    def get_live_data(self, symbol: str) -> Optional[Dict]:
        """Get live data from Alpha Vantage."""
        try:
            params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": self.api_key}

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if "Global Quote" not in data:
                return None

            quote = data["Global Quote"]
            live_data = {
                "symbol": symbol,
                "price": float(quote.get("05. price", 0)),
                "volume": int(quote.get("06. volume", 0)),
                "timestamp": datetime.now().isoformat(),
            }

            self.last_successful_request = datetime.now()
            return live_data

        except Exception as e:
            logger.error(f"Alpha Vantage live data error: {e}")
            return None


class LiveDataFeed:
    """Main data feed with automatic failover."""

    def __init__(self):
        """Initialize the live data feed."""
        self.providers = [PolygonProvider(), FinnhubProvider(), AlphaVantageProvider()]
        self.current_provider_index = 0
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

    def _get_current_provider(self) -> DataProvider:
        """Get the current active provider."""
        return self.providers[self.current_provider_index]

    def _switch_provider(self) -> bool:
        """Switch to the next available provider."""
        original_index = self.current_provider_index

        for i in range(len(self.providers)):
            self.current_provider_index = (self.current_provider_index + 1) % len(self.providers)
            provider = self.providers[self.current_provider_index]

            if provider.check_availability():
                if self.current_provider_index != original_index:
                    logger.info(f"Switched to provider: {provider.name}")
                return True

        logger.error("No providers available")
        return False

    def _get_cache_key(self, symbol: str, start_date: str, end_date: str, interval: str) -> str:
        """Generate cache key for data request."""
        return f"{symbol}_{start_date}_{end_date}_{interval}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False

        cache_time, _ = self.cache[cache_key]
        return (datetime.now() - cache_time).seconds < self.cache_ttl

    def get_historical_data(
        self, symbol: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical data with automatic failover."""
        cache_key = self._get_cache_key(symbol, start_date, end_date, interval)

        # Check cache first
        if self._is_cache_valid(cache_key):
            logger.info(f"Returning cached data for {symbol}")
            return self.cache[cache_key][1]

        # Try current provider
        provider = self._get_current_provider()
        if provider.check_availability():
            data = provider.get_historical_data(symbol, start_date, end_date, interval)
            if data is not None:
                # Cache the result
                self.cache[cache_key] = (datetime.now(), data)
                return data

        # Try other providers
        for _ in range(len(self.providers) - 1):
            if self._switch_provider():
                provider = self._get_current_provider()
                data = provider.get_historical_data(symbol, start_date, end_date, interval)
                if data is not None:
                    self.cache[cache_key] = (datetime.now(), data)
                    return data

        # Generate fallback data
        logger.warning(f"All providers failed for {symbol}, generating fallback data")
        fallback_data = self._get_fallback_historical_data(symbol, start_date, end_date)
        self.cache[cache_key] = (datetime.now(), fallback_data)
        return fallback_data

    def get_live_data(self, symbol: str) -> Optional[Dict]:
        """Get live data with automatic failover."""
        # Try current provider
        provider = self._get_current_provider()
        if provider.check_availability():
            data = provider.get_live_data(symbol)
            if data is not None:
                return data

        # Try other providers
        for _ in range(len(self.providers) - 1):
            if self._switch_provider():
                provider = self._get_current_provider()
                data = provider.get_live_data(symbol)
                if data is not None:
                    return data

        # Generate fallback data
        logger.warning(f"All providers failed for {symbol}, generating fallback data")
        return self._get_fallback_live_data(symbol)

    def _get_fallback_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate fallback historical data."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq="D")

        # Generate realistic price data
        base_price = 100.0
        prices = []
        for i, date in enumerate(dates):
            # Add some randomness and trend
            change = np.random.normal(0, 0.02) + 0.001 * i  # Small upward trend
            base_price *= 1 + change

            # Generate OHLC data
            daily_volatility = np.random.normal(0, 0.01)
            open_price = base_price * (1 + daily_volatility)
            high_price = open_price * (1 + abs(np.random.normal(0, 0.005)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.005)))
            close_price = base_price

            prices.append(
                {
                    "timestamp": date,
                    "Open": round(open_price, 2),
                    "High": round(high_price, 2),
                    "Low": round(low_price, 2),
                    "Close": round(close_price, 2),
                    "Volume": int(np.random.normal(1000000, 200000)),
                }
            )

        return pd.DataFrame(prices)

    def _get_fallback_live_data(self, symbol: str) -> Dict:
        """Generate fallback live data."""
        return {
            "symbol": symbol,
            "price": round(100.0 + np.random.normal(0, 2), 2),
            "volume": int(np.random.normal(1000000, 200000)),
            "timestamp": datetime.now().isoformat(),
            "source": "fallback",
        }

    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers."""
        status = {"current_provider": self._get_current_provider().name, "providers": {}}

        for provider in self.providers:
            status["providers"][provider.name] = provider.get_provider_status()

        return status

    def reset_providers(self) -> None:
        """Reset all providers."""
        for provider in self.providers:
            provider.is_available = True
            provider.error_count = 0
            provider.last_check = None
        self.current_provider_index = 0
        logger.info("All providers reset")

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self.cache.clear()
        logger.info("Data cache cleared")

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        provider_status = self.get_provider_status()
        available_providers = sum(1 for status in provider_status["providers"].values() if status["available"])

        return {
            "status": "healthy" if available_providers > 0 else "degraded",
            "available_providers": available_providers,
            "total_providers": len(self.providers),
            "current_provider": provider_status["current_provider"],
            "cache_size": len(self.cache),
            "provider_status": provider_status["providers"],
        }


def get_data_feed() -> LiveDataFeed:
    """Get a singleton instance of the data feed."""
    if not hasattr(get_data_feed, "_instance"):
        get_data_feed._instance = LiveDataFeed()
    return get_data_feed._instance
