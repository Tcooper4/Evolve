import logging
import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature generation"""

    name: str
    description: str
    category: str
    dependencies: List[str]
    parameters: Dict
    validation_rules: Dict
    is_required: bool = False
    is_custom: bool = False


class FeatureVerificationError(Exception):
    """Custom exception for feature verification errors"""



def verify_feature(func: Callable) -> Callable:
    """Decorator to verify feature generation"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            # Verify result
            if not isinstance(result, pd.Series):
                raise FeatureVerificationError(f"Feature {func.__name__} must return a pandas Series")
            if result.isnull().all():
                raise FeatureVerificationError(f"Feature {func.__name__} returned all null values")
            # Log performance
            execution_time = time.time() - start_time
            logger.info(f"Feature {func.__name__} generated in {execution_time:.2f} seconds")
            return {
                "success": True,
                "result": {
                    "success": True,
                    "result": result,
                    "message": "Operation completed successfully",
                    "timestamp": datetime.now().isoformat(),
                },
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error generating feature {func.__name__}: {str(e)}")
            raise

    return wrapper


class FeatureGenerator:
    """Enhanced feature generator with verification and caching"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.feature_cache = {}
        self.feature_configs = {}
        self._setup_logging()

        return {"success": True, "message": "Initialization completed", "timestamp": datetime.now().isoformat()}

    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        return {"success": True, "message": "Initialization completed", "timestamp": datetime.now().isoformat()}

    def register_feature(self, config: FeatureConfig):
        """Register a new feature configuration"""
        self.feature_configs[config.name] = config
        self.logger.info(f"Registered feature: {config.name}")

    @verify_feature
    def generate_feature(self, data: pd.DataFrame, feature_name: str, **kwargs) -> pd.Series:
        """Generate a feature with verification"""
        if feature_name in self.feature_cache:
            return self.feature_cache[feature_name]

        if feature_name not in self.feature_configs:
            raise ValueError(f"Feature {feature_name} not registered")

        config = self.feature_configs[feature_name]
        try:
            # Check dependencies
            for dep in config.dependencies:
                if dep not in data.columns:
                    raise FeatureVerificationError(f"Missing dependency: {dep}")

            # Generate feature
            feature = self._generate_feature_impl(data, feature_name, **kwargs)

            # Validate feature
            self._validate_feature(feature, config.validation_rules)

            # Cache result
            self.feature_cache[feature_name] = feature
            return feature

        except Exception as e:
            self.logger.error(f"Error generating feature {feature_name}: {str(e)}")
            raise

    def _generate_feature_impl(self, data: pd.DataFrame, feature_name: str, **kwargs) -> pd.Series:
        """Implementation of feature generation"""
        config = self.feature_configs[feature_name]

        if config.is_custom:
            return {
                "success": True,
                "result": self._generate_custom_feature(data, feature_name, **kwargs),
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
            }

        # Map feature names to generation functions
        feature_map = {
            "returns": self._calculate_returns,
            "volatility": self._calculate_volatility,
            "momentum": self._calculate_momentum,
            "rsi": self._calculate_rsi,
            "macd": self._calculate_macd,
            "bollinger_bands": self._calculate_bollinger_bands,
            "volume_profile": self._calculate_volume_profile,
            "price_channels": self._calculate_price_channels,
            "support_resistance": self._calculate_support_resistance,
            "trend_strength": self._calculate_trend_strength,
            "market_regime": self._calculate_market_regime,
            "volatility_regime": self._calculate_volatility_regime,
            "correlation": self._calculate_correlation,
            "beta": self._calculate_beta,
            "alpha": self._calculate_alpha,
            "sharpe_ratio": self._calculate_sharpe_ratio,
            "sortino_ratio": self._calculate_sortino_ratio,
            "max_drawdown": self._calculate_max_drawdown,
            "var": self._calculate_var,
            "cvar": self._calculate_cvar,
            "position_size": self._calculate_position_size,
            "risk_parity": self._calculate_risk_parity,
            "mean_variance": self._calculate_mean_variance,
            "black_litterman": self._calculate_black_litterman,
            "risk_metrics": self._calculate_risk_metrics,
            "performance_metrics": self._calculate_performance_metrics,
            "execution_metrics": self._calculate_execution_metrics,
            "signal_metrics": self._calculate_signal_metrics,
            "portfolio_metrics": self._calculate_portfolio_metrics,
            "regime_metrics": self._calculate_regime_metrics,
        }

        if feature_name not in feature_map:
            raise ValueError(f"Unknown feature: {feature_name}")

        return feature_map[feature_name](data, **kwargs)

    def _validate_feature(self, feature: pd.Series, rules: Dict):
        """Validate generated feature against rules"""
        for rule_name, rule_func in rules.items():
            if not rule_func(feature):
                raise FeatureVerificationError(f"Feature failed validation rule: {rule_name}")

    def _generate_custom_feature(self, data: pd.DataFrame, feature_name: str, **kwargs) -> pd.Series:
        """Generate custom feature"""
        if "custom_func" not in kwargs:
            raise ValueError("Custom function not provided")
        return kwargs["custom_func"](data)

    # Feature generation implementations
    def _calculate_returns(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate returns"""
        return data["close"].pct_change()

    def _calculate_volatility(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate volatility"""
        window = kwargs.get("window", 20)
        return data["returns"].rolling(window=window).std()

    def _calculate_momentum(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate momentum"""
        window = kwargs.get("window", 20)
        return data["close"].pct_change(periods=window)

    def _calculate_rsi(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate RSI"""
        window = kwargs.get("window", 14)
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate MACD"""
        fast = kwargs.get("fast", 12)
        slow = kwargs.get("slow", 26)
        signal = kwargs.get("signal", 9)

        exp1 = data["close"].ewm(span=fast, adjust=False).mean()
        exp2 = data["close"].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    def _calculate_bollinger_bands(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate Bollinger Bands"""
        window = kwargs.get("window", 20)
        num_std = kwargs.get("num_std", 2)

        middle_band = data["close"].rolling(window=window).mean()
        std = data["close"].rolling(window=window).std()
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)

        return (data["close"] - lower_band) / (upper_band - lower_band)

    def _calculate_volume_profile(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate volume profile"""
        window = kwargs.get("window", 20)
        return data["volume"].rolling(window=window).mean() / data["volume"].rolling(window=window).std()

    def _calculate_price_channels(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate price channels"""
        window = kwargs.get("window", 20)
        high_channel = data["high"].rolling(window=window).max()
        low_channel = data["low"].rolling(window=window).min()
        return (data["close"] - low_channel) / (high_channel - low_channel)

    def _calculate_support_resistance(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate support/resistance levels"""
        window = kwargs.get("window", 20)
        return (data["close"] - data["low"].rolling(window=window).min()) / (
            data["high"].rolling(window=window).max() - data["low"].rolling(window=window).min()
        )

    def _calculate_trend_strength(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate trend strength"""
        window = kwargs.get("window", 20)
        returns = data["close"].pct_change()
        return returns.rolling(window=window).mean() / returns.rolling(window=window).std()

    def _calculate_market_regime(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate market regime"""
        window = kwargs.get("window", 20)
        returns = data["close"].pct_change()
        volatility = returns.rolling(window=window).std()
        trend = returns.rolling(window=window).mean()

        # Combine signals
        regime = pd.Series(index=data.index, dtype=float)
        regime[trend > 0.001] = 1  # Bullish
        regime[trend < -0.001] = -1  # Bearish
        regime[volatility > volatility.quantile(0.8)] = 0  # High volatility
        return regime

    def _calculate_volatility_regime(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate volatility regime"""
        window = kwargs.get("window", 20)
        returns = data["close"].pct_change()
        volatility = returns.rolling(window=window).std()

        # Define regimes
        regime = pd.Series(index=data.index, dtype=float)
        regime[volatility > volatility.quantile(0.8)] = 1  # High volatility
        regime[volatility < volatility.quantile(0.2)] = -1  # Low volatility
        return regime

    def _calculate_correlation(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate correlation with market"""
        window = kwargs.get("window", 20)
        market_returns = kwargs.get("market_returns", data["close"].pct_change())
        returns = data["close"].pct_change()
        return returns.rolling(window=window).corr(market_returns)

    def _calculate_beta(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate beta"""
        window = kwargs.get("window", 20)
        market_returns = kwargs.get("market_returns", data["close"].pct_change())
        returns = data["close"].pct_change()

        # Calculate rolling covariance and variance
        covar = returns.rolling(window=window).cov(market_returns)
        market_var = market_returns.rolling(window=window).var()

        return covar / market_var

    def _calculate_alpha(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate alpha"""
        window = kwargs.get("window", 20)
        market_returns = kwargs.get("market_returns", data["close"].pct_change())
        returns = data["close"].pct_change()
        risk_free_rate = kwargs.get("risk_free_rate", 0.0)

        beta = self._calculate_beta(data, market_returns=market_returns, window=window)
        return returns.rolling(window=window).mean() - (
            risk_free_rate + beta * (market_returns.rolling(window=window).mean() - risk_free_rate)
        )

    def _calculate_sharpe_ratio(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate Sharpe ratio"""
        window = kwargs.get("window", 20)
        returns = data["close"].pct_change()
        risk_free_rate = kwargs.get("risk_free_rate", 0.0)

        excess_returns = returns - risk_free_rate
        return excess_returns.rolling(window=window).mean() / returns.rolling(window=window).std()

    def _calculate_sortino_ratio(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate Sortino ratio"""
        window = kwargs.get("window", 20)
        returns = data["close"].pct_change()
        risk_free_rate = kwargs.get("risk_free_rate", 0.0)

        excess_returns = returns - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.rolling(window=window).std()

        return excess_returns.rolling(window=window).mean() / downside_std

    def _calculate_max_drawdown(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate maximum drawdown"""
        window = kwargs.get("window", 20)
        returns = data["close"].pct_change()
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.rolling(window=window).max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.rolling(window=window).min()

    def _calculate_var(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate Value at Risk"""
        window = kwargs.get("window", 20)
        confidence_level = kwargs.get("confidence_level", 0.95)
        returns = data["close"].pct_change()

        return returns.rolling(window=window).quantile(1 - confidence_level)

    def _calculate_cvar(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate Conditional Value at Risk"""
        window = kwargs.get("window", 20)
        confidence_level = kwargs.get("confidence_level", 0.95)
        returns = data["close"].pct_change()

        var = self._calculate_var(data, window=window, confidence_level=confidence_level)
        return returns[returns <= var].rolling(window=window).mean()

    def _calculate_position_size(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate position size based on risk"""
        window = kwargs.get("window", 20)
        risk_per_trade = kwargs.get("risk_per_trade", 0.02)
        account_size = kwargs.get("account_size", 100000)

        volatility = self._calculate_volatility(data, window=window)
        position_size = (account_size * risk_per_trade) / (volatility * data["close"])
        return position_size

    def _calculate_risk_parity(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate risk parity weights"""
        window = kwargs.get("window", 20)
        returns = data["close"].pct_change()
        volatility = returns.rolling(window=window).std()
        return 1 / volatility

    def _calculate_mean_variance(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate mean-variance optimal weights"""
        window = kwargs.get("window", 20)
        returns = data["close"].pct_change()
        mean_returns = returns.rolling(window=window).mean()
        volatility = returns.rolling(window=window).std()
        return mean_returns / (volatility**2)

    def _calculate_black_litterman(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate Black-Litterman model weights"""
        window = kwargs.get("window", 20)
        market_cap = kwargs.get("market_cap", None)
        risk_aversion = kwargs.get("risk_aversion", 2.5)
        tau = kwargs.get("tau", 0.05)

        if market_cap is None:
            market_cap = data["close"] * data["volume"]

        # Calculate market equilibrium returns
        returns = data["close"].pct_change()
        covariance = returns.rolling(window=window).cov()
        market_weights = market_cap / market_cap.sum()
        pi = risk_aversion * covariance.dot(market_weights)

        # Combine with views
        views = kwargs.get("views", None)
        if views is not None:
            omega = kwargs.get("omega", np.diag(np.diag(covariance)) * tau)
            bl_returns = np.linalg.inv(
                np.linalg.inv(tau * covariance) + views.T.dot(np.linalg.inv(omega)).dot(views)
            ).dot(np.linalg.inv(tau * covariance).dot(pi) + views.T.dot(np.linalg.inv(omega)).dot(views))
            return {
                "success": True,
                "result": pd.Series(bl_returns, index=data.index),
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
            }

        return pd.Series(pi, index=data.index)

    def _calculate_risk_metrics(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate comprehensive risk metrics"""
        window = kwargs.get("window", 20)
        returns = data["close"].pct_change()

        # Calculate various risk metrics
        volatility = returns.rolling(window=window).std()
        var = self._calculate_var(data, window=window)
        cvar = self._calculate_cvar(data, window=window)
        max_dd = self._calculate_max_drawdown(data, window=window)

        # Combine metrics
        risk_score = (volatility + abs(var) + abs(cvar) + abs(max_dd)) / 4
        return risk_score

    def _calculate_performance_metrics(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate comprehensive performance metrics"""
        window = kwargs.get("window", 20)
        returns = data["close"].pct_change()

        # Calculate various performance metrics
        sharpe = self._calculate_sharpe_ratio(data, window=window)
        sortino = self._calculate_sortino_ratio(data, window=window)
        alpha = self._calculate_alpha(data, window=window)

        # Combine metrics
        performance_score = (sharpe + sortino + alpha) / 3
        return performance_score

    def _calculate_execution_metrics(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate execution metrics"""
        window = kwargs.get("window", 20)

        # Calculate various execution metrics
        spread = (data["high"] - data["low"]) / data["close"]
        volume_impact = data["volume"] / data["volume"].rolling(window=window).mean()
        price_impact = abs(data["close"].pct_change()) / data["volume"]

        # Combine metrics
        execution_score = (spread + volume_impact + price_impact) / 3
        return execution_score

    def _calculate_signal_metrics(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate signal quality metrics"""
        window = kwargs.get("window", 20)
        returns = data["close"].pct_change()

        # Calculate various signal metrics
        rsi = self._calculate_rsi(data, window=window)
        macd = self._calculate_macd(data, window=window)
        momentum = self._calculate_momentum(data, window=window)

        # Combine metrics
        signal_score = (rsi + macd + momentum) / 3
        return signal_score

    def _calculate_portfolio_metrics(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate portfolio metrics"""
        window = kwargs.get("window", 20)
        returns = data["close"].pct_change()

        # Calculate various portfolio metrics
        sharpe = self._calculate_sharpe_ratio(data, window=window)
        sortino = self._calculate_sortino_ratio(data, window=window)
        max_dd = self._calculate_max_drawdown(data, window=window)

        # Combine metrics
        portfolio_score = (sharpe + sortino + (1 + max_dd)) / 3
        return portfolio_score

    def _calculate_regime_metrics(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate regime-specific metrics"""
        window = kwargs.get("window", 20)

        # Calculate various regime metrics
        market_regime = self._calculate_market_regime(data, window=window)
        volatility_regime = self._calculate_volatility_regime(data, window=window)
        trend_strength = self._calculate_trend_strength(data, window=window)

        # Combine metrics
        regime_score = (market_regime + volatility_regime + trend_strength) / 3
        return regime_score
