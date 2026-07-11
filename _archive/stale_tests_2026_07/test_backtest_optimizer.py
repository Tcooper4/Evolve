"""
Tests for Backtest Optimizer

Tests the backtest optimizer with walk-forward analysis, regime detection,
and comprehensive backtesting capabilities.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from trading.optimization.backtest_optimizer import (
    BacktestOptimizer,
    RegimeDetector,
    RegimeInfo,
    WalkForwardOptimizer,
    WalkForwardResult,
)


class TestRegimeDetector:
    """Test the regime detector."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="D")

        # Create realistic market data with different regimes
        data = pd.DataFrame(index=dates)

        # First regime: Low volatility bull market
        data.loc[:"2020-06-30", "close"] = 100 + np.cumsum(
            np.random.normal(0.001, 0.01, 181)
        )
        data.loc[:"2020-06-30", "volume"] = np.random.randint(1000000, 5000000, 181)

        # Second regime: High volatility bear market
        data.loc["2020-07-01":"2020-12-31", "close"] = data.loc[
            "2020-06-30", "close"
        ] + np.cumsum(np.random.normal(-0.002, 0.03, 184))
        data.loc["2020-07-01":"2020-12-31", "volume"] = np.random.randint(
            2000000, 8000000, 184
        )

        # Third regime: Moderate volatility sideways
        data.loc["2021-01-01":, "close"] = data.loc["2020-12-31", "close"] + np.cumsum(
            np.random.normal(0.0005, 0.02, 135)
        )
        data.loc["2021-01-01":, "volume"] = np.random.randint(1500000, 6000000, 135)

        # Add OHLC columns
        data["open"] = data["close"] * (1 + np.random.normal(0, 0.005, len(data)))
        data["high"] = data[["open", "close"]].max(axis=1) * (
            1 + np.abs(np.random.normal(0, 0.01, len(data)))
        )
        data["low"] = data[["open", "close"]].min(axis=1) * (
            1 - np.abs(np.random.normal(0, 0.01, len(data)))
        )

        return data

    def test_regime_detector_initialization(self):
        """Test regime detector initialization."""
        detector = RegimeDetector(n_regimes=4, lookback_window=100)

        assert detector.n_regimes == 4
        assert detector.lookback_window == 100
        assert detector.scaler is not None
        assert detector.kmeans is not None
        assert len(detector.regime_history) == 0

    def test_calculate_regime_features(self, sample_data):
        """Test regime feature calculation."""
        detector = RegimeDetector()
        features = detector.calculate_regime_features(sample_data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0

        # Check that expected features are present
        expected_features = [
            "volatility",
            "returns",
            "skewness",
            "kurtosis",
            "trend_strength",
            "momentum",
            "rsi",
            "volume_ratio",
            "volume_volatility",
            "volatility_regime",
        ]

        for feature in expected_features:
            if feature in features.columns:
                assert not features[feature].isna().all()

    def test_rsi_calculation(self, sample_data):
        """Test RSI calculation."""
        detector = RegimeDetector()
        rsi = detector._calculate_rsi(sample_data["close"])

        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_data)
        assert rsi.isna().sum() > 0  # Should have some NaN values at the beginning
        assert not rsi.isna().all()  # Should not be all NaN

        # RSI should be between 0 and 100
        rsi_clean = rsi.dropna()
        assert (rsi_clean >= 0).all()
        assert (rsi_clean <= 100).all()

    def test_regime_detection(self, sample_data):
        """Test regime detection."""
        detector = RegimeDetector(n_regimes=3)
        regimes = detector.detect_regimes(sample_data)

        assert isinstance(regimes, list)
        assert len(regimes) > 0
        assert len(regimes) <= 3  # Should not exceed n_regimes

        # Check regime structure
        for regime in regimes:
            assert isinstance(regime, RegimeInfo)
            assert hasattr(regime, "regime_id")
            assert hasattr(regime, "regime_name")
            assert hasattr(regime, "start_date")
            assert hasattr(regime, "end_date")
            assert hasattr(regime, "characteristics")
            assert hasattr(regime, "volatility")
            assert hasattr(regime, "trend_strength")
            assert hasattr(regime, "duration_days")
            assert hasattr(regime, "confidence")

            # Check confidence is reasonable
            assert 0 <= regime.confidence <= 1

    def test_regime_classification(self, sample_data):
        """Test regime classification logic."""
        detector = RegimeDetector()

        # Test different characteristic combinations
        test_cases = [
            {
                "mean_volatility": 0.3,
                "mean_returns": -0.08,
                "mean_trend_strength": -0.03,
            },
            {"mean_volatility": 0.1, "mean_returns": 0.08, "mean_trend_strength": 0.03},
            {"mean_volatility": 0.2, "mean_returns": 0.02, "mean_trend_strength": 0.01},
        ]

        expected_regimes = [
            "High Volatility Bear Market",
            "Low Volatility Bull Market",
            "Moderate Bull Market",
        ]

        for characteristics, expected_regime in zip(test_cases, expected_regimes):
            regime_name = detector._classify_regime(characteristics)
            assert regime_name == expected_regime

    def test_regime_confidence_calculation(self, sample_data):
        """Test regime confidence calculation."""
        detector = RegimeDetector()
        features = detector.calculate_regime_features(sample_data)

        if len(features) < detector.lookback_window:
            pytest.skip("Insufficient data for confidence calculation")

        features_scaled = detector.scaler.fit_transform(features)
        cluster_labels = detector.kmeans.fit_predict(features_scaled)

        # Test confidence calculation for each regime
        for label in np.unique(cluster_labels):
            confidence = detector._calculate_regime_confidence(
                features_scaled, cluster_labels, label
            )
            assert 0 <= confidence <= 1

    def test_get_current_regime(self, sample_data):
        """Test current regime detection."""
        detector = RegimeDetector()

        # First detect all regimes
        detector.detect_regimes(sample_data)

        # Get current regime
        current_regime = detector.get_current_regime(sample_data)

        # Should return a regime if enough data is available
        if len(sample_data) >= 30:
            assert current_regime is not None
            assert isinstance(current_regime, RegimeInfo)
        else:
            assert current_regime is None

    def test_regime_detection_with_insufficient_data(self):
        """Test regime detection with insufficient data."""
        detector = RegimeDetector()

        # Create small dataset
        small_data = pd.DataFrame(
            {"close": np.random.randn(50), "volume": np.random.randint(1000, 10000, 50)}
        )

        regimes = detector.detect_regimes(small_data)
        assert regimes == []  # Should return empty list for insufficient data


class TestWalkForwardOptimizer:
    """Test the walk-forward optimizer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for walk-forward testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=400, freq="D")
        data = pd.DataFrame(
            {
                "close": 100 + np.cumsum(np.random.normal(0.001, 0.02, 400)),
                "volume": np.random.randint(1000000, 5000000, 400),
            },
            index=dates,
        )
        return data

    @pytest.fixture
    def mock_strategy_class(self):
        """Create mock strategy class."""

        class MockStrategy:
            def __init__(self, params):
                self.params = params

            def generate_signals(self, data):
                return pd.Series(
                    np.random.choice([-1, 0, 1], len(data)), index=data.index
                )

            def evaluate_performance(self, signals, data):
                class MockMetrics:
                    def __init__(self):
                        self.sharpe_ratio = np.random.normal(0.5, 0.3)
                        self.total_return = np.random.normal(0.1, 0.05)
                        self.max_drawdown = np.random.normal(-0.1, 0.05)
                        self.win_rate = np.random.normal(0.55, 0.1)

                return MockMetrics()

        return MockStrategy

    def test_walk_forward_optimizer_initialization(self):
        """Test walk-forward optimizer initialization."""
        optimizer = WalkForwardOptimizer(
            training_window=200, validation_window=50, step_size=25
        )

        assert optimizer.training_window == 200
        assert optimizer.validation_window == 50
        assert optimizer.step_size == 25
        assert optimizer.regime_detector is not None
        assert len(optimizer.results) == 0

    def test_generate_windows(self, sample_data):
        """Test window generation."""
        optimizer = WalkForwardOptimizer(
            training_window=100, validation_window=30, step_size=20
        )

        windows = optimizer._generate_windows(sample_data.index)

        assert isinstance(windows, list)
        assert len(windows) > 0

        for train_start, train_end, val_start, val_end in windows:
            # Check window relationships
            assert train_start < train_end
            assert train_end < val_start
            assert val_start < val_end

            # Check window sizes
            training_days = (train_end - train_start).days + 1
            validation_days = (val_end - val_start).days + 1

            assert training_days == optimizer.training_window
            assert validation_days == optimizer.validation_window

    def test_get_regime_for_period(self, sample_data):
        """Test regime identification for specific period."""
        optimizer = WalkForwardOptimizer()

        # Create some mock regimes
        mock_regimes = [
            RegimeInfo(
                regime_id=0,
                regime_name="Bull Market",
                start_date="2020-01-01",
                end_date="2020-06-30",
                characteristics={},
                volatility=0.15,
                trend_strength=0.02,
                correlation_structure={},
                duration_days=180,
                confidence=0.8,
            ),
            RegimeInfo(
                regime_id=1,
                regime_name="Bear Market",
                start_date="2020-07-01",
                end_date="2020-12-31",
                characteristics={},
                volatility=0.25,
                trend_strength=-0.02,
                correlation_structure={},
                duration_days=180,
                confidence=0.7,
            ),
        ]

        # Test period that overlaps with first regime
        start_date = pd.Timestamp("2020-03-01")
        end_date = pd.Timestamp("2020-04-30")

        regime = optimizer._get_regime_for_period(mock_regimes, start_date, end_date)
        assert regime is not None
        assert regime.regime_name == "Bull Market"

        # Test period that doesn't overlap with any regime
        start_date = pd.Timestamp("2021-01-01")
        end_date = pd.Timestamp("2021-02-28")

        regime = optimizer._get_regime_for_period(mock_regimes, start_date, end_date)
        assert regime is None

    def test_optimize_strategy(self, sample_data, mock_strategy_class):
        """Test strategy optimization."""
        optimizer = WalkForwardOptimizer()
        param_space = {"window": [10, 20, 30], "threshold": [0.01, 0.02, 0.03]}

        best_params = optimizer._optimize_strategy(
            sample_data, mock_strategy_class, param_space, "bayesian"
        )

        assert isinstance(best_params, dict)
        assert "window" in best_params
        assert "threshold" in best_params

    def test_evaluate_strategy(self, sample_data, mock_strategy_class):
        """Test strategy evaluation."""
        optimizer = WalkForwardOptimizer()
        params = {"window": 20, "threshold": 0.02}

        performance = optimizer._evaluate_strategy(
            sample_data, mock_strategy_class, params
        )

        assert isinstance(performance, dict)
        assert "sharpe_ratio" in performance
        assert "total_return" in performance
        assert "max_drawdown" in performance
        assert "win_rate" in performance

    def test_walk_forward_analysis(self, sample_data, mock_strategy_class):
        """Test complete walk-forward analysis."""
        optimizer = WalkForwardOptimizer(
            training_window=100, validation_window=30, step_size=50
        )

        param_space = {"window": [10, 20, 30], "threshold": [0.01, 0.02, 0.03]}

        results = optimizer.run_walk_forward_analysis(
            sample_data, mock_strategy_class, param_space
        )

        assert isinstance(results, list)
        assert len(results) > 0

        for result in results:
            assert isinstance(result, WalkForwardResult)
            assert hasattr(result, "period_start")
            assert hasattr(result, "period_end")
            assert hasattr(result, "training_period")
            assert hasattr(result, "validation_period")
            assert hasattr(result, "best_params")
            assert hasattr(result, "validation_performance")
            assert hasattr(result, "out_of_sample_performance")
            assert hasattr(result, "regime")
            assert hasattr(result, "regime_confidence")
            assert hasattr(result, "timestamp")

    def test_analyze_results(self, sample_data, mock_strategy_class):
        """Test results analysis."""
        optimizer = WalkForwardOptimizer(
            training_window=100, validation_window=30, step_size=50
        )

        # Run analysis first
        param_space = {"window": [10, 20], "threshold": [0.01, 0.02]}
        optimizer.run_walk_forward_analysis(
            sample_data, mock_strategy_class, param_space
        )

        analysis = optimizer.analyze_results()

        assert isinstance(analysis, dict)
        assert "summary" in analysis
        assert "regime_analysis" in analysis
        assert "parameter_stability" in analysis

        # Check summary structure
        summary = analysis["summary"]
        assert "total_periods" in summary
        assert "avg_validation_sharpe" in summary
        assert "avg_oos_sharpe" in summary
        assert "sharpe_degradation" in summary

    def test_parameter_stability_analysis(self, sample_data, mock_strategy_class):
        """Test parameter stability analysis."""
        optimizer = WalkForwardOptimizer(
            training_window=100, validation_window=30, step_size=50
        )

        # Run analysis first
        param_space = {"window": [10, 20], "threshold": [0.01, 0.02]}
        optimizer.run_walk_forward_analysis(
            sample_data, mock_strategy_class, param_space
        )

        stability = optimizer._analyze_parameter_stability()

        assert isinstance(stability, dict)
        if stability:  # If there are parameters to analyze
            for param_name, metrics in stability.items():
                assert "mean" in metrics
                assert "std" in metrics
                assert "cv" in metrics
                assert "min" in metrics
                assert "max" in metrics

    def test_plot_results(self, sample_data, mock_strategy_class):
        """Test results plotting."""
        optimizer = WalkForwardOptimizer(
            training_window=100, validation_window=30, step_size=50
        )

        # Run analysis first
        param_space = {"window": [10, 20], "threshold": [0.01, 0.02]}
        optimizer.run_walk_forward_analysis(
            sample_data, mock_strategy_class, param_space
        )

        # Test plotting (should not raise errors)
        try:
            optimizer.plot_results()
        except Exception as e:
            pytest.fail(f"Plotting failed: {e}")

    def test_export_results(self, sample_data, mock_strategy_class, tmp_path):
        """Test results export."""
        optimizer = WalkForwardOptimizer(
            training_window=100, validation_window=30, step_size=50
        )

        # Run analysis first
        param_space = {"window": [10, 20], "threshold": [0.01, 0.02]}
        optimizer.run_walk_forward_analysis(
            sample_data, mock_strategy_class, param_space
        )

        # Export results
        export_file = tmp_path / "walk_forward_results.json"
        result = optimizer.export_results(str(export_file))

        assert result["success"] is True
        assert export_file.exists()

        # Check file content
        import json

        with open(export_file, "r") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "results" in data
        assert "analysis" in data


class TestBacktestOptimizer:
    """Test the main backtest optimizer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for backtest testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=400, freq="D")
        data = pd.DataFrame(
            {
                "close": 100 + np.cumsum(np.random.normal(0.001, 0.02, 400)),
                "volume": np.random.randint(1000000, 5000000, 400),
            },
            index=dates,
        )
        return data

    @pytest.fixture
    def mock_strategy_class(self):
        """Create mock strategy class."""

        class MockStrategy:
            def __init__(self, params):
                self.params = params

            def generate_signals(self, data):
                return pd.Series(
                    np.random.choice([-1, 0, 1], len(data)), index=data.index
                )

            def evaluate_performance(self, signals, data):
                class MockMetrics:
                    def __init__(self):
                        self.sharpe_ratio = np.random.normal(0.5, 0.3)
                        self.total_return = np.random.normal(0.1, 0.05)
                        self.max_drawdown = np.random.normal(-0.1, 0.05)
                        self.win_rate = np.random.normal(0.55, 0.1)

                return MockMetrics()

        return MockStrategy

    def test_backtest_optimizer_initialization(self):
        """Test backtest optimizer initialization."""
        config = {
            "n_regimes": 4,
            "lookback_window": 200,
            "training_window": 252,
            "validation_window": 63,
            "step_size": 21,
        }

        optimizer = BacktestOptimizer(config)

        assert optimizer.config == config
        assert optimizer.regime_detector is not None
        assert optimizer.walk_forward_optimizer is not None
        assert optimizer.regime_detector.n_regimes == 4
        assert optimizer.regime_detector.lookback_window == 200

    def test_comprehensive_backtest(self, sample_data, mock_strategy_class):
        """Test comprehensive backtest."""
        config = {"training_window": 100, "validation_window": 30, "step_size": 50}

        optimizer = BacktestOptimizer(config)
        param_space = {"window": [10, 20], "threshold": [0.01, 0.02]}

        results = optimizer.run_comprehensive_backtest(
            sample_data, mock_strategy_class, param_space
        )

        assert isinstance(results, dict)
        assert "walk_forward_results" in results
        assert "analysis" in results
        assert "regimes" in results
        assert "config" in results

        assert isinstance(results["walk_forward_results"], list)
        assert isinstance(results["analysis"], dict)
        assert isinstance(results["regimes"], list)
        assert results["config"] == config

    def test_regime_recommendations(self, sample_data):
        """Test regime-based recommendations."""
        optimizer = BacktestOptimizer()

        # First detect regimes
        optimizer.regime_detector.detect_regimes(sample_data)

        recommendations = optimizer.get_regime_recommendations(sample_data)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # All recommendations should be strings
        for rec in recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 0

    def test_regime_recommendations_no_data(self):
        """Test regime recommendations with insufficient data."""
        optimizer = BacktestOptimizer()

        # Create small dataset
        small_data = pd.DataFrame(
            {"close": np.random.randn(10), "volume": np.random.randint(1000, 10000, 10)}
        )

        recommendations = optimizer.get_regime_recommendations(small_data)
        assert "Insufficient data" in recommendations[0]

    def test_get_backtest_optimizer(self):
        """Test global backtest optimizer instance."""
        from trading.optimization.backtest_optimizer import get_backtest_optimizer

        optimizer1 = get_backtest_optimizer()
        optimizer2 = get_backtest_optimizer()

        # Should return the same instance
        assert optimizer1 is optimizer2

        # Test with custom config
        config = {"n_regimes": 5}
        optimizer3 = get_backtest_optimizer(config)
        assert optimizer3 is optimizer1  # Should still be the same instance

    def test_walk_forward_with_regime_detection(self, sample_data, mock_strategy_class):
        """Test walk-forward analysis with regime detection."""
        optimizer = BacktestOptimizer(
            {"training_window": 100, "validation_window": 30, "step_size": 50}
        )

        param_space = {"window": [10, 20], "threshold": [0.01, 0.02]}

        results = optimizer.run_comprehensive_backtest(
            sample_data, mock_strategy_class, param_space
        )

        # Check that regimes were detected
        assert len(results["regimes"]) > 0

        # Check that walk-forward results include regime information
        for result in results["walk_forward_results"]:
            assert hasattr(result, "regime")
            assert hasattr(result, "regime_confidence")
            assert isinstance(result.regime, str)
            assert 0 <= result.regime_confidence <= 1

    def test_error_handling(self):
        """Test error handling in backtest optimizer."""
        optimizer = BacktestOptimizer()

        # Test with empty data
        empty_data = pd.DataFrame()

        with pytest.raises(Exception):
            optimizer.run_comprehensive_backtest(empty_data, Mock(), {})

    def test_regime_detection_edge_cases(self, sample_data):
        """Test regime detection edge cases."""
        detector = RegimeDetector(n_regimes=1)

        # Test with single regime
        regimes = detector.detect_regimes(sample_data)
        assert len(regimes) == 1

        # Test with very high number of regimes
        detector_high = RegimeDetector(n_regimes=10)
        regimes_high = detector_high.detect_regimes(sample_data)
        assert len(regimes_high) <= 10

    def test_walk_forward_edge_cases(self, sample_data, mock_strategy_class):
        """Test walk-forward analysis edge cases."""
        # Test with very small windows
        optimizer = WalkForwardOptimizer(
            training_window=10, validation_window=5, step_size=5
        )

        param_space = {"window": [5], "threshold": [0.01]}

        results = optimizer.run_walk_forward_analysis(
            sample_data, mock_strategy_class, param_space
        )

        # Should still produce some results
        assert len(results) > 0

    def test_performance_metrics_validation(self, sample_data, mock_strategy_class):
        """Test validation of performance metrics."""
        optimizer = WalkForwardOptimizer()

        # Mock the evaluation method to return specific metrics
        def mock_evaluate(data, strategy_class, params):
            return {
                "sharpe_ratio": 1.5,
                "total_return": 0.2,
                "max_drawdown": -0.1,
                "win_rate": 0.6,
            }

        with patch.object(optimizer, "_evaluate_strategy", mock_evaluate):
            param_space = {"window": [10], "threshold": [0.01]}
            results = optimizer.run_walk_forward_analysis(
                sample_data, mock_strategy_class, param_space
            )

            # Check that metrics are reasonable
            for result in results:
                assert result.validation_performance["sharpe_ratio"] == 1.5
                assert result.validation_performance["total_return"] == 0.2
                assert result.validation_performance["max_drawdown"] == -0.1
                assert result.validation_performance["win_rate"] == 0.6
