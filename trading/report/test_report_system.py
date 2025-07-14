"""
Test Report System

Comprehensive tests for the report generation system.
"""

import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from report.report_client import ReportClient, generate_quick_report
from report.report_generator import (
    ModelMetrics,
    ReportGenerator,
    StrategyReasoning,
    TradeMetrics,
)


class TestReportGenerator(unittest.TestCase):
    """Test the ReportGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ReportGenerator(output_dir=self.temp_dir)

        # Sample data
        self.sample_trade_data = {
            "trades": [
                {"pnl": 100, "duration": 3600},
                {"pnl": -50, "duration": 1800},
                {"pnl": 200, "duration": 7200},
                {"pnl": 75, "duration": 5400},
                {"pnl": -25, "duration": 2700},
            ]
        }

        self.sample_model_data = {
            "predictions": [100, 102, 98, 105, 103, 107, 101, 104, 106, 108],
            "actuals": [100, 101, 99, 104, 102, 106, 100, 103, 105, 107],
        }

        self.sample_strategy_data = {
            "strategy_name": "Test Strategy",
            "symbol": "AAPL",
            "timeframe": "1h",
            "signals": ["BUY", "SELL", "BUY"],
            "market_conditions": {"trend": "bullish"},
            "performance": {"total_return": 0.15},
            "parameters": {"param1": 10},
        }

        return {
            "success": True,
            "message": "Initialization completed",
            "timestamp": datetime.now().isoformat(),
        }

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_calculate_trade_metrics(self):
        """Test trade metrics calculation."""
        metrics = self.generator._calculate_trade_metrics(self.sample_trade_data)

        self.assertEqual(metrics.total_trades, 5)
        self.assertEqual(metrics.winning_trades, 3)
        self.assertEqual(metrics.losing_trades, 2)
        self.assertEqual(metrics.win_rate, 0.6)
        self.assertEqual(metrics.total_pnl, 300)
        self.assertAlmostEqual(metrics.avg_gain, 125.0)
        self.assertAlmostEqual(metrics.avg_loss, 37.5)

    def test_calculate_model_metrics(self):
        """Test model metrics calculation."""
        metrics = self.generator._calculate_model_metrics(self.sample_model_data)

        self.assertIsInstance(metrics.mse, float)
        self.assertIsInstance(metrics.mae, float)
        self.assertIsInstance(metrics.rmse, float)
        self.assertIsInstance(metrics.sharpe_ratio, float)
        self.assertIsInstance(metrics.volatility, float)
        self.assertIsInstance(metrics.accuracy, float)

    def test_generate_fallback_reasoning(self):
        """Test fallback reasoning generation."""
        reasoning = self.generator._generate_fallback_reasoning(
            self.sample_strategy_data
        )

        self.assertIsInstance(reasoning, StrategyReasoning)
        self.assertIsInstance(reasoning.summary, str)
        self.assertIsInstance(reasoning.key_factors, list)
        self.assertIsInstance(reasoning.confidence_level, float)
        self.assertIsInstance(reasoning.recommendations, list)

    def test_generate_charts(self):
        """Test chart generation."""
        charts = self.generator._generate_charts(
            self.sample_trade_data, self.sample_model_data, "AAPL"
        )

        self.assertIsInstance(charts, dict)
        # Charts should be base64 encoded strings
        for chart_data in charts.values():
            self.assertIsInstance(chart_data, str)

    def test_generate_markdown_report(self):
        """Test Markdown report generation."""
        report_data = {
            "report_id": "test_report",
            "timestamp": datetime.now().isoformat(),
            "symbol": "AAPL",
            "timeframe": "1h",
            "period": "7d",
            "trade_metrics": self.generator._calculate_trade_metrics(
                self.sample_trade_data
            ),
            "model_metrics": self.generator._calculate_model_metrics(
                self.sample_model_data
            ),
            "strategy_reasoning": self.generator._generate_fallback_reasoning(
                self.sample_strategy_data
            ),
            "charts": {},
        }

        filepath = self.generator._generate_markdown_report(report_data)

        self.assertTrue(filepath.exists())
        self.assertEqual(filepath.suffix, ".md")

        # Check content
        content = filepath.read_text()
        self.assertIn("AAPL", content)
        self.assertIn("Trade Performance", content)
        self.assertIn("Model Performance", content)

    def test_generate_html_report(self):
        """Test HTML report generation."""
        report_data = {
            "report_id": "test_report",
            "timestamp": datetime.now().isoformat(),
            "symbol": "AAPL",
            "timeframe": "1h",
            "period": "7d",
            "trade_metrics": self.generator._calculate_trade_metrics(
                self.sample_trade_data
            ),
            "model_metrics": self.generator._calculate_model_metrics(
                self.sample_model_data
            ),
            "strategy_reasoning": self.generator._generate_fallback_reasoning(
                self.sample_strategy_data
            ),
            "charts": {},
        }

        filepath = self.generator._generate_html_report(report_data)

        self.assertTrue(filepath.exists())
        self.assertEqual(filepath.suffix, ".html")

        # Check content
        content = filepath.read_text()
        self.assertIn("<html", content)
        self.assertIn("AAPL", content)
        self.assertIn("Trading Report", content)

    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation."""
        report_data = self.generator.generate_comprehensive_report(
            trade_data=self.sample_trade_data,
            model_data=self.sample_model_data,
            strategy_data=self.sample_strategy_data,
            symbol="AAPL",
            timeframe="1h",
            period="7d",
            report_id="test_comprehensive",
        )

        self.assertIsInstance(report_data, dict)
        self.assertIn("report_id", report_data)
        self.assertIn("trade_metrics", report_data)
        self.assertIn("model_metrics", report_data)
        self.assertIn("strategy_reasoning", report_data)
        self.assertIn("files", report_data)

        # Check that files were created
        files = report_data["files"]
        self.assertIn("markdown", files)
        self.assertIn("html", files)
        self.assertIn("pdf", files)

        # Check file existence
        for file_path in files.values():
            self.assertTrue(Path(file_path).exists())


class TestReportClient(unittest.TestCase):
    """Test the ReportClient class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.client = ReportClient(output_dir=self.temp_dir)

        # Sample data
        self.sample_trade_data = {
            "trades": [
                {"pnl": 100, "duration": 3600},
                {"pnl": -50, "duration": 1800},
                {"pnl": 200, "duration": 7200},
            ]
        }

        self.sample_model_data = {
            "predictions": [100, 102, 98, 105, 103],
            "actuals": [100, 101, 99, 104, 102],
        }

        self.sample_strategy_data = {
            "strategy_name": "Test Strategy",
            "symbol": "AAPL",
            "timeframe": "1h",
            "signals": ["BUY", "SELL", "BUY"],
            "market_conditions": {"trend": "bullish"},
            "performance": {"total_return": 0.15},
            "parameters": {"param1": 10},
        }

        return {
            "success": True,
            "message": "Initialization completed",
            "timestamp": datetime.now().isoformat(),
        }

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_generate_report(self):
        """Test report generation."""
        report_data = self.client.generate_report(
            trade_data=self.sample_trade_data,
            model_data=self.sample_model_data,
            strategy_data=self.sample_strategy_data,
            symbol="AAPL",
            timeframe="1h",
            period="7d",
        )

        self.assertIsInstance(report_data, dict)
        self.assertIn("report_id", report_data)
        self.assertIn("trade_metrics", report_data)
        self.assertIn("model_metrics", report_data)
        self.assertIn("strategy_reasoning", report_data)

    def test_list_available_reports(self):
        """Test listing available reports."""
        # Generate a report first
        self.client.generate_report(
            trade_data=self.sample_trade_data,
            model_data=self.sample_model_data,
            strategy_data=self.sample_strategy_data,
            symbol="AAPL",
            timeframe="1h",
            period="7d",
        )

        reports = self.client.list_available_reports()
        self.assertIsInstance(reports, list)

        # Should have at least one report
        self.assertGreater(len(reports), 0)

        # Check report structure
        if reports:
            report = reports[0]
            self.assertIn("report_id", report)
            self.assertIn("format", report)
            self.assertIn("file_path", report)
            self.assertIn("created", report)

    def test_get_service_status(self):
        """Test service status retrieval."""
        status = self.client.get_service_status()

        self.assertIsInstance(status, dict)
        self.assertIn("service_name", status)
        self.assertIn("running", status)
        self.assertIn("redis_connected", status)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Sample data
        self.sample_trade_data = {
            "trades": [
                {"pnl": 100, "duration": 3600},
                {"pnl": -50, "duration": 1800},
                {"pnl": 200, "duration": 7200},
            ]
        }

        self.sample_model_data = {
            "predictions": [100, 102, 98, 105, 103],
            "actuals": [100, 101, 99, 104, 102],
        }

        self.sample_strategy_data = {
            "strategy_name": "Test Strategy",
            "symbol": "AAPL",
            "timeframe": "1h",
            "signals": ["BUY", "SELL", "BUY"],
            "market_conditions": {"trend": "bullish"},
            "performance": {"total_return": 0.15},
            "parameters": {"param1": 10},
        }

        return {
            "success": True,
            "message": "Initialization completed",
            "timestamp": datetime.now().isoformat(),
        }

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_generate_quick_report(self):
        """Test quick report generation."""
        report_data = generate_quick_report(
            trade_data=self.sample_trade_data,
            model_data=self.sample_model_data,
            strategy_data=self.sample_strategy_data,
            symbol="AAPL",
            timeframe="1h",
            period="7d",
        )

        self.assertIsInstance(report_data, dict)
        self.assertIn("report_id", report_data)
        self.assertIn("trade_metrics", report_data)
        self.assertIn("model_metrics", report_data)
        self.assertIn("strategy_reasoning", report_data)


class TestDataStructures(unittest.TestCase):
    """Test data structures."""

    def test_trade_metrics(self):
        """Test TradeMetrics dataclass."""
        metrics = TradeMetrics(
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            win_rate=0.6,
            total_pnl=500.0,
            avg_gain=100.0,
            avg_loss=50.0,
            max_drawdown=100.0,
            sharpe_ratio=1.5,
            profit_factor=3.0,
            avg_trade_duration=3600.0,
        )

        self.assertEqual(metrics.total_trades, 10)
        self.assertEqual(metrics.winning_trades, 6)
        self.assertEqual(metrics.win_rate, 0.6)
        self.assertEqual(metrics.total_pnl, 500.0)
        self.assertEqual(metrics.sharpe_ratio, 1.5)

    def test_model_metrics(self):
        """Test ModelMetrics dataclass."""
        metrics = ModelMetrics(
            mse=0.01,
            mae=0.1,
            rmse=0.1,
            sharpe_ratio=1.2,
            volatility=0.15,
            max_drawdown=0.05,
            accuracy=0.85,
            precision=0.8,
            recall=0.9,
            f1_score=0.85,
        )

        self.assertEqual(metrics.mse, 0.01)
        self.assertEqual(metrics.accuracy, 0.85)
        self.assertEqual(metrics.sharpe_ratio, 1.2)
        self.assertEqual(metrics.f1_score, 0.85)

    def test_strategy_reasoning(self):
        """Test StrategyReasoning dataclass."""
        reasoning = StrategyReasoning(
            summary="Test summary",
            key_factors=["factor1", "factor2"],
            risk_assessment="Low risk",
            confidence_level=0.8,
            recommendations=["rec1", "rec2"],
            market_conditions="Bullish market",
        )

        self.assertEqual(reasoning.summary, "Test summary")
        self.assertEqual(len(reasoning.key_factors), 2)
        self.assertEqual(reasoning.confidence_level, 0.8)
        self.assertEqual(len(reasoning.recommendations), 2)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestReportGenerator,
        TestReportClient,
        TestConvenienceFunctions,
        TestDataStructures,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
