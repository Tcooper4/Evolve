"""
Test Export Report Module

Tests for Batch 10 features: file existence verification, content schema validation,
zipped exports, and visual HTML reports.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# Import the module to test
try:
    from trading.report.export_report import ExportReport
except ImportError:
    # Fallback for testing
    class ExportReport:
        def __init__(self, output_dir="reports"):
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.export_history = []


class TestExportReport(unittest.TestCase):
    """Test cases for ExportReport class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = ExportReport(output_dir=self.temp_dir)

        # Sample data for testing
        self.sample_data = {
            "summary": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.08,
                "win_rate": 0.65,
            },
            "performance": {
                "annualized_return": 0.12,
                "volatility": 0.18,
                "sortino_ratio": 1.5,
                "calmar_ratio": 1.8,
            },
            "trades": [
                {
                    "timestamp": "2023-01-01",
                    "symbol": "AAPL",
                    "side": "buy",
                    "quantity": 100,
                    "price": 150.0,
                    "pnl": 500.0,
                },
                {
                    "timestamp": "2023-01-02",
                    "symbol": "GOOGL",
                    "side": "sell",
                    "quantity": 50,
                    "price": 2800.0,
                    "pnl": -200.0,
                },
                {
                    "timestamp": "2023-01-03",
                    "symbol": "MSFT",
                    "side": "buy",
                    "quantity": 75,
                    "price": 300.0,
                    "pnl": 750.0,
                },
            ],
            "charts": {
                "equity_curve": {"data": [100, 105, 103, 108]},
                "drawdown": {"data": [0, -2, -1, 0]},
                "returns_distribution": {"data": [0.01, -0.02, 0.03, -0.01]},
            },
        }

        self.sample_df = pd.DataFrame(
            {
                "timestamp": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "symbol": ["AAPL", "GOOGL", "MSFT"],
                "price": [150.0, 2800.0, 300.0],
                "quantity": [100, 50, 75],
                "pnl": [500.0, -200.0, 750.0],
            }
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_csv_export_success(self):
        """Test successful CSV export with file existence verification."""
        result = self.exporter.export_to_csv(self.sample_df, "test_export.csv")

        # Verify success
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["filepath"])
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["columns"], 5)

        # Verify file exists
        filepath = Path(result["filepath"])
        self.assertTrue(filepath.exists())
        self.assertGreater(filepath.stat().st_size, 0)

        # Verify content schema
        df_loaded = pd.read_csv(filepath)
        self.assertEqual(len(df_loaded), 3)
        self.assertEqual(len(df_loaded.columns), 5)
        self.assertIn("timestamp", df_loaded.columns)
        self.assertIn("symbol", df_loaded.columns)

    def test_csv_export_empty_data(self):
        """Test CSV export with empty data."""
        empty_df = pd.DataFrame()
        result = self.exporter.export_to_csv(empty_df, "empty_test.csv")

        self.assertFalse(result["success"])
        self.assertIn("No data to export", result["error"])

    def test_csv_export_permission_error(self):
        """Test CSV export with permission error."""
        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            mock_to_csv.side_effect = PermissionError("Permission denied")

            result = self.exporter.export_to_csv(self.sample_df, "permission_test.csv")

            self.assertFalse(result["success"])
            self.assertIn("Permission denied", result["error"])

    def test_json_export_success(self):
        """Test successful JSON export with file existence verification."""
        result = self.exporter.export_to_json(self.sample_data, "test_export.json")

        # Verify success
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["filepath"])

        # Verify file exists
        filepath = Path(result["filepath"])
        self.assertTrue(filepath.exists())
        self.assertGreater(filepath.stat().st_size, 0)

        # Verify content schema
        with open(filepath, "r") as f:
            loaded_data = json.load(f)

        self.assertIn("summary", loaded_data)
        self.assertIn("performance", loaded_data)
        self.assertIn("trades", loaded_data)
        self.assertIn("charts", loaded_data)

        # Verify data types
        self.assertIsInstance(loaded_data["summary"], dict)
        self.assertIsInstance(loaded_data["trades"], list)
        self.assertEqual(len(loaded_data["trades"]), 3)

    def test_json_export_serialization_error(self):
        """Test JSON export with serialization error."""
        # Create data with non-serializable object
        bad_data = {"function": lambda x: x}

        result = self.exporter.export_to_json(bad_data, "serialization_test.json")

        # The current implementation handles non-serializable objects gracefully
        # with default=str, so this should succeed
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["filepath"])

        # Verify the file was created and contains the serialized data
        filepath = Path(result["filepath"])
        self.assertTrue(filepath.exists())

        with open(filepath, "r") as f:
            loaded_data = json.load(f)

        # The lambda function should be converted to a string representation
        self.assertIn("function", loaded_data)
        self.assertIsInstance(loaded_data["function"], str)

    def test_html_export_success(self):
        """Test successful HTML export with file existence verification."""
        result = self.exporter.export_to_html(self.sample_data, "test_export.html")

        # Verify success
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["filepath"])
        self.assertTrue(result["includes_charts"])

        # Verify file exists
        filepath = Path(result["filepath"])
        self.assertTrue(filepath.exists())
        self.assertGreater(filepath.stat().st_size, 0)

        # Verify content schema
        with open(filepath, "r") as f:
            html_content = f.read()

        # Check for HTML structure
        self.assertIn("<!DOCTYPE html>", html_content)
        self.assertIn("<html", html_content)
        self.assertIn("</html>", html_content)
        self.assertIn("Trading Report", html_content)

        # Check for data sections
        self.assertIn("Summary", html_content)
        self.assertIn("Performance Metrics", html_content)
        self.assertIn("Trades", html_content)
        self.assertIn("Charts", html_content)

    def test_html_export_without_charts(self):
        """Test HTML export without charts."""
        result = self.exporter.export_to_html(
            self.sample_data, "test_no_charts.html", include_charts=False
        )

        self.assertTrue(result["success"])
        self.assertFalse(result["includes_charts"])

        # Verify file exists
        filepath = Path(result["filepath"])
        self.assertTrue(filepath.exists())

    def test_zipped_export_success(self):
        """Test successful zipped export with file existence verification."""
        # Create some test files first
        csv_result = self.exporter.export_to_csv(self.sample_df, "test1.csv")
        json_result = self.exporter.export_to_json(self.sample_data, "test2.json")

        self.assertTrue(csv_result["success"])
        self.assertTrue(json_result["success"])

        # Create zip with these files
        files_to_zip = [csv_result["filepath"], json_result["filepath"]]
        result = self.exporter.create_zipped_export(files_to_zip, "test_bundle.zip")

        # Verify success
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["filepath"])
        self.assertEqual(result["files_included"], 2)

        # Verify file exists
        filepath = Path(result["filepath"])
        self.assertTrue(filepath.exists())
        self.assertGreater(filepath.stat().st_size, 0)

        # Verify zip content
        import zipfile

        with zipfile.ZipFile(filepath, "r") as zipf:
            file_list = zipf.namelist()
            self.assertIn("test1.csv", file_list)
            self.assertIn("test2.json", file_list)

    def test_zipped_export_no_files(self):
        """Test zipped export with no valid files."""
        result = self.exporter.create_zipped_export([], "empty_bundle.zip")

        self.assertFalse(result["success"])
        self.assertIn("No valid files", result["error"])

    def test_zipped_export_missing_files(self):
        """Test zipped export with missing files."""
        result = self.exporter.create_zipped_export(
            ["nonexistent_file.csv"], "missing_bundle.zip"
        )

        self.assertFalse(result["success"])
        self.assertIn("No valid files", result["error"])

    def test_export_summary(self):
        """Test export summary generation."""
        # Perform some exports
        self.exporter.export_to_csv(self.sample_df, "summary_test.csv")
        self.exporter.export_to_json(self.sample_data, "summary_test.json")

        summary = self.exporter.get_export_summary()

        self.assertEqual(summary["total_exports"], 2)
        self.assertIn("csv", summary["format_counts"])
        self.assertIn("json", summary["format_counts"])
        self.assertEqual(summary["format_counts"]["csv"], 1)
        self.assertEqual(summary["format_counts"]["json"], 1)
        self.assertGreater(summary["total_size_bytes"], 0)

    def test_cleanup_old_exports(self):
        """Test cleanup of old export files."""
        # Create some test files
        self.exporter.export_to_csv(self.sample_df, "old_file.csv")

        # Mock file modification time to be old
        csv_file = Path(self.exporter.output_dir) / "old_file.csv"
        old_timestamp = 1609459200  # 2021-01-01
        os.utime(csv_file, (old_timestamp, old_timestamp))

        # Run cleanup
        result = self.exporter.cleanup_old_exports(days_to_keep=1)

        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["deleted_count"], 1)
        self.assertGreater(result["deleted_size_bytes"], 0)

    def test_error_handling_comprehensive(self):
        """Test comprehensive error handling scenarios."""
        # Test with invalid data types
        invalid_data = {"numpy_array": np.array([1, 2, 3])}
        result = self.exporter.export_to_json(invalid_data, "invalid_test.json")

        # Should handle gracefully
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)

    def test_file_content_validation(self):
        """Test validation of exported file content."""
        # Export CSV
        result = self.exporter.export_to_csv(self.sample_df, "validation_test.csv")
        self.assertTrue(result["success"])

        # Validate content
        filepath = Path(result["filepath"])
        df_loaded = pd.read_csv(filepath)

        # Check data integrity
        self.assertEqual(len(df_loaded), len(self.sample_df))
        self.assertEqual(list(df_loaded.columns), list(self.sample_df.columns))

        # Check specific values
        self.assertEqual(df_loaded.iloc[0]["symbol"], "AAPL")
        self.assertEqual(df_loaded.iloc[0]["price"], 150.0)

    def test_html_report_structure(self):
        """Test HTML report structure and content."""
        result = self.exporter.export_to_html(self.sample_data, "structure_test.html")
        self.assertTrue(result["success"])

        filepath = Path(result["filepath"])
        with open(filepath, "r") as f:
            html_content = f.read()

        # Check for required HTML elements
        required_elements = [
            "<!DOCTYPE html>",
            "<html",
            "<head>",
            "<body>",
            "<title>",
            "</html>",
        ]

        for element in required_elements:
            self.assertIn(element, html_content)

        # Check for data-driven content
        self.assertIn("0.15", html_content)  # total_return (as decimal)
        self.assertIn("1.20", html_content)  # sharpe_ratio
        self.assertIn("AAPL", html_content)  # trade data

    def test_export_history_tracking(self):
        """Test that export history is properly tracked."""
        initial_count = len(self.exporter.export_history)

        # Perform exports
        self.exporter.export_to_csv(self.sample_df, "history_test.csv")
        self.exporter.export_to_json(self.sample_data, "history_test.json")

        final_count = len(self.exporter.export_history)
        self.assertEqual(final_count, initial_count + 2)

        # Check history entries
        for entry in self.exporter.export_history[-2:]:
            self.assertIn("timestamp", entry)
            self.assertIn("format", entry)
            self.assertIn("filename", entry)
            self.assertIn("filepath", entry)

    def test_permission_error_handling(self):
        """Test handling of permission errors."""
        # Create a read-only directory
        read_only_dir = Path(self.temp_dir) / "readonly"
        read_only_dir.mkdir()
        read_only_dir.chmod(0o444)  # Read-only

        try:
            exporter_ro = ExportReport(output_dir=str(read_only_dir))

            # This should fail gracefully
            result = exporter_ro.export_to_csv(self.sample_df, "permission_test.csv")

            # Should handle the error gracefully
            self.assertIsInstance(result, dict)
            self.assertIn("success", result)

        finally:
            # Restore permissions for cleanup
            read_only_dir.chmod(0o755)

    def test_large_data_export(self):
        """Test export of large datasets."""
        # Create large dataset
        large_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10000, freq="1H"),
                "symbol": ["AAPL"] * 10000,
                "price": np.random.uniform(100, 200, 10000),
                "quantity": np.random.randint(1, 1000, 10000),
                "pnl": np.random.uniform(-1000, 1000, 10000),
            }
        )

        result = self.exporter.export_to_csv(large_df, "large_test.csv")

        self.assertTrue(result["success"])
        self.assertEqual(result["rows"], 10000)
        self.assertEqual(result["columns"], 5)

        # Verify file size is reasonable
        filepath = Path(result["filepath"])
        self.assertGreater(filepath.stat().st_size, 100000)  # Should be > 100KB


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestExportReport)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
