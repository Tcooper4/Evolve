"""
Tests for the report exporter functionality.

This module tests the FallbackReportExporter with comprehensive
coverage including file content validation and schema verification.
"""

import json
import os
import tempfile
from datetime import datetime

import pytest

from fallback.report_exporter import FallbackReportExporter


class TestFallbackReportExporter:
    """Test cases for FallbackReportExporter."""

    @pytest.fixture
    def exporter(self):
        """Create an exporter instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = FallbackReportExporter()
            exporter._export_dir = temp_dir
            yield exporter

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return {
            "trades": [
                {"symbol": "AAPL", "action": "buy", "quantity": 100, "price": 150.0},
                {"symbol": "GOOGL", "action": "sell", "quantity": 50, "price": 2800.0},
            ],
            "performance": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.08,
            },
            "metadata": {"strategy": "RSI", "period": "2024-01-01 to 2024-01-31"},
        }

    def test_initialization(self, exporter):
        """Test exporter initialization."""
        assert exporter._status == "fallback"
        # The export directory is set to temp dir in fixture, so just check it exists
        assert exporter._export_dir is not None

    def test_export_json(self, exporter, sample_data):
        """Test JSON export with content validation."""
        filepath = exporter.export_report(sample_data, format="json")

        # Check file exists
        assert os.path.exists(filepath)

        # Validate file content
        with open(filepath, "r") as f:
            exported_data = json.load(f)

        # Check schema
        assert "data" in exported_data
        assert "metadata" in exported_data
        assert exported_data["metadata"]["fallback_mode"] is True

        # Validate data structure
        assert "trades" in exported_data["data"]
        assert "performance" in exported_data["data"]
        assert len(exported_data["data"]["trades"]) == 2

        # Validate specific values
        assert exported_data["data"]["trades"][0]["symbol"] == "AAPL"
        assert exported_data["data"]["performance"]["total_return"] == 0.15

    def test_export_csv(self, exporter, sample_data):
        """Test CSV export with content validation."""
        filepath = exporter.export_report(sample_data, format="csv")

        # Check file exists
        assert os.path.exists(filepath)

        # Validate file content
        with open(filepath, "r") as f:
            content = f.read()

        # Check CSV structure
        lines = content.strip().split("\n")
        assert len(lines) > 1  # Header + data

        # Check header
        header = lines[0]
        assert "key" in header.lower()
        assert "value" in header.lower()

        # Check data rows
        data_lines = lines[1:]
        assert len(data_lines) > 0

        # Validate specific data presence
        content_lower = content.lower()
        assert "aapl" in content_lower
        assert "0.15" in content  # total_return value

    def test_export_text(self, exporter, sample_data):
        """Test text export with content validation."""
        filepath = exporter.export_report(sample_data, format="txt")

        # Check file exists
        assert os.path.exists(filepath)

        # Validate file content
        with open(filepath, "r") as f:
            content = f.read()

        # Check text structure
        assert "FALLBACK REPORT EXPORT" in content
        assert "METADATA:" in content
        assert "REPORT DATA:" in content

        # Validate specific data presence
        assert "AAPL" in content
        assert "0.15" in content  # total_return
        assert "RSI" in content  # strategy

    def test_filename_uniqueness(self, exporter, sample_data):
        """Test that exported filenames are unique."""
        filepath1 = exporter.export_report(sample_data, format="json")
        filepath2 = exporter.export_report(sample_data, format="json")

        # Check files are different
        assert filepath1 != filepath2

        # Check both files exist
        assert os.path.exists(filepath1)
        assert os.path.exists(filepath2)

        # Validate both files have correct content
        for filepath in [filepath1, filepath2]:
            with open(filepath, "r") as f:
                data = json.load(f)
                assert "data" in data
                assert "metadata" in data

    def test_custom_filename(self, exporter, sample_data):
        """Test export with custom filename."""
        custom_filename = "my_custom_report.json"
        filepath = exporter.export_report(
            sample_data, format="json", filename=custom_filename
        )

        # Check filename is preserved
        assert custom_filename in filepath
        assert os.path.exists(filepath)

        # Validate content
        with open(filepath, "r") as f:
            data = json.load(f)
            assert "data" in data

    def test_unsupported_format(self, exporter, sample_data):
        """Test handling of unsupported format."""
        filepath = exporter.export_report(sample_data, format="unsupported")

        # Should default to JSON
        assert filepath.endswith(".json")
        assert os.path.exists(filepath)

        # Validate content is still correct
        with open(filepath, "r") as f:
            data = json.load(f)
            assert "data" in data

    def test_empty_data(self, exporter):
        """Test export with empty data."""
        empty_data = {}
        filepath = exporter.export_report(empty_data, format="json")

        assert os.path.exists(filepath)

        # Validate content
        with open(filepath, "r") as f:
            data = json.load(f)
            assert "data" in data
            assert data["data"] == {}

    def test_nested_data(self, exporter):
        """Test export with deeply nested data."""
        nested_data = {
            "level1": {
                "level2": {
                    "level3": {"value": 42, "list": [1, 2, 3], "dict": {"key": "value"}}
                }
            }
        }

        filepath = exporter.export_report(nested_data, format="json")

        # Validate content preserves structure
        with open(filepath, "r") as f:
            data = json.load(f)
            assert data["data"]["level1"]["level2"]["level3"]["value"] == 42
            assert data["data"]["level1"]["level2"]["level3"]["list"] == [1, 2, 3]

    def test_large_data(self, exporter):
        """Test export with large dataset."""
        large_data = {
            "trades": [
                {"symbol": f"STOCK_{i}", "price": i * 10.0, "quantity": i}
                for i in range(1000)
            ],
            "performance": {"total_return": 0.15},
        }

        filepath = exporter.export_report(large_data, format="json")

        # Check file exists and has content
        assert os.path.exists(filepath)
        file_size = os.path.getsize(filepath)
        assert file_size > 0

        # Validate content structure
        with open(filepath, "r") as f:
            data = json.load(f)
            assert len(data["data"]["trades"]) == 1000
            assert data["data"]["trades"][0]["symbol"] == "STOCK_0"
            assert data["data"]["trades"][999]["symbol"] == "STOCK_999"

    def test_special_characters(self, exporter):
        """Test export with special characters in data."""
        special_data = {
            "trades": [
                {"symbol": "AAPL", "description": 'Apple Inc. (AAPL) - "iPhone" maker'},
                {
                    "symbol": "TSLA",
                    "description": "Tesla, Inc. - Electric vehicles & energy",
                },
            ],
            "notes": "Special chars: émojis 🚀, unicode 中文, symbols @#$%",
        }

        filepath = exporter.export_report(special_data, format="json")

        # Validate content preserves special characters
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "émojis" in data["data"]["notes"]
            assert "🚀" in data["data"]["notes"]
            assert "中文" in data["data"]["notes"]

    def test_error_handling(self, exporter):
        """Test error handling during export."""
        # Test with non-serializable data
        non_serializable_data = {
            "function": lambda x: x,  # Functions can't be serialized
            "object": object(),  # Some objects can't be serialized
        }

        # Should handle gracefully
        filepath = exporter.export_report(non_serializable_data, format="json")
        assert os.path.exists(filepath)

    def test_metadata_inclusion(self, exporter, sample_data):
        """Test that metadata is properly included."""
        filepath = exporter.export_report(sample_data, format="json")

        with open(filepath, "r") as f:
            data = json.load(f)

        # Check metadata structure
        metadata = data["metadata"]
        assert "export_timestamp" in metadata
        assert "format" in metadata
        assert metadata["format"] == "json"
        assert metadata["fallback_mode"] is True
        assert metadata["exporter"] == "FallbackReportExporter"

        # Validate timestamp format
        timestamp = metadata["export_timestamp"]
        datetime.fromisoformat(timestamp)  # Should not raise exception

    def test_multiple_formats(self, exporter, sample_data):
        """Test export in multiple formats simultaneously."""
        formats = ["json", "csv", "txt"]
        filepaths = []

        for fmt in formats:
            filepath = exporter.export_report(sample_data, format=fmt)
            filepaths.append(filepath)
            assert os.path.exists(filepath)

        # All files should be different
        assert len(set(filepaths)) == len(filepaths)

        # Validate each format has correct content
        for filepath in filepaths:
            assert os.path.getsize(filepath) > 0

    def test_file_permissions(self, exporter, sample_data):
        """Test file permission handling."""
        filepath = exporter.export_report(sample_data, format="json")

        # Check file is readable
        assert os.access(filepath, os.R_OK)

        # Check file is writable (for potential updates)
        assert os.access(filepath, os.W_OK)

    def test_directory_creation(self, exporter):
        """Test that export directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_export_dir = os.path.join(temp_dir, "new_export_dir")
            exporter._export_dir = new_export_dir

            # Directory should not exist initially
            assert not os.path.exists(new_export_dir)

            # Export should create directory
            sample_data = {"test": "data"}
            filepath = exporter.export_report(sample_data, format="json")

            # Directory should now exist
            assert os.path.exists(new_export_dir)
            assert os.path.exists(filepath)
