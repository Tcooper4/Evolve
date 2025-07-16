"""
Test file to verify pandera migration from great_expectations.
Tests the DataQualityManager functionality with pandera schemas.
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

from scripts.manage_data_quality import DataQualityManager


class TestPanderaMigration(unittest.TestCase):
    """Test cases for pandera migration from great_expectations."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = DataQualityManager()
        
        # Create sample data with floats for numeric columns to ensure validation
        self.sample_data = pd.DataFrame({
            'id': [1.0, 2.0, 3.0, 4.0, 5.0],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25.0, 30.0, 35.0, 28.0, 32.0],
            'salary': [50000.0, 60000.0, 70000.0, 55000.0, 65000.0],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'diana@test.com', 'eve@test.com']
        })
        
        # Create sample schema
        self.sample_schema = {
            "columns": {
                "id": {"type": "numeric"},
                "name": {"type": "categorical"},
                "age": {"type": "numeric"},
                "salary": {"type": "numeric"},
                "email": {"type": "categorical"}
            },
            "constraints": [
                {"type": "unique", "column": "id"},
                {"type": "not_null", "column": "name"},
                {"type": "range", "column": "age", "min": 18, "max": 100},
                {"type": "range", "column": "salary", "min": 0, "max": 1000000}
            ]
        }

    def test_build_pandera_schema(self):
        """Test building pandera schema from configuration."""
        schema = self.manager._build_pandera_schema(self.sample_schema)
        self.assertIsInstance(schema, DataFrameSchema)
        expected_columns = ['id', 'name', 'age', 'salary', 'email']
        self.assertEqual(set(schema.columns.keys()), set(expected_columns))
        # Numeric columns should be pa.Float (check the actual dtype)
        for col in ['id', 'age', 'salary']:
            self.assertEqual(str(schema.columns[col].dtype), 'float64')
        # Categorical columns should be pa.String (check the actual dtype)
        for col in ['name', 'email']:
            self.assertEqual(str(schema.columns[col].dtype), 'str')

    def test_build_quality_schema(self):
        """Test building quality validation schema."""
        schema = self.manager._build_quality_schema(self.sample_data)
        self.assertIsInstance(schema, DataFrameSchema)
        self.assertEqual(set(schema.columns.keys()), set(self.sample_data.columns))
        for col in ['id', 'age', 'salary']:
            self.assertEqual(str(schema.columns[col].dtype), 'float64')
        for col in ['name', 'email']:
            self.assertEqual(str(schema.columns[col].dtype), 'str')

    def test_validate_schema_success(self):
        """Test successful schema validation."""
        result = self.manager._validate_schema(self.sample_data, self.sample_schema)
        self.assertTrue(result["success"])
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["type"], "success")

    def test_validate_schema_failure(self):
        """Test schema validation failure with invalid data."""
        invalid_data = pd.DataFrame({
            'id': [1, 1, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 150, 28, 32],
            'salary': [50000, 60000, 70000, 55000, 65000],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'diana@test.com', 'eve@test.com']
        })
        result = self.manager._validate_schema(invalid_data, self.sample_schema)
        self.assertFalse(result["success"])
        self.assertEqual(result["results"][0]["type"], "error")

    def test_validate_quality_success(self):
        """Test successful quality validation."""
        result = self.manager._validate_quality(self.sample_data)
        self.assertTrue(result["success"])
        self.assertEqual(result["results"][0]["type"], "success")

    def test_validate_quality_with_missing_values(self):
        """Test quality validation with missing values."""
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'name'] = None
        data_with_missing.loc[1, 'age'] = None
        result = self.manager._validate_quality(data_with_missing)
        self.assertTrue(result["success"])
        warning_messages = [r["message"] for r in result["results"] if r["type"] == "warning"]
        self.assertTrue(any("missing values" in msg for msg in warning_messages))

    def test_end_to_end_validation(self):
        """Test end-to-end validation with temporary files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            data_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_schema, f)
            schema_path = f.name
        try:
            success = self.manager.validate_data(data_path, schema_path)
            self.assertTrue(success)
        finally:
            Path(data_path).unlink()
            Path(schema_path).unlink()

    def test_pandera_imports(self):
        """Test that pandera imports work correctly."""
        self.assertTrue(hasattr(pa, 'DataFrameSchema'))
        self.assertTrue(hasattr(pa, 'Column'))
        self.assertTrue(hasattr(pa, 'Check'))
        self.assertTrue(hasattr(pa, 'Float'))
        self.assertTrue(hasattr(pa, 'String'))

    def test_schema_error_handling(self):
        """Test that SchemaError exceptions are handled properly."""
        failing_schema = {
            "columns": {
                "id": {"type": "numeric"}
            },
            "constraints": [
                {"type": "range", "column": "id", "min": 100, "max": 200}
            ]
        }
        failing_data = pd.DataFrame({'id': [1, 2, 3, 4, 5]})
        result = self.manager._validate_schema(failing_data, failing_schema)
        self.assertFalse(result["success"])
        self.assertEqual(result["results"][0]["type"], "error")
        self.assertIn("Schema validation failed", result["results"][0]["message"])

    def test_mixed_numeric_types(self):
        """Test schema validation with mixed int and float columns."""
        mixed_data = pd.DataFrame({
            'id': [1, 2, 3, 4.0, 5],  # mix of int and float
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25.0, 30, 35, 28, 32],
            'salary': [50000, 60000.0, 70000, 55000, 65000],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'diana@test.com', 'eve@test.com']
        })
        result = self.manager._validate_schema(mixed_data, self.sample_schema)
        self.assertTrue(result["success"])


if __name__ == '__main__':
    unittest.main() 