#!/usr/bin/env python3
"""
Data quality management script.
Provides commands for assessing and improving data quality, including profiling, fixing, and reporting issues.

This script supports:
- Profiling data quality
- Fixing data quality issues
- Generating data quality reports

Usage:
    python manage_data_quality.py <command> [options]

Commands:
    profile     Profile data quality
    fix         Fix data quality issues
    report      Generate data quality report

Examples:
    # Profile data quality
    python manage_data_quality.py profile --file data/input.csv

    # Fix data quality issues
    python manage_data_quality.py fix --file data/input.csv --output data/fixed.csv

    # Generate data quality report
    python manage_data_quality.py report --file data/input.csv --output reports/data_quality.json
"""

import os
import sys
import argparse
import logging
import logging.config
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import great_expectations as ge
from great_expectations.dataset import PandasDataset

class DataQualityManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the data quality manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.data_dir = Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load application configuration."""
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)
        
        with open(config_path) as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Initialize logging configuration."""
        log_config_path = Path("config/logging_config.yaml")
        if not log_config_path.exists():
            print("Error: logging_config.yaml not found")
            sys.exit(1)
        
        with open(log_config_path) as f:
            log_config = yaml.safe_load(f)
        
        logging.config.dictConfig(log_config)

    def validate_data(self, data_path: str, schema_path: Optional[str] = None):
        """Validate data against schema and quality rules."""
        self.logger.info(f"Validating data: {data_path}")
        
        try:
            # Load data
            data = self._load_data(data_path)
            
            # Create Great Expectations dataset
            ge_data = PandasDataset(data)
            
            # Validate schema if provided
            if schema_path:
                schema = self._load_schema(schema_path)
                schema_validation = self._validate_schema(ge_data, schema)
            else:
                schema_validation = {"success": True, "results": []}
            
            # Validate data quality
            quality_validation = self._validate_quality(ge_data)
            
            # Combine results
            validation_results = {
                "timestamp": datetime.now().isoformat(),
                "data_path": data_path,
                "schema_validation": schema_validation,
                "quality_validation": quality_validation
            }
            
            # Save results
            self._save_validation_results(validation_results)
            
            # Print results
            self._print_validation_results(validation_results)
            
            return validation_results["schema_validation"]["success"] and validation_results["quality_validation"]["success"]
        except Exception as e:
            self.logger.error(f"Failed to validate data: {e}")
            return False

    def clean_data(self, data_path: str, output_path: Optional[str] = None):
        """Clean and preprocess data."""
        self.logger.info(f"Cleaning data: {data_path}")
        
        try:
            # Load data
            data = self._load_data(data_path)
            
            # Clean data
            cleaned_data = self._clean_data(data)
            
            # Save cleaned data
            if output_path:
                self._save_data(cleaned_data, output_path)
            else:
                output_path = str(Path(data_path).with_suffix(".cleaned.csv"))
                self._save_data(cleaned_data, output_path)
            
            self.logger.info(f"Data cleaned and saved to: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clean data: {e}")
            return False

    def analyze_data(self, data_path: str):
        """Analyze data quality and statistics."""
        self.logger.info(f"Analyzing data: {data_path}")
        
        try:
            # Load data
            data = self._load_data(data_path)
            
            # Generate analysis
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "data_path": data_path,
                "basic_stats": self._get_basic_stats(data),
                "quality_metrics": self._get_quality_metrics(data),
                "correlations": self._get_correlations(data)
            }
            
            # Save analysis
            self._save_analysis_results(analysis)
            
            # Print analysis
            self._print_analysis_results(analysis)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to analyze data: {e}")
            return False

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from file."""
        try:
            if data_path.endswith(".csv"):
                return pd.read_csv(data_path)
            elif data_path.endswith(".json"):
                return pd.read_json(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

    def _load_schema(self, schema_path: str) -> Dict[str, Any]:
        """Load data schema."""
        try:
            with open(schema_path) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load schema: {e}")
            raise

    def _validate_schema(self, data: PandasDataset, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against schema."""
        results = []
        success = True
        
        try:
            # Validate column types
            for column, type_info in schema["columns"].items():
                if column not in data.columns:
                    results.append({
                        "type": "error",
                        "message": f"Column {column} not found in data"
                    })
                    success = False
                    continue
                
                if type_info["type"] == "numeric":
                    if not data[column].dtype in ["int64", "float64"]:
                        results.append({
                            "type": "error",
                            "message": f"Column {column} is not numeric"
                        })
                        success = False
                elif type_info["type"] == "categorical":
                    if not data[column].dtype == "object":
                        results.append({
                            "type": "error",
                            "message": f"Column {column} is not categorical"
                        })
                        success = False
            
            # Validate constraints
            for constraint in schema.get("constraints", []):
                if constraint["type"] == "unique":
                    if not data.expect_column_values_to_be_unique(constraint["column"]).success:
                        results.append({
                            "type": "error",
                            "message": f"Column {constraint['column']} contains duplicate values"
                        })
                        success = False
                elif constraint["type"] == "not_null":
                    if not data.expect_column_values_to_not_be_null(constraint["column"]).success:
                        results.append({
                            "type": "error",
                            "message": f"Column {constraint['column']} contains null values"
                        })
                        success = False
                elif constraint["type"] == "range":
                    if not data.expect_column_values_to_be_between(
                        constraint["column"],
                        constraint["min"],
                        constraint["max"]
                    ).success:
                        results.append({
                            "type": "error",
                            "message": f"Column {constraint['column']} contains values outside range"
                        })
                        success = False
            
            return {
                "success": success,
                "results": results
            }
        except Exception as e:
            self.logger.error(f"Failed to validate schema: {e}")
            raise

    def _validate_quality(self, data: PandasDataset) -> Dict[str, Any]:
        """Validate data quality."""
        results = []
        success = True
        
        try:
            # Check for missing values
            for column in data.columns:
                missing_ratio = data[column].isnull().mean()
                if missing_ratio > 0.1:  # More than 10% missing
                    results.append({
                        "type": "warning",
                        "message": f"Column {column} has {missing_ratio:.2%} missing values"
                    })
            
            # Check for outliers
            numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns
            for column in numeric_columns:
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                outlier_ratio = (z_scores > 3).mean()
                if outlier_ratio > 0.05:  # More than 5% outliers
                    results.append({
                        "type": "warning",
                        "message": f"Column {column} has {outlier_ratio:.2%} outliers"
                    })
            
            # Check for data consistency
            for column in data.columns:
                if data[column].dtype == "object":
                    unique_ratio = data[column].nunique() / len(data)
                    if unique_ratio > 0.9:  # More than 90% unique values
                        results.append({
                            "type": "warning",
                            "message": f"Column {column} has high cardinality ({unique_ratio:.2%} unique values)"
                        })
            
            return {
                "success": success,
                "results": results
            }
        except Exception as e:
            self.logger.error(f"Failed to validate quality: {e}")
            raise

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data."""
        try:
            # Create a copy of the data
            cleaned_data = data.copy()
            
            # Handle missing values
            numeric_columns = cleaned_data.select_dtypes(include=["int64", "float64"]).columns
            categorical_columns = cleaned_data.select_dtypes(include=["object"]).columns
            
            # Impute numeric columns with mean
            if len(numeric_columns) > 0:
                imputer = SimpleImputer(strategy="mean")
                cleaned_data[numeric_columns] = imputer.fit_transform(cleaned_data[numeric_columns])
            
            # Impute categorical columns with mode
            if len(categorical_columns) > 0:
                imputer = SimpleImputer(strategy="most_frequent")
                cleaned_data[categorical_columns] = imputer.fit_transform(cleaned_data[categorical_columns])
            
            # Handle outliers
            for column in numeric_columns:
                z_scores = np.abs((cleaned_data[column] - cleaned_data[column].mean()) / cleaned_data[column].std())
                cleaned_data.loc[z_scores > 3, column] = cleaned_data[column].mean()
            
            # Scale numeric features
            if len(numeric_columns) > 0:
                scaler = StandardScaler()
                cleaned_data[numeric_columns] = scaler.fit_transform(cleaned_data[numeric_columns])
            
            return cleaned_data
        except Exception as e:
            self.logger.error(f"Failed to clean data: {e}")
            raise

    def _get_basic_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistics for data."""
        try:
            stats = {
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                "numeric_stats": data.describe().to_dict(),
                "categorical_stats": {
                    col: data[col].value_counts().to_dict()
                    for col in data.select_dtypes(include=["object"]).columns
                }
            }
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get basic stats: {e}")
            raise

    def _get_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get data quality metrics."""
        try:
            metrics = {
                "missing_values": {
                    col: data[col].isnull().sum()
                    for col in data.columns
                },
                "unique_values": {
                    col: data[col].nunique()
                    for col in data.columns
                },
                "duplicate_rows": data.duplicated().sum()
            }
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to get quality metrics: {e}")
            raise

    def _get_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get correlations between numeric columns."""
        try:
            numeric_data = data.select_dtypes(include=["int64", "float64"])
            if len(numeric_data.columns) > 1:
                return numeric_data.corr().to_dict()
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get correlations: {e}")
            raise

    def _save_data(self, data: pd.DataFrame, output_path: str):
        """Save data to file."""
        try:
            if output_path.endswith(".csv"):
                data.to_csv(output_path, index=False)
            elif output_path.endswith(".json"):
                data.to_json(output_path, orient="records")
            else:
                raise ValueError(f"Unsupported file format: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")
            raise

    def _save_validation_results(self, results: Dict[str, Any]):
        """Save validation results."""
        try:
            # Create results directory
            results_dir = self.data_dir / "validation_results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"validation_{timestamp}.json"
            
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Validation results saved to {results_file}")
        except Exception as e:
            self.logger.error(f"Failed to save validation results: {e}")
            raise

    def _save_analysis_results(self, analysis: Dict[str, Any]):
        """Save analysis results."""
        try:
            # Create results directory
            results_dir = self.data_dir / "analysis_results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"analysis_{timestamp}.json"
            
            with open(results_file, "w") as f:
                json.dump(analysis, f, indent=2)
            
            self.logger.info(f"Analysis results saved to {results_file}")
        except Exception as e:
            self.logger.error(f"Failed to save analysis results: {e}")
            raise

    def _print_validation_results(self, results: Dict[str, Any]):
        """Print validation results."""
        print("\nValidation Results:")
        print(f"\nTimestamp: {results['timestamp']}")
        print(f"Data Path: {results['data_path']}")
        
        print("\nSchema Validation:")
        if results["schema_validation"]["success"]:
            print("  ��� Schema validation passed")
        else:
            print("  ��� Schema validation failed")
            for result in results["schema_validation"]["results"]:
                print(f"    - {result['message']}")
        
        print("\nQuality Validation:")
        if results["quality_validation"]["success"]:
            print("  ��� Quality validation passed")
        else:
            print("  ��� Quality validation failed")
            for result in results["quality_validation"]["results"]:
                print(f"    - {result['message']}")

    def _print_analysis_results(self, analysis: Dict[str, Any]):
        """Print analysis results."""
        print("\nData Analysis Results:")
        print(f"\nTimestamp: {analysis['timestamp']}")
        print(f"Data Path: {analysis['data_path']}")
        
        print("\nBasic Statistics:")
        print(f"  Shape: {analysis['basic_stats']['shape']}")
        print(f"  Columns: {', '.join(analysis['basic_stats']['columns'])}")
        
        print("\nQuality Metrics:")
        print("  Missing Values:")
        for column, count in analysis["quality_metrics"]["missing_values"].items():
            if count > 0:
                print(f"    - {column}: {count}")
        
        print("\nCorrelations:")
        for col1 in analysis["correlations"]:
            for col2, value in analysis["correlations"][col1].items():
                if abs(value) > 0.5 and col1 != col2:
                    print(f"  {col1} - {col2}: {value:.2f}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Data Quality Manager")
    parser.add_argument(
        "command",
        choices=["validate", "clean", "analyze"],
        help="Command to execute"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to input data file"
    )
    parser.add_argument(
        "--schema-path",
        help="Path to data schema file"
    )
    parser.add_argument(
        "--output-path",
        help="Path to output file"
    )
    
    args = parser.parse_args()
    manager = DataQualityManager()
    
    commands = {
        "validate": lambda: manager.validate_data(args.data_path, args.schema_path),
        "clean": lambda: manager.clean_data(args.data_path, args.output_path),
        "analyze": lambda: manager.analyze_data(args.data_path)
    }
    
    if args.command in commands:
        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 