# -*- coding: utf-8 -*-
"""Data quality agent for monitoring and maintaining data quality."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trading.base_agent import BaseMetaAgent
from trading.data.data_loader import DataLoader
from trading.data.data_processor import DataProcessor

class DataQualityAgent(BaseMetaAgent):
    """Agent for monitoring and maintaining data quality."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the data quality agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("data_quality", config)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.data_processor = DataProcessor()
        
        # Quality thresholds
        self.thresholds = {
            "missing_ratio": 0.05,
            "outlier_ratio": 0.02,
            "min_data_points": 1000,
            "max_gap_days": 3
        }
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def run(self) -> Dict[str, Any]:
        """Run data quality monitoring.
        
        Returns:
            Dict containing data quality analysis results
        """
        results = {
            "data_sources": self._analyze_data_sources(),
            "data_quality": self._analyze_data_quality(),
            "suggested_actions": []
        }
        
        # Generate actions
        actions = self._generate_actions(results)
        results["suggested_actions"] = actions
        
        # Log results
        self.log_action("Data quality monitoring completed", results)
        
        return {'success': True, 'result': results, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _analyze_data_sources(self) -> Dict[str, Any]:
        """Analyze data sources.
        
        Returns:
            Dict containing data source analysis results
        """
        results = {}
        
        # Get available data sources
        for source in self.data_loader.get_available_sources():
            data = self.data_loader.load_data(source)
            if data is not None:
                results[source] = {
                    "last_update": data.index[-1],
                    "data_points": len(data),
                    "columns": list(data.columns),
                    "missing_ratio": self._calculate_missing_ratio(data),
                    "outlier_ratio": self._calculate_outlier_ratio(data)
                }
        
        return {'success': True, 'result': results, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze data quality.
        
        Returns:
            Dict containing data quality analysis results
        """
        results = {
            "gaps": self._find_data_gaps(),
            "anomalies": self._detect_anomalies(),
            "consistency": self._check_data_consistency()
        }
        
        return {'success': True, 'result': results, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _generate_actions(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions based on data quality analysis.
        
        Args:
            results: Results from data quality analysis
            
        Returns:
            List of suggested actions
        """
        actions = []
        
        # Check data sources
        for source, metrics in results["data_sources"].items():
            if metrics["missing_ratio"] > self.thresholds["missing_ratio"]:
                actions.append({
                    "type": "clean",
                    "target": source,
                    "description": "High missing ratio",
                    "suggestion": "Clean missing data"
                })
            
            if metrics["outlier_ratio"] > self.thresholds["outlier_ratio"]:
                actions.append({
                    "type": "filter",
                    "target": source,
                    "description": "High outlier ratio",
                    "suggestion": "Filter outliers"
                })
        
        # Check data gaps
        for gap in results["data_quality"]["gaps"]:
            if gap["days"] > self.thresholds["max_gap_days"]:
                actions.append({
                    "type": "fill",
                    "target": gap["source"],
                    "description": "Large data gap",
                    "suggestion": f"Fill gap from {gap['start']} to {gap['end']}"
                })
        
        return {'success': True, 'result': actions, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _calculate_missing_ratio(self, data: pd.DataFrame) -> float:
        """Calculate ratio of missing values.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Ratio of missing values
        """
        return {'success': True, 'result': data.isnull().sum().sum() / (data.shape[0] * data.shape[1]), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _calculate_outlier_ratio(self, data: pd.DataFrame) -> float:
        """Calculate ratio of outliers.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Ratio of outliers
        """
        outliers = 0
        total = 0
        
        for col in data.select_dtypes(include=[np.number]).columns:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            outliers += ((data[col] < (q1 - 1.5 * iqr)) | (data[col] > (q3 + 1.5 * iqr))).sum()
            total += len(data)
        
        return {'success': True, 'result': outliers / total if total > 0 else 0, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _find_data_gaps(self) -> List[Dict[str, Any]]:
        """Find gaps in data.
        
        Returns:
            List of data gaps
        """
        gaps = []
        
        for source in self.data_loader.get_available_sources():
            data = self.data_loader.load_data(source)
            if data is not None and not data.empty:
                # Find gaps in index
                index = pd.date_range(data.index[0], data.index[-1], freq='D')
                missing_dates = index.difference(data.index)
                
                if len(missing_dates) > 0:
                    # Group consecutive missing dates
                    groups = []
                    current_group = [missing_dates[0]]
                    
                    for date in missing_dates[1:]:
                        if (date - current_group[-1]).days == 1:
                            current_group.append(date)
                        else:
                            groups.append(current_group)
                            current_group = [date]
                    
                    groups.append(current_group)
                    
                    # Add gaps to list
                    for group in groups:
                        gaps.append({
                            "source": source,
                            "start": group[0],
                            "end": group[-1],
                            "days": len(group)
                        })
        
        return {'success': True, 'result': gaps, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in data.
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        for source in self.data_loader.get_available_sources():
            data = self.data_loader.load_data(source)
            if data is not None:
                for col in data.select_dtypes(include=[np.number]).columns:
                    # Calculate z-scores
                    z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                    
                    # Find anomalies
                    anomaly_indices = z_scores[z_scores > 3].index
                    
                    for idx in anomaly_indices:
                        anomalies.append({
                            "source": source,
                            "column": col,
                            "timestamp": idx,
                            "value": data.loc[idx, col],
                            "z_score": z_scores[idx]
                        })
        
        return {'success': True, 'result': anomalies, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _check_data_consistency(self) -> Dict[str, Any]:
        """Check data consistency across sources.
        
        Returns:
            Dict containing consistency check results
        """
        consistency = {
            "matching_columns": [],
            "mismatching_columns": [],
            "inconsistent_values": []
        }
        
        sources = self.data_loader.get_available_sources()
        if len(sources) < 2:
            return {'success': True, 'result': consistency, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        
        # Compare columns
        all_columns = set()
        for source in sources:
            data = self.data_loader.load_data(source)
            if data is not None:
                all_columns.update(data.columns)
        
        for source in sources:
            data = self.data_loader.load_data(source)
            if data is not None:
                missing_cols = all_columns - set(data.columns)
                if missing_cols:
                    consistency["mismatching_columns"].append({
                        "source": source,
                        "missing_columns": list(missing_cols)
                    })
                else:
                    consistency["matching_columns"].append(source)
        
        return consistency
