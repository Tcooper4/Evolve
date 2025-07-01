# -*- coding: utf-8 -*-
"""
Data Quality & Anomaly Agent for detecting data issues and managing recovery.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from trading.data.providers.alpha_vantage_provider import AlphaVantageProvider
from trading.data.providers.yfinance_provider import YFinanceProvider
from trading.data.data_loader import DataLoader
from trading.memory.agent_memory import AgentMemory
from .base_agent_interface import BaseAgent, AgentConfig, AgentResult

class AnomalyType(str, Enum):
    """Types of data anomalies."""
    MISSING_DATA = "missing_data"
    OUTLIER = "outlier"
    DELAYED_DATA = "delayed_data"
    INCONSISTENT_DATA = "inconsistent_data"
    VOLUME_SPIKE = "volume_spike"
    PRICE_GAP = "price_gap"
    ZERO_VOLUME = "zero_volume"
    DUPLICATE_DATA = "duplicate_data"

class DataQualityLevel(str, Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class DataAnomaly:
    """Data anomaly information."""
    anomaly_type: AnomalyType
    timestamp: datetime
    symbol: str
    severity: str  # low, medium, high, critical
    description: str
    affected_columns: List[str]
    confidence: float
    suggested_action: str

@dataclass
class DataQualityReport:
    """Data quality assessment report."""
    symbol: str
    timestamp: datetime
    quality_level: DataQualityLevel
    completeness_score: float
    consistency_score: float
    timeliness_score: float
    accuracy_score: float
    anomalies: List[DataAnomaly]
    recommendations: List[str]
    overall_score: float

class DataQualityAgent(BaseAgent):
    """
    Agent responsible for:
    - Detecting data quality issues and anomalies
    - Monitoring data consistency and timeliness
    - Automatically routing to backup data providers
    - Providing data quality reports and recommendations
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="DataQualityAgent",
                enabled=True,
                priority=1,
                max_concurrent_runs=1,
                timeout_seconds=300,
                retry_attempts=3,
                custom_config={}
            )
        super().__init__(config)
        self.config_dict = config.custom_config or {}
        self.logger = logging.getLogger(__name__)
        self.memory = AgentMemory()
        self.data_loader = DataLoader()
        
        # Data providers
        self.primary_provider = AlphaVantageProvider()
        self.backup_providers = {
            'yfinance': YFinanceProvider(),
            'alpha_vantage': AlphaVantageProvider()
        }
        
        # Configuration
        self.anomaly_thresholds = self.config_dict.get('anomaly_thresholds', {
            'z_score_threshold': 3.0,
            'iqr_multiplier': 1.5,
            'missing_data_threshold': 0.05,
            'delay_threshold_minutes': 15,
            'volume_spike_threshold': 5.0,
            'price_gap_threshold': 0.1
        })
        
        self.quality_thresholds = self.config_dict.get('quality_thresholds', {
            'excellent': 0.9,
            'good': 0.8,
            'fair': 0.7,
            'poor': 0.6
        })
        
        # Anomaly detection methods
        self.detection_methods = {
            'z_score': self._detect_z_score_anomalies,
            'iqr': self._detect_iqr_anomalies,
            'rolling_std': self._detect_rolling_std_anomalies,
            'statistical': self._detect_statistical_anomalies
        }
        
        # Quality tracking
        self.quality_history: Dict[str, List[DataQualityReport]] = {}
        self.anomaly_history: Dict[str, List[DataAnomaly]] = {}
        
        # Load existing data
        self._load_quality_history()

    def _setup(self):
        pass

    async def execute(self, **kwargs) -> AgentResult:
        """Execute the data quality assessment logic.
        Args:
            **kwargs: data, symbol, data_source, action, etc.
        Returns:
            AgentResult
        """
        try:
            action = kwargs.get('action', 'assess_quality')
            
            if action == 'assess_quality':
                data = kwargs.get('data')
                symbol = kwargs.get('symbol')
                data_source = kwargs.get('data_source', 'primary')
                
                if data is None or symbol is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameters: data and symbol"
                    )
                
                report = await self.assess_data_quality(data, symbol, data_source)
                return AgentResult(success=True, data={
                    "quality_report": report.__dict__,
                    "quality_level": report.quality_level.value,
                    "overall_score": report.overall_score,
                    "anomalies_count": len(report.anomalies)
                })
                
            elif action == 'route_to_backup':
                symbol = kwargs.get('symbol')
                data_issues = kwargs.get('data_issues', [])
                
                if symbol is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameter: symbol"
                    )
                
                backup_data = await self.route_to_backup_provider(symbol, data_issues)
                if backup_data is not None:
                    return AgentResult(success=True, data={
                        "backup_data_shape": backup_data.shape,
                        "backup_data_columns": list(backup_data.columns)
                    })
                else:
                    return AgentResult(success=False, error_message="No backup data available")
                    
            elif action == 'get_quality_summary':
                symbol = kwargs.get('symbol')
                
                if symbol is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameter: symbol"
                    )
                
                summary = self.get_quality_summary(symbol)
                if summary:
                    return AgentResult(success=True, data={"quality_summary": summary})
                else:
                    return AgentResult(success=False, error_message="No quality summary available")
                    
            else:
                return AgentResult(success=False, error_message=f"Unknown action: {action}")
                
        except Exception as e:
            return self.handle_error(e)

    async def assess_data_quality(self, 
                                data: pd.DataFrame,
                                symbol: str,
                                data_source: str = 'primary') -> DataQualityReport:
        """
        Assess the quality of market data.
        
        Args:
            data: Market data DataFrame
            symbol: Asset symbol
            data_source: Source of the data
            
        Returns:
            DataQualityReport with quality assessment
        """
        try:
            self.logger.info(f"Assessing data quality for {symbol}")
            
            # Detect anomalies
            anomalies = await self._detect_anomalies(data, symbol)
            
            # Calculate quality scores
            completeness_score = self._calculate_completeness_score(data)
            consistency_score = self._calculate_consistency_score(data)
            timeliness_score = self._calculate_timeliness_score(data, symbol)
            accuracy_score = self._calculate_accuracy_score(data, anomalies)
            
            # Calculate overall score
            overall_score = (
                0.3 * completeness_score +
                0.3 * consistency_score +
                0.2 * timeliness_score +
                0.2 * accuracy_score
            )
            
            # Determine quality level
            quality_level = self._determine_quality_level(overall_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                data, anomalies, overall_score, symbol
            )
            
            # Create report
            report = DataQualityReport(
                symbol=symbol,
                timestamp=datetime.now(),
                quality_level=quality_level,
                completeness_score=completeness_score,
                consistency_score=consistency_score,
                timeliness_score=timeliness_score,
                accuracy_score=accuracy_score,
                anomalies=anomalies,
                recommendations=recommendations,
                overall_score=overall_score
            )
            
            # Store report
            self._store_quality_report(report)
            
            # Log assessment
            self.logger.info(f"Data quality assessment for {symbol}: {quality_level.value} "
                           f"(score: {overall_score:.3f})")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error assessing data quality: {str(e)}")
            return self._create_error_report(symbol, str(e))
    
    async def _detect_anomalies(self, data: pd.DataFrame, symbol: str) -> List[DataAnomaly]:
        """Detect anomalies in the data."""
        try:
            anomalies = []
            
            # Check for missing data
            missing_anomalies = self._detect_missing_data(data, symbol)
            anomalies.extend(missing_anomalies)
            
            # Check for outliers in price data
            price_anomalies = self._detect_price_anomalies(data, symbol)
            anomalies.extend(price_anomalies)
            
            # Check for volume anomalies
            volume_anomalies = self._detect_volume_anomalies(data, symbol)
            anomalies.extend(volume_anomalies)
            
            # Check for data consistency
            consistency_anomalies = self._detect_consistency_anomalies(data, symbol)
            anomalies.extend(consistency_anomalies)
            
            # Check for duplicates
            duplicate_anomalies = self._detect_duplicate_data(data, symbol)
            anomalies.extend(duplicate_anomalies)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    def _detect_missing_data(self, data: pd.DataFrame, symbol: str) -> List[DataAnomaly]:
        """Detect missing data points."""
        try:
            anomalies = []
            threshold = self.anomaly_thresholds['missing_data_threshold']
            
            for column in ['open', 'high', 'low', 'close', 'volume']:
                if column in data.columns:
                    missing_ratio = data[column].isnull().sum() / len(data)
                    
                    if missing_ratio > threshold:
                        anomaly = DataAnomaly(
                            anomaly_type=AnomalyType.MISSING_DATA,
                            timestamp=datetime.now(),
                            symbol=symbol,
                            severity='high' if missing_ratio > 0.1 else 'medium',
                            description=f"Missing {missing_ratio:.1%} of {column} data",
                            affected_columns=[column],
                            confidence=min(1.0, missing_ratio / threshold),
                            suggested_action="Fetch missing data from backup provider"
                        )
                        anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting missing data: {str(e)}")
            return []
    
    def _detect_price_anomalies(self, data: pd.DataFrame, symbol: str) -> List[DataAnomaly]:
        """Detect price anomalies using multiple methods."""
        try:
            anomalies = []
            
            if 'close' not in data.columns:
                return anomalies
            
            # Z-score method
            z_score_anomalies = self._detect_z_score_anomalies(data['close'], symbol, 'close')
            anomalies.extend(z_score_anomalies)
            
            # IQR method
            iqr_anomalies = self._detect_iqr_anomalies(data['close'], symbol, 'close')
            anomalies.extend(iqr_anomalies)
            
            # Price gaps
            gap_anomalies = self._detect_price_gaps(data, symbol)
            anomalies.extend(gap_anomalies)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting price anomalies: {str(e)}")
            return []
    
    def _detect_z_score_anomalies(self, 
                                 series: pd.Series, 
                                 symbol: str, 
                                 column: str) -> List[DataAnomaly]:
        """Detect anomalies using Z-score method."""
        try:
            anomalies = []
            threshold = self.anomaly_thresholds['z_score_threshold']
            
            # Calculate Z-scores
            mean_val = series.mean()
            std_val = series.std()
            
            if std_val == 0:
                return anomalies
            
            z_scores = abs((series - mean_val) / std_val)
            outlier_indices = z_scores > threshold
            
            if outlier_indices.any():
                outlier_values = series[outlier_indices]
                max_z_score = z_scores.max()
                
                anomaly = DataAnomaly(
                    anomaly_type=AnomalyType.OUTLIER,
                    timestamp=datetime.now(),
                    symbol=symbol,
                    severity='high' if max_z_score > threshold * 2 else 'medium',
                    description=f"Z-score anomaly in {column}: max Z-score = {max_z_score:.2f}",
                    affected_columns=[column],
                    confidence=min(1.0, max_z_score / (threshold * 2)),
                    suggested_action="Verify data source and check for data errors"
                )
                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting Z-score anomalies: {str(e)}")
            return []
    
    def _detect_iqr_anomalies(self, 
                             series: pd.Series, 
                             symbol: str, 
                             column: str) -> List[DataAnomaly]:
        """Detect anomalies using IQR method."""
        try:
            anomalies = []
            multiplier = self.anomaly_thresholds['iqr_multiplier']
            
            # Calculate IQR
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            
            if iqr == 0:
                return anomalies
            
            # Define bounds
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            
            # Find outliers
            outliers = (series < lower_bound) | (series > upper_bound)
            
            if outliers.any():
                outlier_count = outliers.sum()
                outlier_ratio = outlier_count / len(series)
                
                anomaly = DataAnomaly(
                    anomaly_type=AnomalyType.OUTLIER,
                    timestamp=datetime.now(),
                    symbol=symbol,
                    severity='medium' if outlier_ratio < 0.05 else 'high',
                    description=f"IQR anomaly in {column}: {outlier_count} outliers ({outlier_ratio:.1%})",
                    affected_columns=[column],
                    confidence=min(1.0, outlier_ratio * 10),
                    suggested_action="Review outlier data points for validity"
                )
                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting IQR anomalies: {str(e)}")
            return []
    
    def _detect_price_gaps(self, data: pd.DataFrame, symbol: str) -> List[DataAnomaly]:
        """Detect price gaps between consecutive periods."""
        try:
            anomalies = []
            threshold = self.anomaly_thresholds['price_gap_threshold']
            
            if 'close' not in data.columns:
                return anomalies
            
            # Calculate price changes
            price_changes = data['close'].pct_change().abs()
            large_gaps = price_changes > threshold
            
            if large_gaps.any():
                max_gap = price_changes.max()
                gap_count = large_gaps.sum()
                
                anomaly = DataAnomaly(
                    anomaly_type=AnomalyType.PRICE_GAP,
                    timestamp=datetime.now(),
                    symbol=symbol,
                    severity='high' if max_gap > threshold * 2 else 'medium',
                    description=f"Large price gap detected: {max_gap:.1%} change ({gap_count} occurrences)",
                    affected_columns=['close'],
                    confidence=min(1.0, max_gap / (threshold * 2)),
                    suggested_action="Verify price data and check for corporate actions"
                )
                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting price gaps: {str(e)}")
            return []
    
    def _detect_volume_anomalies(self, data: pd.DataFrame, symbol: str) -> List[DataAnomaly]:
        """Detect volume anomalies."""
        try:
            anomalies = []
            
            if 'volume' not in data.columns:
                return anomalies
            
            # Volume spikes
            volume_ma = data['volume'].rolling(window=20).mean()
            volume_ratio = data['volume'] / volume_ma
            spike_threshold = self.anomaly_thresholds['volume_spike_threshold']
            
            volume_spikes = volume_ratio > spike_threshold
            if volume_spikes.any():
                max_spike = volume_ratio.max()
                spike_count = volume_spikes.sum()
                
                anomaly = DataAnomaly(
                    anomaly_type=AnomalyType.VOLUME_SPIKE,
                    timestamp=datetime.now(),
                    symbol=symbol,
                    severity='medium',
                    description=f"Volume spike detected: {max_spike:.1f}x average ({spike_count} occurrences)",
                    affected_columns=['volume'],
                    confidence=min(1.0, max_spike / (spike_threshold * 2)),
                    suggested_action="Check for news events or corporate actions"
                )
                anomalies.append(anomaly)
            
            # Zero volume
            zero_volume = (data['volume'] == 0).sum()
            if zero_volume > 0:
                zero_ratio = zero_volume / len(data)
                
                anomaly = DataAnomaly(
                    anomaly_type=AnomalyType.ZERO_VOLUME,
                    timestamp=datetime.now(),
                    symbol=symbol,
                    severity='high' if zero_ratio > 0.1 else 'medium',
                    description=f"Zero volume periods: {zero_volume} occurrences ({zero_ratio:.1%})",
                    affected_columns=['volume'],
                    confidence=min(1.0, zero_ratio * 5),
                    suggested_action="Verify data source and check for trading halts"
                )
                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting volume anomalies: {str(e)}")
            return []
    
    def _detect_consistency_anomalies(self, data: pd.DataFrame, symbol: str) -> List[DataAnomaly]:
        """Detect data consistency issues."""
        try:
            anomalies = []
            
            # Check OHLC consistency
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                # High should be >= Low
                invalid_hl = data['high'] < data['low']
                if invalid_hl.any():
                    invalid_count = invalid_hl.sum()
                    
                    anomaly = DataAnomaly(
                        anomaly_type=AnomalyType.INCONSISTENT_DATA,
                        timestamp=datetime.now(),
                        symbol=symbol,
                        severity='critical',
                        description=f"Invalid OHLC data: high < low in {invalid_count} periods",
                        affected_columns=['high', 'low'],
                        confidence=1.0,
                        suggested_action="Immediate data source verification required"
                    )
                    anomalies.append(anomaly)
                
                # Close should be between High and Low
                invalid_close = (data['close'] > data['high']) | (data['close'] < data['low'])
                if invalid_close.any():
                    invalid_count = invalid_close.sum()
                    
                    anomaly = DataAnomaly(
                        anomaly_type=AnomalyType.INCONSISTENT_DATA,
                        timestamp=datetime.now(),
                        symbol=symbol,
                        severity='high',
                        description=f"Invalid close price: outside high-low range in {invalid_count} periods",
                        affected_columns=['close', 'high', 'low'],
                        confidence=1.0,
                        suggested_action="Verify close price data"
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting consistency anomalies: {str(e)}")
            return []
    
    def _detect_duplicate_data(self, data: pd.DataFrame, symbol: str) -> List[DataAnomaly]:
        """Detect duplicate data entries."""
        try:
            anomalies = []
            
            # Check for duplicate timestamps
            if 'timestamp' in data.columns or data.index.name == 'timestamp':
                duplicates = data.duplicated()
                if duplicates.any():
                    duplicate_count = duplicates.sum()
                    
                    anomaly = DataAnomaly(
                        anomaly_type=AnomalyType.DUPLICATE_DATA,
                        timestamp=datetime.now(),
                        symbol=symbol,
                        severity='medium',
                        description=f"Duplicate data entries: {duplicate_count} duplicates found",
                        affected_columns=list(data.columns),
                        confidence=1.0,
                        suggested_action="Remove duplicate entries"
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting duplicate data: {str(e)}")
            return []
    
    def _calculate_completeness_score(self, data: pd.DataFrame) -> float:
        """Calculate data completeness score."""
        try:
            if data.empty:
                return 0.0
            
            # Calculate missing data ratio for each column
            missing_ratios = []
            for column in ['open', 'high', 'low', 'close', 'volume']:
                if column in data.columns:
                    missing_ratio = data[column].isnull().sum() / len(data)
                    missing_ratios.append(missing_ratio)
            
            if not missing_ratios:
                return 1.0
            
            # Completeness score is inverse of average missing ratio
            avg_missing_ratio = np.mean(missing_ratios)
            return max(0.0, 1.0 - avg_missing_ratio)
            
        except Exception as e:
            self.logger.error(f"Error calculating completeness score: {str(e)}")
            return 0.5
    
    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """Calculate data consistency score."""
        try:
            if data.empty:
                return 0.0
            
            consistency_issues = 0
            total_checks = 0
            
            # Check OHLC consistency
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                total_checks += 2
                
                # High >= Low
                invalid_hl = (data['high'] < data['low']).sum()
                consistency_issues += invalid_hl / len(data)
                
                # Close between High and Low
                invalid_close = ((data['close'] > data['high']) | (data['close'] < data['low'])).sum()
                consistency_issues += invalid_close / len(data)
            
            # Check for negative values
            for column in ['open', 'high', 'low', 'close', 'volume']:
                if column in data.columns:
                    total_checks += 1
                    negative_count = (data[column] < 0).sum()
                    consistency_issues += negative_count / len(data)
            
            if total_checks == 0:
                return 1.0
            
            # Consistency score is inverse of issue ratio
            issue_ratio = consistency_issues / total_checks
            return max(0.0, 1.0 - issue_ratio)
            
        except Exception as e:
            self.logger.error(f"Error calculating consistency score: {str(e)}")
            return 0.5
    
    def _calculate_timeliness_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Calculate data timeliness score."""
        try:
            if data.empty:
                return 0.0
            
            # Check if data is recent
            if 'timestamp' in data.columns:
                latest_timestamp = pd.to_datetime(data['timestamp'].max())
            else:
                latest_timestamp = pd.to_datetime(data.index.max())
            
            current_time = datetime.now()
            time_diff = current_time - latest_timestamp
            
            # Score based on how recent the data is
            if time_diff.total_seconds() < 300:  # 5 minutes
                return 1.0
            elif time_diff.total_seconds() < 3600:  # 1 hour
                return 0.8
            elif time_diff.total_seconds() < 86400:  # 1 day
                return 0.6
            else:
                return 0.2
            
        except Exception as e:
            self.logger.error(f"Error calculating timeliness score: {str(e)}")
            return 0.5
    
    def _calculate_accuracy_score(self, data: pd.DataFrame, anomalies: List[DataAnomaly]) -> float:
        """Calculate data accuracy score based on anomalies."""
        try:
            if not anomalies:
                return 1.0
            
            # Weight anomalies by severity
            severity_weights = {
                'low': 0.1,
                'medium': 0.3,
                'high': 0.6,
                'critical': 1.0
            }
            
            total_weight = 0
            weighted_issues = 0
            
            for anomaly in anomalies:
                weight = severity_weights.get(anomaly.severity, 0.3)
                total_weight += weight
                weighted_issues += weight * anomaly.confidence
            
            if total_weight == 0:
                return 1.0
            
            # Accuracy score is inverse of weighted issue ratio
            issue_ratio = weighted_issues / total_weight
            return max(0.0, 1.0 - issue_ratio)
            
        except Exception as e:
            self.logger.error(f"Error calculating accuracy score: {str(e)}")
            return 0.5
    
    def _determine_quality_level(self, overall_score: float) -> DataQualityLevel:
        """Determine data quality level based on overall score."""
        try:
            thresholds = self.quality_thresholds
            
            if overall_score >= thresholds['excellent']:
                return DataQualityLevel.EXCELLENT
            elif overall_score >= thresholds['good']:
                return DataQualityLevel.GOOD
            elif overall_score >= thresholds['fair']:
                return DataQualityLevel.FAIR
            elif overall_score >= thresholds['poor']:
                return DataQualityLevel.POOR
            else:
                return DataQualityLevel.CRITICAL
                
        except Exception as e:
            self.logger.error(f"Error determining quality level: {str(e)}")
            return DataQualityLevel.FAIR
    
    def _generate_recommendations(self, 
                                data: pd.DataFrame,
                                anomalies: List[DataAnomaly],
                                overall_score: float,
                                symbol: str) -> List[str]:
        """Generate recommendations for data quality improvement."""
        try:
            recommendations = []
            
            # General recommendations based on overall score
            if overall_score < 0.6:
                recommendations.append("Consider switching to backup data provider")
                recommendations.append("Implement additional data validation checks")
            
            if overall_score < 0.8:
                recommendations.append("Increase data quality monitoring frequency")
            
            # Specific recommendations based on anomalies
            for anomaly in anomalies:
                if anomaly.anomaly_type == AnomalyType.MISSING_DATA:
                    recommendations.append("Fetch missing data from alternative sources")
                
                elif anomaly.anomaly_type == AnomalyType.OUTLIER:
                    recommendations.append("Implement outlier detection and handling")
                
                elif anomaly.anomaly_type == AnomalyType.DELAYED_DATA:
                    recommendations.append("Switch to real-time data feed")
                
                elif anomaly.anomaly_type == AnomalyType.INCONSISTENT_DATA:
                    recommendations.append("Verify data source integrity")
            
            # Remove duplicates
            recommendations = list(set(recommendations))
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Review data quality manually"]
    
    async def route_to_backup_provider(self, 
                                     symbol: str,
                                     data_issues: List[DataAnomaly]) -> Optional[pd.DataFrame]:
        """Route data request to backup provider when primary fails."""
        try:
            self.logger.info(f"Routing {symbol} to backup provider due to data issues")
            
            # Try backup providers in order
            for provider_name, provider in self.backup_providers.items():
                try:
                    self.logger.info(f"Trying backup provider: {provider_name}")
                    
                    # Get data from backup provider
                    backup_data = await provider.get_historical_data(
                        symbol=symbol,
                        period='1y',
                        interval='1d'
                    )
                    
                    if backup_data is not None and not backup_data.empty:
                        # Assess quality of backup data
                        backup_report = await self.assess_data_quality(
                            backup_data, symbol, provider_name
                        )
                        
                        if backup_report.quality_level in [DataQualityLevel.EXCELLENT, DataQualityLevel.GOOD]:
                            self.logger.info(f"Successfully retrieved data from {provider_name}")
                            
                            # Store backup usage
                            self._store_backup_usage(symbol, provider_name, data_issues)
                            
                            return backup_data
                        else:
                            self.logger.warning(f"Backup data from {provider_name} has quality issues")
                    
                except Exception as e:
                    self.logger.error(f"Error with backup provider {provider_name}: {str(e)}")
                    continue
            
            self.logger.error(f"All backup providers failed for {symbol}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error routing to backup provider: {str(e)}")
            return None
    
    def _store_quality_report(self, report: DataQualityReport):
        """Store quality report in history."""
        try:
            symbol = report.symbol
            
            if symbol not in self.quality_history:
                self.quality_history[symbol] = []
            
            self.quality_history[symbol].append(report)
            
            # Keep only recent reports
            cutoff_date = datetime.now() - timedelta(days=30)
            self.quality_history[symbol] = [
                r for r in self.quality_history[symbol]
                if r.timestamp > cutoff_date
            ]
            
            # Store in memory
            memory_key = f"quality_report_{symbol}"
            self.memory.store(memory_key, {
                'reports': [r.__dict__ for r in self.quality_history[symbol]],
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            self.logger.error(f"Error storing quality report: {str(e)}")
    
    def _store_backup_usage(self, 
                          symbol: str, 
                          provider_name: str, 
                          data_issues: List[DataAnomaly]):
        """Store backup provider usage information."""
        try:
            usage_data = {
                'symbol': symbol,
                'provider': provider_name,
                'timestamp': datetime.now().isoformat(),
                'issues': [issue.__dict__ for issue in data_issues]
            }
            
            self.memory.store('backup_usage', usage_data)
            
        except Exception as e:
            self.logger.error(f"Error storing backup usage: {str(e)}")
    
    def _create_error_report(self, symbol: str, error_message: str) -> DataQualityReport:
        """Create error report when assessment fails."""
        return DataQualityReport(
            symbol=symbol,
            timestamp=datetime.now(),
            quality_level=DataQualityLevel.CRITICAL,
            completeness_score=0.0,
            consistency_score=0.0,
            timeliness_score=0.0,
            accuracy_score=0.0,
            anomalies=[],
            recommendations=[f"Fix assessment error: {error_message}"],
            overall_score=0.0
        )
    
    def get_quality_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get quality summary for a symbol."""
        try:
            if symbol not in self.quality_history:
                return None
            
            reports = self.quality_history[symbol]
            if not reports:
                return None
            
            recent_report = reports[-1]
            
            return {
                'symbol': symbol,
                'current_quality': recent_report.quality_level.value,
                'overall_score': recent_report.overall_score,
                'anomaly_count': len(recent_report.anomalies),
                'last_assessment': recent_report.timestamp.isoformat(),
                'recommendations': recent_report.recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Error getting quality summary: {str(e)}")

    def _load_quality_history(self):
        """Load quality history from memory."""
        try:
            history_file = Path("memory/quality_history.json")
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for symbol, reports in data.items():
                        self.quality_history[symbol] = [
                            DataQualityReport(**r) for r in reports
                        ]
                        
        except Exception as e:
            self.logger.error(f"Error loading quality history: {str(e)}")
    
    def save_quality_history(self):
        """Save quality history to file."""
        try:
            history_file = Path("memory/quality_history.json")
            history_file.parent.mkdir(exist_ok=True)
            
            data = {}
            for symbol, reports in self.quality_history.items():
                data[symbol] = [r.__dict__ for r in reports]
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving quality history: {str(e)}")