"""
Risk Model Compliance Flags - Batch 20
Dynamic risk thresholds with rolling percentiles and compliance logging
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk levels for compliance flags."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceFlagType(Enum):
    """Types of compliance flags."""
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"
    CONCENTRATION = "concentration"
    LIQUIDITY = "liquidity"
    CORRELATION = "correlation"
    LEVERAGE = "leverage"
    EXPOSURE = "exposure"

@dataclass
class ComplianceFlag:
    """Compliance flag with rationale."""
    flag_type: ComplianceFlagType
    risk_level: RiskLevel
    current_value: float
    threshold: float
    percentile: float
    rationale: str
    triggered_at: datetime
    asset_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskThresholds:
    """Dynamic risk thresholds."""
    volatility_95th: float
    volatility_99th: float
    drawdown_95th: float
    drawdown_99th: float
    concentration_95th: float
    concentration_99th: float
    correlation_95th: float
    correlation_99th: float
    updated_at: datetime
    sample_count: int

class RiskModelComplianceFlags:
    """
    Enhanced risk model with dynamic compliance flags.
    
    Features:
    - Dynamic thresholds based on rolling percentiles
    - Comprehensive compliance flag logging
    - Multi-timeframe risk analysis
    - Adaptive threshold updates
    - Risk rationale documentation
    """
    
    def __init__(self, 
                 rolling_window: int = 252,  # 1 year of trading days
                 update_frequency: int = 5,  # Update every 5 days
                 enable_adaptive_thresholds: bool = True,
                 log_rationale: bool = True):
        """
        Initialize risk model compliance flags.
        
        Args:
            rolling_window: Rolling window for percentile calculations
            update_frequency: Frequency of threshold updates
            enable_adaptive_thresholds: Enable adaptive threshold updates
            log_rationale: Enable detailed rationale logging
        """
        self.rolling_window = rolling_window
        self.update_frequency = update_frequency
        self.enable_adaptive_thresholds = enable_adaptive_thresholds
        self.log_rationale = log_rationale
        
        # Risk data storage
        self.risk_data: Dict[str, pd.DataFrame] = {}
        self.thresholds: Dict[str, RiskThresholds] = {}
        
        # Compliance flag history
        self.compliance_flags: List[ComplianceFlag] = []
        self.active_flags: Dict[str, List[ComplianceFlag]] = {}
        
        # Statistics
        self.stats = {
            'total_flags': 0,
            'critical_flags': 0,
            'high_flags': 0,
            'medium_flags': 0,
            'low_flags': 0,
            'threshold_updates': 0
        }
        
        logger.info(f"RiskModelComplianceFlags initialized with rolling window: {rolling_window}")
    
    def add_risk_data(self, 
                     portfolio_id: str,
                     risk_metrics: Dict[str, float],
                     timestamp: datetime = None) -> None:
        """
        Add risk data for threshold calculation.
        
        Args:
            portfolio_id: Portfolio identifier
            risk_metrics: Risk metrics dictionary
            timestamp: Timestamp for the data
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if portfolio_id not in self.risk_data:
            self.risk_data[portfolio_id] = pd.DataFrame()
        
        # Create data row
        data_row = {
            'timestamp': timestamp,
            **risk_metrics
        }
        
        # Append to dataframe
        df = self.risk_data[portfolio_id]
        new_row = pd.DataFrame([data_row])
        self.risk_data[portfolio_id] = pd.concat([df, new_row], ignore_index=True)
        
        # Limit data to rolling window
        if len(self.risk_data[portfolio_id]) > self.rolling_window:
            self.risk_data[portfolio_id] = self.risk_data[portfolio_id].tail(self.rolling_window)
        
        # Update thresholds periodically
        if len(self.risk_data[portfolio_id]) % self.update_frequency == 0:
            self._update_thresholds(portfolio_id)
    
    def check_compliance(self, 
                        portfolio_id: str,
                        current_metrics: Dict[str, float],
                        asset_id: Optional[str] = None) -> List[ComplianceFlag]:
        """
        Check compliance against dynamic thresholds.
        
        Args:
            portfolio_id: Portfolio identifier
            current_metrics: Current risk metrics
            asset_id: Optional asset identifier
            
        Returns:
            List of triggered compliance flags
        """
        triggered_flags = []
        
        # Get current thresholds
        thresholds = self.thresholds.get(portfolio_id)
        if not thresholds:
            logger.warning(f"No thresholds found for portfolio {portfolio_id}")
            return triggered_flags
        
        # Check volatility
        if 'volatility' in current_metrics:
            vol_flag = self._check_volatility_compliance(
                current_metrics['volatility'], thresholds, portfolio_id, asset_id
            )
            if vol_flag:
                triggered_flags.append(vol_flag)
        
        # Check drawdown
        if 'drawdown' in current_metrics:
            dd_flag = self._check_drawdown_compliance(
                current_metrics['drawdown'], thresholds, portfolio_id, asset_id
            )
            if dd_flag:
                triggered_flags.append(dd_flag)
        
        # Check concentration
        if 'concentration' in current_metrics:
            conc_flag = self._check_concentration_compliance(
                current_metrics['concentration'], thresholds, portfolio_id, asset_id
            )
            if conc_flag:
                triggered_flags.append(conc_flag)
        
        # Check correlation
        if 'correlation' in current_metrics:
            corr_flag = self._check_correlation_compliance(
                current_metrics['correlation'], thresholds, portfolio_id, asset_id
            )
            if corr_flag:
                triggered_flags.append(corr_flag)
        
        # Check leverage
        if 'leverage' in current_metrics:
            lev_flag = self._check_leverage_compliance(
                current_metrics['leverage'], thresholds, portfolio_id, asset_id
            )
            if lev_flag:
                triggered_flags.append(lev_flag)
        
        # Store flags
        for flag in triggered_flags:
            self.compliance_flags.append(flag)
            if portfolio_id not in self.active_flags:
                self.active_flags[portfolio_id] = []
            self.active_flags[portfolio_id].append(flag)
        
        # Update statistics
        self._update_flag_stats(triggered_flags)
        
        # Log rationale if enabled
        if self.log_rationale and triggered_flags:
            self._log_compliance_rationale(triggered_flags)
        
        return triggered_flags
    
    def _check_volatility_compliance(self, 
                                   current_vol: float,
                                   thresholds: RiskThresholds,
                                   portfolio_id: str,
                                   asset_id: Optional[str]) -> Optional[ComplianceFlag]:
        """Check volatility compliance."""
        if current_vol > thresholds.volatility_99th:
            risk_level = RiskLevel.CRITICAL
            threshold = thresholds.volatility_99th
            percentile = 99.0
        elif current_vol > thresholds.volatility_95th:
            risk_level = RiskLevel.HIGH
            threshold = thresholds.volatility_95th
            percentile = 95.0
        else:
            return None
        
        rationale = (
            f"Volatility {current_vol:.4f} exceeds {percentile}th percentile threshold "
            f"({threshold:.4f}). Current volatility is in the top {100-percentile}% "
            f"of historical values."
        )
        
        return ComplianceFlag(
            flag_type=ComplianceFlagType.VOLATILITY,
            risk_level=risk_level,
            current_value=current_vol,
            threshold=threshold,
            percentile=percentile,
            rationale=rationale,
            triggered_at=datetime.now(),
            asset_id=asset_id,
            portfolio_id=portfolio_id
        )
    
    def _check_drawdown_compliance(self, 
                                 current_dd: float,
                                 thresholds: RiskThresholds,
                                 portfolio_id: str,
                                 asset_id: Optional[str]) -> Optional[ComplianceFlag]:
        """Check drawdown compliance."""
        if current_dd > thresholds.drawdown_99th:
            risk_level = RiskLevel.CRITICAL
            threshold = thresholds.drawdown_99th
            percentile = 99.0
        elif current_dd > thresholds.drawdown_95th:
            risk_level = RiskLevel.HIGH
            threshold = thresholds.drawdown_95th
            percentile = 95.0
        else:
            return None
        
        rationale = (
            f"Drawdown {current_dd:.4f} exceeds {percentile}th percentile threshold "
            f"({threshold:.4f}). Current drawdown is in the top {100-percentile}% "
            f"of historical values, indicating significant portfolio stress."
        )
        
        return ComplianceFlag(
            flag_type=ComplianceFlagType.DRAWDOWN,
            risk_level=risk_level,
            current_value=current_dd,
            threshold=threshold,
            percentile=percentile,
            rationale=rationale,
            triggered_at=datetime.now(),
            asset_id=asset_id,
            portfolio_id=portfolio_id
        )
    
    def _check_concentration_compliance(self, 
                                     current_conc: float,
                                     thresholds: RiskThresholds,
                                     portfolio_id: str,
                                     asset_id: Optional[str]) -> Optional[ComplianceFlag]:
        """Check concentration compliance."""
        if current_conc > thresholds.concentration_99th:
            risk_level = RiskLevel.CRITICAL
            threshold = thresholds.concentration_99th
            percentile = 99.0
        elif current_conc > thresholds.concentration_95th:
            risk_level = RiskLevel.HIGH
            threshold = thresholds.concentration_95th
            percentile = 95.0
        else:
            return None
        
        rationale = (
            f"Concentration {current_conc:.4f} exceeds {percentile}th percentile threshold "
            f"({threshold:.4f}). Portfolio is highly concentrated in top positions, "
            f"increasing idiosyncratic risk."
        )
        
        return ComplianceFlag(
            flag_type=ComplianceFlagType.CONCENTRATION,
            risk_level=risk_level,
            current_value=current_conc,
            threshold=threshold,
            percentile=percentile,
            rationale=rationale,
            triggered_at=datetime.now(),
            asset_id=asset_id,
            portfolio_id=portfolio_id
        )
    
    def _check_correlation_compliance(self, 
                                   current_corr: float,
                                   thresholds: RiskThresholds,
                                   portfolio_id: str,
                                   asset_id: Optional[str]) -> Optional[ComplianceFlag]:
        """Check correlation compliance."""
        if current_corr > thresholds.correlation_99th:
            risk_level = RiskLevel.CRITICAL
            threshold = thresholds.correlation_99th
            percentile = 99.0
        elif current_corr > thresholds.correlation_95th:
            risk_level = RiskLevel.HIGH
            threshold = thresholds.correlation_95th
            percentile = 95.0
        else:
            return None
        
        rationale = (
            f"Correlation {current_corr:.4f} exceeds {percentile}th percentile threshold "
            f"({threshold:.4f}). High correlation indicates reduced diversification "
            f"benefits and increased systematic risk."
        )
        
        return ComplianceFlag(
            flag_type=ComplianceFlagType.CORRELATION,
            risk_level=risk_level,
            current_value=current_corr,
            threshold=threshold,
            percentile=percentile,
            rationale=rationale,
            triggered_at=datetime.now(),
            asset_id=asset_id,
            portfolio_id=portfolio_id
        )
    
    def _check_leverage_compliance(self, 
                                current_lev: float,
                                thresholds: RiskThresholds,
                                portfolio_id: str,
                                asset_id: Optional[str]) -> Optional[ComplianceFlag]:
        """Check leverage compliance."""
        # Use concentration thresholds for leverage (similar risk profile)
        if current_lev > thresholds.concentration_99th:
            risk_level = RiskLevel.CRITICAL
            threshold = thresholds.concentration_99th
            percentile = 99.0
        elif current_lev > thresholds.concentration_95th:
            risk_level = RiskLevel.HIGH
            threshold = thresholds.concentration_95th
            percentile = 95.0
        else:
            return None
        
        rationale = (
            f"Leverage {current_lev:.4f} exceeds {percentile}th percentile threshold "
            f"({threshold:.4f}). High leverage amplifies both gains and losses, "
            f"increasing portfolio risk."
        )
        
        return ComplianceFlag(
            flag_type=ComplianceFlagType.LEVERAGE,
            risk_level=risk_level,
            current_value=current_lev,
            threshold=threshold,
            percentile=percentile,
            rationale=rationale,
            triggered_at=datetime.now(),
            asset_id=asset_id,
            portfolio_id=portfolio_id
        )
    
    def _update_thresholds(self, portfolio_id: str) -> None:
        """
        Update dynamic thresholds for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
        """
        if portfolio_id not in self.risk_data:
            return
        
        df = self.risk_data[portfolio_id]
        if len(df) < 30:  # Need minimum data for percentiles
            return
        
        try:
            # Calculate percentiles for each metric
            thresholds = RiskThresholds(
                volatility_95th=df['volatility'].quantile(0.95) if 'volatility' in df.columns else 0.0,
                volatility_99th=df['volatility'].quantile(0.99) if 'volatility' in df.columns else 0.0,
                drawdown_95th=df['drawdown'].quantile(0.95) if 'drawdown' in df.columns else 0.0,
                drawdown_99th=df['drawdown'].quantile(0.99) if 'drawdown' in df.columns else 0.0,
                concentration_95th=df['concentration'].quantile(0.95) if 'concentration' in df.columns else 0.0,
                concentration_99th=df['concentration'].quantile(0.99) if 'concentration' in df.columns else 0.0,
                correlation_95th=df['correlation'].quantile(0.95) if 'correlation' in df.columns else 0.0,
                correlation_99th=df['correlation'].quantile(0.99) if 'correlation' in df.columns else 0.0,
                updated_at=datetime.now(),
                sample_count=len(df)
            )
            
            self.thresholds[portfolio_id] = thresholds
            self.stats['threshold_updates'] += 1
            
            logger.info(f"Updated thresholds for portfolio {portfolio_id} with {len(df)} samples")
            
        except Exception as e:
            logger.error(f"Error updating thresholds for portfolio {portfolio_id}: {e}")
    
    def _log_compliance_rationale(self, flags: List[ComplianceFlag]) -> None:
        """Log detailed rationale for compliance flags."""
        for flag in flags:
            log_message = (
                f"COMPLIANCE FLAG: {flag.flag_type.value.upper()} - {flag.risk_level.value.upper()}\n"
                f"Portfolio: {flag.portfolio_id}\n"
                f"Asset: {flag.asset_id or 'N/A'}\n"
                f"Current Value: {flag.current_value:.4f}\n"
                f"Threshold ({flag.percentile}th percentile): {flag.threshold:.4f}\n"
                f"Rationale: {flag.rationale}\n"
                f"Triggered: {flag.triggered_at}"
            )
            
            if flag.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                logger.warning(log_message)
            else:
                logger.info(log_message)
    
    def _update_flag_stats(self, flags: List[ComplianceFlag]) -> None:
        """Update flag statistics."""
        self.stats['total_flags'] += len(flags)
        
        for flag in flags:
            if flag.risk_level == RiskLevel.CRITICAL:
                self.stats['critical_flags'] += 1
            elif flag.risk_level == RiskLevel.HIGH:
                self.stats['high_flags'] += 1
            elif flag.risk_level == RiskLevel.MEDIUM:
                self.stats['medium_flags'] += 1
            elif flag.risk_level == RiskLevel.LOW:
                self.stats['low_flags'] += 1
    
    def get_active_flags(self, portfolio_id: str) -> List[ComplianceFlag]:
        """Get active compliance flags for a portfolio."""
        return self.active_flags.get(portfolio_id, [])
    
    def clear_resolved_flags(self, portfolio_id: str, flag_types: Optional[List[ComplianceFlagType]] = None) -> int:
        """
        Clear resolved compliance flags.
        
        Args:
            portfolio_id: Portfolio identifier
            flag_types: Specific flag types to clear (all if None)
            
        Returns:
            Number of flags cleared
        """
        if portfolio_id not in self.active_flags:
            return 0
        
        active_flags = self.active_flags[portfolio_id]
        
        if flag_types:
            # Clear specific flag types
            flags_to_remove = [f for f in active_flags if f.flag_type in flag_types]
        else:
            # Clear all flags
            flags_to_remove = active_flags.copy()
        
        # Remove flags
        for flag in flags_to_remove:
            active_flags.remove(flag)
        
        logger.info(f"Cleared {len(flags_to_remove)} flags for portfolio {portfolio_id}")
        return len(flags_to_remove)
    
    def get_compliance_summary(self, portfolio_id: str) -> Dict[str, Any]:
        """Get compliance summary for a portfolio."""
        active_flags = self.get_active_flags(portfolio_id)
        thresholds = self.thresholds.get(portfolio_id)
        
        summary = {
            'portfolio_id': portfolio_id,
            'active_flags_count': len(active_flags),
            'critical_flags': len([f for f in active_flags if f.risk_level == RiskLevel.CRITICAL]),
            'high_flags': len([f for f in active_flags if f.risk_level == RiskLevel.HIGH]),
            'medium_flags': len([f for f in active_flags if f.risk_level == RiskLevel.MEDIUM]),
            'low_flags': len([f for f in active_flags if f.risk_level == RiskLevel.LOW]),
            'has_thresholds': thresholds is not None,
            'last_threshold_update': thresholds.updated_at.isoformat() if thresholds else None,
            'sample_count': thresholds.sample_count if thresholds else 0
        }
        
        # Add flag details
        if active_flags:
            summary['flag_details'] = [
                {
                    'type': flag.flag_type.value,
                    'risk_level': flag.risk_level.value,
                    'current_value': flag.current_value,
                    'threshold': flag.threshold,
                    'triggered_at': flag.triggered_at.isoformat()
                }
                for flag in active_flags
            ]
        
        return summary
    
    def get_risk_statistics(self) -> Dict[str, Any]:
        """Get overall risk statistics."""
        stats = self.stats.copy()
        
        # Add portfolio-level statistics
        stats['total_portfolios'] = len(self.risk_data)
        stats['portfolios_with_thresholds'] = len(self.thresholds)
        stats['portfolios_with_active_flags'] = len(self.active_flags)
        
        # Add flag type distribution
        all_flags = [f for flags in self.active_flags.values() for f in flags]
        flag_types = [f.flag_type.value for f in all_flags]
        stats['flag_type_distribution'] = {
            flag_type: flag_types.count(flag_type) 
            for flag_type in set(flag_types)
        }
        
        return stats
    
    def set_rolling_window(self, window: int):
        """Update rolling window size."""
        self.rolling_window = window
        logger.info(f"Updated rolling window to: {window}")
    
    def enable_rationale_logging(self, enable: bool = True):
        """Enable or disable rationale logging."""
        self.log_rationale = enable
        logger.info(f"Rationale logging {'enabled' if enable else 'disabled'}")

def create_risk_model_compliance_flags(rolling_window: int = 252) -> RiskModelComplianceFlags:
    """Factory function to create risk model compliance flags."""
    return RiskModelComplianceFlags(rolling_window=rolling_window)
