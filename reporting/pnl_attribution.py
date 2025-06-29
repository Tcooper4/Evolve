"""PnL Attribution System for Evolve Trading Platform.

This module provides comprehensive PnL attribution analysis, breaking down
returns by model, strategy, time period, and market regime.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class AttributionBreakdown:
    """PnL attribution breakdown."""
    total_pnl: float
    model_attribution: Dict[str, float]
    strategy_attribution: Dict[str, float]
    time_attribution: Dict[str, float]
    regime_attribution: Dict[str, float]
    factor_attribution: Dict[str, float]
    residual_pnl: float
    attribution_date: str

class PnLAttributor:
    """Main PnL attribution engine."""
    
    def __init__(self):
        """Initialize the PnL attributor."""
        self.trades_history: List[Dict] = []
        self.attribution_history: List[AttributionBreakdown] = []
        
    def add_trade(self, trade: Dict) -> None:
        """Add a trade to the attribution system."""
        required_fields = ['timestamp', 'symbol', 'pnl', 'model', 'strategy']
        for field in required_fields:
            if field not in trade:
                logger.warning(f"Trade missing required field: {field}")
                return
        
        self.trades_history.append(trade)
        logger.debug(f"Added trade: {trade['symbol']} PnL={trade['pnl']:.2f}")
    
    def calculate_model_attribution(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate PnL attribution by model."""
        model_pnl = {}
        
        for trade in trades:
            model = trade.get('model', 'unknown')
            pnl = trade.get('pnl', 0)
            
            if model not in model_pnl:
                model_pnl[model] = 0
            model_pnl[model] += pnl
        
        return model_pnl
    
    def calculate_strategy_attribution(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate PnL attribution by strategy."""
        strategy_pnl = {}
        
        for trade in trades:
            strategy = trade.get('strategy', 'unknown')
            pnl = trade.get('pnl', 0)
            
            if strategy not in strategy_pnl:
                strategy_pnl[strategy] = 0
            strategy_pnl[strategy] += pnl
        
        return strategy_pnl
    
    def calculate_time_attribution(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate PnL attribution by time period."""
        time_pnl = {
            'daily': {},
            'weekly': {},
            'monthly': {},
            'hourly': {}
        }
        
        for trade in trades:
            timestamp = pd.to_datetime(trade.get('timestamp'))
            pnl = trade.get('pnl', 0)
            
            # Daily attribution
            day_key = timestamp.strftime('%Y-%m-%d')
            if day_key not in time_pnl['daily']:
                time_pnl['daily'][day_key] = 0
            time_pnl['daily'][day_key] += pnl
            
            # Weekly attribution
            week_key = timestamp.strftime('%Y-W%U')
            if week_key not in time_pnl['weekly']:
                time_pnl['weekly'][week_key] = 0
            time_pnl['weekly'][week_key] += pnl
            
            # Monthly attribution
            month_key = timestamp.strftime('%Y-%m')
            if month_key not in time_pnl['monthly']:
                time_pnl['monthly'][month_key] = 0
            time_pnl['monthly'][month_key] += pnl
            
            # Hourly attribution
            hour_key = timestamp.strftime('%Y-%m-%d %H:00')
            if hour_key not in time_pnl['hourly']:
                time_pnl['hourly'][hour_key] = 0
            time_pnl['hourly'][hour_key] += pnl
        
        return time_pnl
    
    def calculate_regime_attribution(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate PnL attribution by market regime."""
        regime_pnl = {}
        
        for trade in trades:
            regime = trade.get('regime', 'unknown')
            pnl = trade.get('pnl', 0)
            
            if regime not in regime_pnl:
                regime_pnl[regime] = 0
            regime_pnl[regime] += pnl
        
        return regime_pnl
    
    def calculate_factor_attribution(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate PnL attribution by market factors."""
        factor_pnl = {
            'momentum': 0,
            'mean_reversion': 0,
            'volatility': 0,
            'correlation': 0,
            'liquidity': 0
        }
        
        for trade in trades:
            factors = trade.get('factors', {})
            pnl = trade.get('pnl', 0)
            
            # Distribute PnL across factors based on weights
            total_weight = sum(factors.values()) if factors else 1
            if total_weight > 0:
                for factor, weight in factors.items():
                    if factor in factor_pnl:
                        factor_pnl[factor] += (pnl * weight / total_weight)
        
        return factor_pnl
    
    def run_attribution_analysis(self, 
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               symbols: Optional[List[str]] = None) -> AttributionBreakdown:
        """Run comprehensive PnL attribution analysis."""
        
        # Filter trades by date and symbols
        filtered_trades = self.trades_history.copy()
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            filtered_trades = [t for t in filtered_trades 
                             if pd.to_datetime(t.get('timestamp')) >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            filtered_trades = [t for t in filtered_trades 
                             if pd.to_datetime(t.get('timestamp')) <= end_dt]
        
        if symbols:
            filtered_trades = [t for t in filtered_trades 
                             if t.get('symbol') in symbols]
        
        if not filtered_trades:
            logger.warning("No trades found for attribution analysis")
            return AttributionBreakdown(
                total_pnl=0,
                model_attribution={},
                strategy_attribution={},
                time_attribution={},
                regime_attribution={},
                factor_attribution={},
                residual_pnl=0,
                attribution_date=datetime.now().isoformat()
            )
        
        # Calculate total PnL
        total_pnl = sum(trade.get('pnl', 0) for trade in filtered_trades)
        
        # Calculate attributions
        model_attribution = self.calculate_model_attribution(filtered_trades)
        strategy_attribution = self.calculate_strategy_attribution(filtered_trades)
        time_attribution = self.calculate_time_attribution(filtered_trades)
        regime_attribution = self.calculate_regime_attribution(filtered_trades)
        factor_attribution = self.calculate_factor_attribution(filtered_trades)
        
        # Calculate residual (unexplained PnL)
        explained_pnl = (
            sum(model_attribution.values()) +
            sum(strategy_attribution.values()) +
            sum(regime_attribution.values()) +
            sum(factor_attribution.values())
        )
        residual_pnl = total_pnl - explained_pnl
        
        attribution = AttributionBreakdown(
            total_pnl=total_pnl,
            model_attribution=model_attribution,
            strategy_attribution=strategy_attribution,
            time_attribution=time_attribution,
            regime_attribution=regime_attribution,
            factor_attribution=factor_attribution,
            residual_pnl=residual_pnl,
            attribution_date=datetime.now().isoformat()
        )
        
        self.attribution_history.append(attribution)
        logger.info(f"Completed attribution analysis: Total PnL={total_pnl:.2f}")
        
        return attribution
    
    def get_attribution_summary(self, 
                              lookback_days: int = 30) -> Dict[str, Any]:
        """Get summary of recent attribution analysis."""
        if not self.attribution_history:
            return {}
        
        recent_attributions = [
            a for a in self.attribution_history
            if (datetime.now() - pd.to_datetime(a.attribution_date)).days <= lookback_days
        ]
        
        if not recent_attributions:
            return {}
        
        # Aggregate recent attributions
        total_pnl = sum(a.total_pnl for a in recent_attributions)
        
        # Aggregate model performance
        model_performance = {}
        for attr in recent_attributions:
            for model, pnl in attr.model_attribution.items():
                if model not in model_performance:
                    model_performance[model] = 0
                model_performance[model] += pnl
        
        # Aggregate strategy performance
        strategy_performance = {}
        for attr in recent_attributions:
            for strategy, pnl in attr.strategy_attribution.items():
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = 0
                strategy_performance[strategy] += pnl
        
        # Find best and worst performers
        best_model = max(model_performance.items(), key=lambda x: x[1])[0] if model_performance else None
        worst_model = min(model_performance.items(), key=lambda x: x[1])[0] if model_performance else None
        best_strategy = max(strategy_performance.items(), key=lambda x: x[1])[0] if strategy_performance else None
        worst_strategy = min(strategy_performance.items(), key=lambda x: x[1])[0] if strategy_performance else None
        
        summary = {
            'total_pnl': total_pnl,
            'num_attributions': len(recent_attributions),
            'lookback_days': lookback_days,
            'model_performance': model_performance,
            'strategy_performance': strategy_performance,
            'best_model': best_model,
            'worst_model': worst_model,
            'best_strategy': best_strategy,
            'worst_strategy': worst_strategy,
            'avg_daily_pnl': total_pnl / lookback_days if lookback_days > 0 else 0
        }
        
        return summary
    
    def export_attribution_report(self, 
                                filepath: str = "reports/pnl_attribution.json") -> None:
        """Export attribution report to file."""
        try:
            report = {
                'attribution_history': [asdict(attr) for attr in self.attribution_history],
                'summary': self.get_attribution_summary(),
                'trades_count': len(self.trades_history),
                'export_date': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Exported attribution report to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting attribution report: {e}")

# Global attributor instance
attributor = PnLAttributor()

def get_attributor() -> PnLAttributor:
    """Get the global PnL attributor instance."""
    return attributor 