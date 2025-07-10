"""
Unified Trade Reporter

This module provides comprehensive trade reporting capabilities with:
- Enhanced performance metrics calculation
- Advanced risk analysis
- Comprehensive chart generation
- Multiple export formats
- Performance attribution analysis
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# Set matplotlib backend to avoid tkinter issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Local imports
from .report_generator import ReportGenerator, TradeMetrics, ModelMetrics, StrategyReasoning
from ..backtesting.backtester import Backtester
from ..backtesting.performance_analysis import PerformanceAnalyzer
from ..backtesting.risk_metrics import RiskMetricsEngine

logger = logging.getLogger(__name__)

@dataclass
class EnhancedTradeMetrics:
    """Enhanced trade performance metrics with additional risk measures."""
    # Basic metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_gain: float
    avg_loss: float
    avg_trade_duration: float
    
    # Risk metrics
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional Value at Risk 95%
    
    # Additional metrics
    total_return: float
    annualized_return: float
    volatility: float
    beta: float
    alpha: float
    information_ratio: float
    treynor_ratio: float
    
    # Trade quality metrics
    avg_win_loss_ratio: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    avg_trade_return: float
    risk_reward_ratio: float

@dataclass
class EquityCurveData:
    """Equity curve data with detailed breakdown."""
    dates: List[datetime]
    equity_values: List[float]
    cash_values: List[float]
    position_values: List[float]
    returns: List[float]
    cumulative_returns: List[float]
    drawdown: List[float]
    running_max: List[float]

@dataclass
class TradeAnalysis:
    """Detailed trade analysis."""
    trade_log: List[Dict[str, Any]]
    equity_curve: EquityCurveData
    metrics: EnhancedTradeMetrics
    risk_metrics: Dict[str, float]
    performance_attribution: Dict[str, float]

class UnifiedTradeReporter:
    """
    Unified trade reporting engine that consolidates all reporting capabilities.
    
    This class provides a single interface for generating comprehensive trade reports,
    integrating with backtesting results, and exporting in multiple formats.
    """
    
    def __init__(self, 
                 output_dir: str = "reports",
                 risk_free_rate: float = 0.02,
                 trading_days_per_year: int = 252,
                 report_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the UnifiedTradeReporter.
        
        Args:
            output_dir: Directory to save reports
            risk_free_rate: Risk-free rate for calculations
            trading_days_per_year: Trading days per year for annualization
            report_config: Configuration for report generation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        
        # Initialize components
        self.report_generator = ReportGenerator(output_dir=str(self.output_dir))
        self.performance_analyzer = PerformanceAnalyzer()
        self.risk_metrics_engine = RiskMetricsEngine(
            risk_free_rate=risk_free_rate,
            period=trading_days_per_year
        )
        
        # Report configuration
        self.report_config = report_config or {
            'include_equity_curve': True,
            'include_trade_log': True,
            'include_risk_metrics': True,
            'include_performance_attribution': True,
            'include_charts': True,
            'export_formats': ['csv', 'pdf', 'html', 'json'],
            'chart_formats': ['png', 'svg']
        }
        
        # Create subdirectories
        for subdir in ['csv', 'pdf', 'html', 'json', 'charts']:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        logger.info("UnifiedTradeReporter initialized")
    
    def generate_comprehensive_report(self,
                                    trade_data: Dict[str, Any],
                                    model_data: Optional[Dict[str, Any]] = None,
                                    strategy_data: Optional[Dict[str, Any]] = None,
                                    symbol: str = "Unknown",
                                    timeframe: str = "Unknown",
                                    period: str = "Unknown",
                                    report_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive trade report with all metrics and analysis.
        
        Args:
            trade_data: Trade performance data
            model_data: Model performance data (optional)
            strategy_data: Strategy execution data (optional)
            symbol: Trading symbol
            timeframe: Timeframe used
            period: Analysis period
            report_id: Unique report identifier
            
        Returns:
            Dictionary containing comprehensive report data and file paths
        """
        try:
            report_id = report_id or f"unified_report_{int(time.time())}"
            timestamp = datetime.now()
            
            logger.info(f"Generating comprehensive report: {report_id}")
            
            # Extract and validate trade data
            trades = trade_data.get('trades', [])
            if not trades:
                logger.warning("No trades found in trade_data")
                return self._generate_empty_report(report_id, symbol, timeframe, period)
            
            # Calculate enhanced metrics
            enhanced_metrics = self._calculate_enhanced_metrics(trades)
            
            # Generate equity curve
            equity_curve = self._generate_equity_curve(trades)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(equity_curve)
            
            # Generate performance attribution
            performance_attribution = self._calculate_performance_attribution(trades, equity_curve)
            
            # Create trade analysis
            trade_analysis = TradeAnalysis(
                trade_log=trades,
                equity_curve=equity_curve,
                metrics=enhanced_metrics,
                risk_metrics=risk_metrics,
                performance_attribution=performance_attribution
            )
            
            # Generate charts if requested
            charts = {}
            if self.report_config.get('include_charts', True):
                charts = self._generate_charts(trade_analysis, symbol)
            
            # Create comprehensive report data
            report_data = {
                'report_id': report_id,
                'timestamp': timestamp.isoformat(),
                'symbol': symbol,
                'timeframe': timeframe,
                'period': period,
                'trade_analysis': asdict(trade_analysis),
                'charts': charts,
                'summary': self._generate_summary(trade_analysis)
            }
            
            # Add model and strategy data if provided
            if model_data:
                report_data['model_data'] = model_data
            if strategy_data:
                report_data['strategy_data'] = strategy_data
            
            # Export in multiple formats
            export_paths = self._export_report(report_data, report_id)
            report_data['export_paths'] = export_paths
            
            logger.info(f"Comprehensive report generated successfully: {report_id}")
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return self._generate_error_report(report_id, str(e))
    
    def _calculate_enhanced_metrics(self, trades: List[Dict[str, Any]]) -> EnhancedTradeMetrics:
        """Calculate enhanced trade performance metrics."""
        try:
            if not trades:
                return self._get_empty_metrics()
            
            # Basic metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            losing_trades = len([t for t in trades if t.get('pnl', 0) < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # PnL metrics
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            gains = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]
            losses = [abs(t.get('pnl', 0)) for t in trades if t.get('pnl', 0) < 0]
            
            avg_gain = np.mean(gains) if gains else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
            avg_win_loss_ratio = avg_gain / avg_loss if avg_loss > 0 else float('inf')
            
            # Trade duration
            durations = [t.get('duration', 0) for t in trades if t.get('duration')]
            avg_trade_duration = np.mean(durations) if durations else 0.0
            
            # Largest wins/losses
            largest_win = max(gains) if gains else 0.0
            largest_loss = max(losses) if losses else 0.0
            
            # Consecutive wins/losses
            consecutive_wins, consecutive_losses = self._calculate_consecutive_trades(trades)
            
            # Calculate returns for risk metrics
            returns = [t.get('pnl', 0) for t in trades]
            if len(returns) > 1:
                total_return = sum(returns)
                avg_trade_return = np.mean(returns)
                volatility = np.std(returns)
                
                # Risk-adjusted metrics
                sharpe_ratio = (avg_trade_return - self.risk_free_rate/self.trading_days_per_year) / volatility if volatility > 0 else 0.0
                
                # Sortino ratio (downside deviation)
                downside_returns = [r for r in returns if r < 0]
                downside_deviation = np.std(downside_returns) if downside_returns else 0.0
                sortino_ratio = (avg_trade_return - self.risk_free_rate/self.trading_days_per_year) / downside_deviation if downside_deviation > 0 else 0.0
                
                # VaR and CVaR
                var_95 = np.percentile(returns, 5)  # 95% VaR
                cvar_95 = np.mean([r for r in returns if r <= var_95]) if var_95 else 0.0
                
                # Profit factor
                total_gains = sum(gains) if gains else 0.0
                total_losses = sum(losses) if losses else 0.0
                profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
                
                # Risk-reward ratio
                risk_reward_ratio = avg_gain / avg_loss if avg_loss > 0 else float('inf')
                
            else:
                # Fallback for insufficient data
                total_return = sum(returns) if returns else 0.0
                avg_trade_return = total_return
                volatility = 0.0
                sharpe_ratio = 0.0
                sortino_ratio = 0.0
                var_95 = 0.0
                cvar_95 = 0.0
                profit_factor = 0.0
                risk_reward_ratio = 0.0
            
            # Calculate drawdown (will be calculated in equity curve)
            max_drawdown = 0.0  # Placeholder, will be updated
            
            # Annualized return
            if avg_trade_duration > 0:
                trades_per_year = self.trading_days_per_year / avg_trade_duration
                annualized_return = avg_trade_return * trades_per_year
            else:
                annualized_return = 0.0
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
            
            # Beta and Alpha (simplified - would need benchmark data)
            beta = 1.0  # Default to market beta
            alpha = avg_trade_return - (self.risk_free_rate/self.trading_days_per_year * beta)
            
            # Information ratio (simplified)
            information_ratio = alpha / volatility if volatility > 0 else 0.0
            
            # Treynor ratio
            treynor_ratio = (avg_trade_return - self.risk_free_rate/self.trading_days_per_year) / beta if beta != 0 else 0.0
            
            return EnhancedTradeMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                avg_gain=avg_gain,
                avg_loss=avg_loss,
                avg_trade_duration=avg_trade_duration,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,  # Will be updated
                profit_factor=profit_factor,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                var_95=var_95,
                cvar_95=cvar_95,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                beta=beta,
                alpha=alpha,
                information_ratio=information_ratio,
                treynor_ratio=treynor_ratio,
                avg_win_loss_ratio=avg_win_loss_ratio,
                largest_win=largest_win,
                largest_loss=largest_loss,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses,
                avg_trade_return=avg_trade_return,
                risk_reward_ratio=risk_reward_ratio
            )
            
        except Exception as e:
            logger.error(f"Error calculating enhanced metrics: {e}")
            return self._get_empty_metrics()
    
    def _calculate_consecutive_trades(self, trades: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Calculate consecutive wins and losses."""
        if not trades:
            return 0, 0
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            pnl = trade.get('pnl', 0)
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        return max_consecutive_wins, max_consecutive_losses
    
    def _generate_equity_curve(self, trades: List[Dict[str, Any]]) -> EquityCurveData:
        """Generate equity curve from trade data."""
        try:
            if not trades:
                return self._get_empty_equity_curve()
            
            # Sort trades by timestamp
            sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', 0))
            
            # Initialize equity curve
            dates = []
            equity_values = []
            cash_values = []
            position_values = []
            returns = []
            cumulative_returns = []
            drawdown = []
            running_max = []
            
            # Starting values
            initial_cash = 100000.0  # Default starting cash
            current_cash = initial_cash
            current_equity = initial_cash
            cumulative_return = 0.0
            peak_equity = initial_cash
            
            for trade in sorted_trades:
                timestamp = trade.get('timestamp')
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid timestamp format: {timestamp}, using current time. Error: {e}")
                        timestamp = datetime.now()
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.now()
                
                # Update cash and equity
                pnl = trade.get('pnl', 0)
                current_cash += pnl
                current_equity = current_cash
                
                # Calculate return
                trade_return = pnl / initial_cash if initial_cash > 0 else 0.0
                cumulative_return += trade_return
                
                # Update peak and drawdown
                if current_equity > peak_equity:
                    peak_equity = current_equity
                
                current_drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0
                
                # Store values
                dates.append(timestamp)
                equity_values.append(current_equity)
                cash_values.append(current_cash)
                position_values.append(0.0)  # Simplified - could track actual positions
                returns.append(trade_return)
                cumulative_returns.append(cumulative_return)
                drawdown.append(current_drawdown)
                running_max.append(peak_equity)
            
            return EquityCurveData(
                dates=dates,
                equity_values=equity_values,
                cash_values=cash_values,
                position_values=position_values,
                returns=returns,
                cumulative_returns=cumulative_returns,
                drawdown=drawdown,
                running_max=running_max
            )
            
        except Exception as e:
            logger.error(f"Error generating equity curve: {e}")
            return self._get_empty_equity_curve()
    
    def _calculate_risk_metrics(self, equity_curve: EquityCurveData) -> Dict[str, float]:
        """Calculate comprehensive risk metrics from equity curve."""
        try:
            if not equity_curve.returns:
                return {}
            
            returns = np.array(equity_curve.returns)
            
            # Basic risk metrics
            volatility = np.std(returns) * np.sqrt(self.trading_days_per_year) if len(returns) > 1 else 0.0
            var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0.0
            cvar_95 = np.mean(returns[returns <= var_95]) if var_95 and len(returns) > 0 else 0.0
            
            # Maximum drawdown
            max_drawdown = max(equity_curve.drawdown) if equity_curve.drawdown else 0.0
            
            # Downside deviation
            downside_returns = returns[returns < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(self.trading_days_per_year) if len(downside_returns) > 1 else 0.0
            
            # Skewness and kurtosis
            skewness = self._calculate_skewness(returns) if len(returns) > 2 else 0.0
            kurtosis = self._calculate_kurtosis(returns) if len(returns) > 2 else 0.0
            
            return {
                'volatility': volatility,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'max_drawdown': max_drawdown,
                'downside_deviation': downside_deviation,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'total_return': sum(returns),
                'avg_return': np.mean(returns),
                'return_std': np.std(returns)
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        if len(returns) < 3:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        return np.mean(((returns - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns."""
        if len(returns) < 4:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        return np.mean(((returns - mean) / std) ** 4) - 3
    
    def _calculate_performance_attribution(self, trades: List[Dict[str, Any]], 
                                         equity_curve: EquityCurveData) -> Dict[str, float]:
        """Calculate performance attribution analysis."""
        try:
            if not trades:
                return {}
            
            # Total performance
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            
            # Performance by trade type
            buy_trades = [t for t in trades if t.get('type') == 'buy']
            sell_trades = [t for t in trades if t.get('type') == 'sell']
            
            buy_pnl = sum(t.get('pnl', 0) for t in buy_trades)
            sell_pnl = sum(t.get('pnl', 0) for t in sell_trades)
            
            # Performance by strategy (if available)
            strategies = {}
            for trade in trades:
                strategy = trade.get('strategy', 'unknown')
                if strategy not in strategies:
                    strategies[strategy] = []
                strategies[strategy].append(trade.get('pnl', 0))
            
            strategy_performance = {
                strategy: sum(pnls) for strategy, pnls in strategies.items()
            }
            
            # Time-based attribution
            if equity_curve.dates:
                # Monthly performance
                monthly_performance = self._calculate_monthly_performance(trades, equity_curve.dates)
            else:
                monthly_performance = {}
            
            return {
                'total_pnl': total_pnl,
                'buy_pnl': buy_pnl,
                'sell_pnl': sell_pnl,
                'strategy_performance': strategy_performance,
                'monthly_performance': monthly_performance,
                'avg_trade_pnl': total_pnl / len(trades) if trades else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance attribution: {e}")
            return {}
    
    def _calculate_monthly_performance(self, trades: List[Dict[str, Any]], 
                                     dates: List[datetime]) -> Dict[str, float]:
        """Calculate monthly performance breakdown."""
        try:
            monthly_pnl = {}
            
            for i, trade in enumerate(trades):
                if i < len(dates):
                    date = dates[i]
                    month_key = f"{date.year}-{date.month:02d}"
                    pnl = trade.get('pnl', 0)
                    
                    if month_key not in monthly_pnl:
                        monthly_pnl[month_key] = 0.0
                    monthly_pnl[month_key] += pnl
            
            return monthly_pnl
            
        except Exception as e:
            logger.error(f"Error calculating monthly performance: {e}")
            return {}
    
    def _generate_charts(self, trade_analysis: TradeAnalysis, symbol: str) -> Dict[str, str]:
        """Generate charts for the trade analysis."""
        try:
            # Set matplotlib backend to non-interactive to avoid Tkinter issues
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from pathlib import Path
            import time
            
            charts_dir = self.output_dir / 'charts'
            charts_dir.mkdir(exist_ok=True)
            charts = {}
            
            # Generate equity curve chart
            try:
                equity_curve = trade_analysis.equity_curve
                if len(equity_curve.dates) > 1:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(equity_curve.dates, equity_curve.equity_values, linewidth=2, color='blue', label='Portfolio Value')
                    ax.plot(equity_curve.dates, equity_curve.running_max, linewidth=1, color='green', alpha=0.7, label='Running Max')
                    
                    # Format x-axis
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                    
                    ax.set_title(f'Equity Curve - {symbol}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Portfolio Value ($)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    chart_path = charts_dir / f'equity_curve_{symbol}_{int(time.time())}.png'
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    charts['equity_curve'] = str(chart_path)
            except Exception as e:
                logger.warning(f"Could not generate equity curve chart: {e}")
            
            # Generate drawdown chart
            try:
                if len(equity_curve.dates) > 1:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.fill_between(equity_curve.dates, equity_curve.drawdown, 0, alpha=0.3, color='red', label='Drawdown')
                    ax.plot(equity_curve.dates, equity_curve.drawdown, linewidth=1, color='red')
                    
                    # Format x-axis
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                    
                    ax.set_title(f'Drawdown Analysis - {symbol}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Drawdown (%)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    chart_path = charts_dir / f'drawdown_{symbol}_{int(time.time())}.png'
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    charts['drawdown'] = str(chart_path)
            except Exception as e:
                logger.warning(f"Could not generate drawdown chart: {e}")
            
            # Generate monthly returns heatmap
            try:
                trades = trade_analysis.trade_log
                if trades:
                    # Group trades by month
                    monthly_returns = {}
                    for trade in trades:
                        trade_date = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
                        month_key = f"{trade_date.year}-{trade_date.month:02d}"
                        if month_key not in monthly_returns:
                            monthly_returns[month_key] = 0
                        monthly_returns[month_key] += trade.get('pnl', 0)
                    
                    if monthly_returns:
                        # Create heatmap data
                        years = sorted(list(set(k.split('-')[0] for k in monthly_returns.keys())))
                        months = list(range(1, 13))
                        
                        returns_matrix = []
                        for year in years:
                            row = []
                            for month in months:
                                month_key = f"{year}-{month:02d}"
                                row.append(monthly_returns.get(month_key, 0))
                            returns_matrix.append(row)
                        
                        if returns_matrix:
                            fig, ax = plt.subplots(figsize=(12, 8))
                            im = ax.imshow(returns_matrix, cmap='RdYlGn', aspect='auto')
                            ax.set_xticks(range(12))
                            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                            ax.set_yticks(range(len(years)))
                            ax.set_yticklabels(years)
                            ax.set_title(f'Monthly Returns Heatmap - {symbol}')
                            
                            # Add colorbar
                            cbar = plt.colorbar(im, ax=ax)
                            cbar.set_label('PnL ($)')
                            
                            chart_path = charts_dir / f'monthly_heatmap_{symbol}_{int(time.time())}.png'
                            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                            plt.close()
                            charts['monthly_heatmap'] = str(chart_path)
            except Exception as e:
                logger.warning(f"Could not generate monthly heatmap chart: {e}")
            
            return charts
            
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            # Return empty dict instead of failing completely
            return {}
    
    def _generate_summary(self, trade_analysis: TradeAnalysis) -> Dict[str, Any]:
        """Generate a comprehensive summary of the trade analysis."""
        try:
            metrics = trade_analysis.metrics
            
            summary = {
                'overview': {
                    'total_trades': metrics.total_trades,
                    'win_rate': f"{metrics.win_rate:.2%}",
                    'total_pnl': f"${metrics.total_pnl:,.2f}",
                    'total_return': f"{metrics.total_return:.2%}",
                    'annualized_return': f"{metrics.annualized_return:.2%}"
                },
                'risk_metrics': {
                    'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                    'max_drawdown': f"{metrics.max_drawdown:.2%}",
                    'volatility': f"{metrics.volatility:.2%}",
                    'var_95': f"${metrics.var_95:.2f}",
                    'profit_factor': f"{metrics.profit_factor:.2f}"
                },
                'trade_quality': {
                    'avg_gain': f"${metrics.avg_gain:.2f}",
                    'avg_loss': f"${metrics.avg_loss:.2f}",
                    'avg_win_loss_ratio': f"{metrics.avg_win_loss_ratio:.2f}",
                    'largest_win': f"${metrics.largest_win:.2f}",
                    'largest_loss': f"${metrics.largest_loss:.2f}"
                },
                'risk_adjusted_metrics': {
                    'sortino_ratio': f"{metrics.sortino_ratio:.2f}",
                    'calmar_ratio': f"{metrics.calmar_ratio:.2f}",
                    'information_ratio': f"{metrics.information_ratio:.2f}",
                    'treynor_ratio': f"{metrics.treynor_ratio:.2f}"
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {}
    
    def _export_report(self, report_data: Dict[str, Any], report_id: str) -> Dict[str, str]:
        """Export report in multiple formats."""
        try:
            export_paths = {}
            
            # Export formats
            formats = self.report_config.get('export_formats', ['csv', 'pdf', 'html', 'json'])
            
            # JSON export
            if 'json' in formats:
                json_path = self.output_dir / 'json' / f'{report_id}.json'
                with open(json_path, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
                export_paths['json'] = str(json_path)
            
            # CSV export (trade log)
            if 'csv' in formats and 'trade_analysis' in report_data:
                trade_log = report_data['trade_analysis'].get('trade_log', [])
                if trade_log:
                    csv_path = self.output_dir / 'csv' / f'{report_id}_trades.csv'
                    df = pd.DataFrame(trade_log)
                    df.to_csv(csv_path, index=False)
                    export_paths['csv'] = str(csv_path)
            
            # HTML export
            if 'html' in formats:
                html_path = self.output_dir / 'html' / f'{report_id}.html'
                html_content = self._generate_html_report(report_data)
                with open(html_path, 'w') as f:
                    f.write(html_content)
                export_paths['html'] = str(html_path)
            
            # PDF export
            if 'pdf' in formats:
                pdf_path = self.output_dir / 'pdf' / f'{report_id}.pdf'
                # Note: PDF generation would require additional libraries like reportlab or weasyprint
                # For now, we'll create a placeholder
                export_paths['pdf'] = str(pdf_path)
            
            return export_paths
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return {}
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report."""
        try:
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trading Report - {symbol}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                    .chart {{ text-align: center; margin: 20px 0; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Trading Report - {symbol}</h1>
                    <p>Generated: {timestamp}</p>
                    <p>Period: {period}</p>
                    <p>Timeframe: {timeframe}</p>
                </div>
                
                <div class="section">
                    <h2>Performance Summary</h2>
                    <div class="metric">Total Trades: {total_trades}</div>
                    <div class="metric">Win Rate: {win_rate}</div>
                    <div class="metric">Total PnL: {total_pnl}</div>
                    <div class="metric">Sharpe Ratio: {sharpe_ratio}</div>
                    <div class="metric">Max Drawdown: {max_drawdown}</div>
                </div>
                
                <div class="section">
                    <h2>Charts</h2>
                    {charts_html}
                </div>
            </body>
            </html>
            """
            
            # Extract data for template
            symbol = report_data.get('symbol', 'Unknown')
            timestamp = report_data.get('timestamp', 'Unknown')
            period = report_data.get('period', 'Unknown')
            timeframe = report_data.get('timeframe', 'Unknown')
            
            # Extract metrics
            trade_analysis = report_data.get('trade_analysis', {})
            metrics = trade_analysis.get('metrics', {})
            
            total_trades = metrics.get('total_trades', 0)
            win_rate = f"{metrics.get('win_rate', 0):.2%}"
            total_pnl = f"${metrics.get('total_pnl', 0):,.2f}"
            sharpe_ratio = f"{metrics.get('sharpe_ratio', 0):.2f}"
            max_drawdown = f"{metrics.get('max_drawdown', 0):.2%}"
            
            # Generate charts HTML
            charts_html = ""
            charts = report_data.get('charts', {})
            for chart_name, chart_path in charts.items():
                charts_html += f'<div class="chart"><h3>{chart_name.replace("_", " ").title()}</h3><img src="{chart_path}" style="max-width: 100%;"></div>'
            
            return html_template.format(
                symbol=symbol,
                timestamp=timestamp,
                period=period,
                timeframe=timeframe,
                total_trades=total_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                charts_html=charts_html
            )
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return f"<html><body><h1>Error generating report: {e}</h1></body></html>"
    
    def _generate_empty_report(self, report_id: str, symbol: str, timeframe: str, period: str) -> Dict[str, Any]:
        """Generate an empty report when no trades are available."""
        return {
            'report_id': report_id,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'period': period,
            'trade_analysis': asdict(TradeAnalysis(
                trade_log=[],
                equity_curve=self._get_empty_equity_curve(),
                metrics=self._get_empty_metrics(),
                risk_metrics={},
                performance_attribution={}
            )),
            'charts': {},
            'summary': {'message': 'No trades available for analysis'},
            'export_paths': {}
        }
    
    def _generate_error_report(self, report_id: str, error_message: str) -> Dict[str, Any]:
        """Generate an error report."""
        return {
            'report_id': report_id,
            'timestamp': datetime.now().isoformat(),
            'error': error_message,
            'trade_analysis': asdict(TradeAnalysis(
                trade_log=[],
                equity_curve=self._get_empty_equity_curve(),
                metrics=self._get_empty_metrics(),
                risk_metrics={},
                performance_attribution={}
            )),
            'charts': {},
            'summary': {'error': error_message},
            'export_paths': {}
        }
    
    def _get_empty_metrics(self) -> EnhancedTradeMetrics:
        """Get empty metrics structure."""
        return EnhancedTradeMetrics(
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
            total_pnl=0.0, avg_gain=0.0, avg_loss=0.0, avg_trade_duration=0.0,
            sharpe_ratio=0.0, max_drawdown=0.0, profit_factor=0.0, calmar_ratio=0.0,
            sortino_ratio=0.0, var_95=0.0, cvar_95=0.0, total_return=0.0,
            annualized_return=0.0, volatility=0.0, beta=1.0, alpha=0.0,
            information_ratio=0.0, treynor_ratio=0.0, avg_win_loss_ratio=0.0,
            largest_win=0.0, largest_loss=0.0, consecutive_wins=0, consecutive_losses=0,
            avg_trade_return=0.0, risk_reward_ratio=0.0
        )
    
    def _get_empty_equity_curve(self) -> EquityCurveData:
        """Get empty equity curve structure."""
        return EquityCurveData(
            dates=[], equity_values=[], cash_values=[], position_values=[],
            returns=[], cumulative_returns=[], drawdown=[], running_max=[]
        )

# Convenience functions
def generate_unified_report(trade_data: Dict[str, Any],
                           model_data: Optional[Dict[str, Any]] = None,
                           strategy_data: Optional[Dict[str, Any]] = None,
                           symbol: str = "Unknown",
                           timeframe: str = "Unknown",
                           period: str = "Unknown",
                           **kwargs) -> Dict[str, Any]:
    """Generate a unified trade report with all enhancements."""
    reporter = UnifiedTradeReporter(**kwargs)
    return reporter.generate_comprehensive_report(
        trade_data, model_data, strategy_data, symbol, timeframe, period
    )

def export_trade_report(report_data: Dict[str, Any], 
                       export_formats: List[str] = None,
                       output_dir: str = "reports") -> Dict[str, str]:
    """Export trade report in specified formats."""
    reporter = UnifiedTradeReporter(output_dir=output_dir)
    if export_formats:
        reporter.report_config['export_formats'] = export_formats
    return reporter._export_report(report_data, report_data.get('report_id', 'report')) 