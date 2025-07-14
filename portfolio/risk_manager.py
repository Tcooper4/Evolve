"""
Portfolio Risk Manager

This module implements comprehensive risk management for portfolios:
- Risk limit enforcement (max drawdown, max exposure, volatility targeting)
- Portfolio simulation and backtesting
- Dynamic rebalancing logic
- Risk metrics calculation
- Position sizing and leverage control
- Stress testing and scenario analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Local imports
from utils.cache_utils import cache_result
from utils.common_helpers import safe_json_save, load_config
from pathlib import Path
from enum import Enum


class RiskMetric(Enum):
    """Risk metrics for monitoring"""
    VAR = "value_at_risk"
    CVAR = "conditional_var"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    CORRELATION = "correlation"
    EXPOSURE = "exposure"
    LEVERAGE = "leverage"


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_drawdown: float = 0.15  # 15% maximum drawdown
    max_exposure: float = 0.3   # 30% maximum single position exposure
    max_leverage: float = 2.0   # 2x maximum leverage
    target_volatility: float = 0.15  # 15% target volatility
    var_limit: float = 0.02     # 2% daily VaR limit
    max_correlation: float = 0.7  # 70% maximum correlation
    sector_limit: float = 0.4   # 40% maximum sector exposure
    liquidity_limit: float = 0.1  # 10% maximum illiquid position


@dataclass
class PortfolioState:
    """Current portfolio state"""
    timestamp: str
    positions: Dict[str, float]  # ticker -> weight
    portfolio_value: float
    cash: float
    leverage: float
    volatility: float
    drawdown: float
    var_95: float
    exposure_concentration: float
    sector_exposure: Dict[str, float]
    risk_metrics: Dict[str, float]


@dataclass
class RiskViolation:
    """Risk limit violation"""
    timestamp: str
    risk_metric: RiskMetric
    current_value: float
    limit_value: float
    severity: str  # 'warning', 'critical'
    action_required: str
    affected_positions: List[str]


@dataclass
class RebalancingAction:
    """Rebalancing action to take"""
    action_type: str  # 'buy', 'sell', 'rebalance', 'hedge'
    ticker: str
    current_weight: float
    target_weight: float
    trade_amount: float
    priority: int  # 1-5, 5 being highest
    reason: str


class PortfolioRiskManager:
    """
    Comprehensive portfolio risk management system
    """
    
    def __init__(self, config_path: str = "config/app_config.yaml"):
        # Load configuration
        self.config = load_config(config_path)
        self.risk_config = self.config.get('risk_management', {})
        
        # Risk limits
        self.risk_limits = RiskLimits(
            max_drawdown=self.risk_config.get('max_drawdown', 0.15),
            max_exposure=self.risk_config.get('max_exposure', 0.3),
            max_leverage=self.risk_config.get('max_leverage', 2.0),
            target_volatility=self.risk_config.get('target_volatility', 0.15),
            var_limit=self.risk_config.get('var_limit', 0.02),
            max_correlation=self.risk_config.get('max_correlation', 0.7),
            sector_limit=self.risk_config.get('sector_limit', 0.4),
            liquidity_limit=self.risk_config.get('liquidity_limit', 0.1)
        )
        
        # Portfolio history
        self.portfolio_history: List[PortfolioState] = []
        self.violations_history: List[RiskViolation] = []
        
        # Risk calculation parameters
        self.var_confidence = self.risk_config.get('var_confidence', 0.95)
        self.lookback_period = self.risk_config.get('lookback_period', 252)  # 1 year
        self.rebalancing_frequency = self.risk_config.get('rebalancing_frequency', 'daily')
        
        # Market data cache
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        
        # Sector classifications
        self.sector_classifications = self._load_sector_classifications()
    
    def _load_sector_classifications(self) -> Dict[str, str]:
        """Load sector classifications for assets"""
        # This would typically load from a database or file
        # For now, return a sample mapping
        return {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'TSLA': 'Automotive',
            'NVDA': 'Technology',
            'AMD': 'Technology',
            'META': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'JPM': 'Financial',
            'JNJ': 'Healthcare'
        }
    
    def calculate_portfolio_risk(self, 
                               positions: Dict[str, float],
                               market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio risk metrics
        """
        if not positions:
            return {}
        
        # Calculate returns for each asset
        returns_data = {}
        for ticker, weight in positions.items():
            if ticker in market_data and not market_data[ticker].empty:
                returns_data[ticker] = market_data[ticker]['returns'].dropna()
        
        if not returns_data:
            return {}
        
        # Portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(positions, returns_data)
        
        # Risk metrics
        risk_metrics = {}
        
        # Volatility
        risk_metrics['volatility'] = portfolio_returns.std() * np.sqrt(252)  # Annualized
        
        # Value at Risk
        risk_metrics['var_95'] = np.percentile(portfolio_returns, 5)
        risk_metrics['var_99'] = np.percentile(portfolio_returns, 1)
        
        # Conditional Value at Risk (Expected Shortfall)
        var_95 = risk_metrics['var_95']
        risk_metrics['cvar_95'] = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Beta (if market data available)
        if 'SPY' in returns_data:
            market_returns = returns_data['SPY']
            # Align dates
            common_dates = portfolio_returns.index.intersection(market_returns.index)
            if len(common_dates) > 30:
                portfolio_aligned = portfolio_returns.loc[common_dates]
                market_aligned = market_returns.loc[common_dates]
                covariance = np.cov(portfolio_aligned, market_aligned)[0, 1]
                market_variance = np.var(market_aligned)
                risk_metrics['beta'] = covariance / market_variance if market_variance > 0 else 1.0
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        risk_metrics['max_drawdown'] = drawdown.min()
        risk_metrics['current_drawdown'] = drawdown.iloc[-1]
        
        # Concentration metrics
        risk_metrics['exposure_concentration'] = max(positions.values()) if positions else 0
        risk_metrics['herfindahl_index'] = sum(w**2 for w in positions.values())
        
        # Sector concentration
        sector_exposure = self._calculate_sector_exposure(positions)
        risk_metrics['max_sector_exposure'] = max(sector_exposure.values()) if sector_exposure else 0
        
        # Correlation risk
        if len(returns_data) > 1:
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            # Average correlation
            n_assets = len(returns_data)
            total_correlation = 0
            count = 0
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    total_correlation += abs(correlation_matrix.iloc[i, j])
                    count += 1
            risk_metrics['avg_correlation'] = total_correlation / count if count > 0 else 0
        
        return risk_metrics
    
    def _calculate_portfolio_returns(self, 
                                   positions: Dict[str, float],
                                   returns_data: Dict[str, pd.Series]) -> pd.Series:
        """Calculate portfolio returns from asset returns"""
        # Align all return series
        all_returns = []
        for ticker, returns in returns_data.items():
            if ticker in positions:
                weighted_returns = returns * positions[ticker]
                all_returns.append(weighted_returns)
        
        if not all_returns:
            return pd.Series(dtype=float)
        
        # Sum weighted returns
        portfolio_returns = sum(all_returns)
        return portfolio_returns
    
    def _calculate_sector_exposure(self, positions: Dict[str, float]) -> Dict[str, float]:
        """Calculate sector exposure"""
        sector_exposure = {}
        
        for ticker, weight in positions.items():
            sector = self.sector_classifications.get(ticker, 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
        
        return sector_exposure
    
    def check_risk_limits(self, 
                         positions: Dict[str, float],
                         risk_metrics: Dict[str, float]) -> List[RiskViolation]:
        """
        Check if current portfolio violates risk limits
        """
        violations = []
        timestamp = datetime.now().isoformat()
        
        # Check drawdown limit
        current_drawdown = abs(risk_metrics.get('current_drawdown', 0))
        if current_drawdown > self.risk_limits.max_drawdown:
            violations.append(RiskViolation(
                timestamp=timestamp,
                risk_metric=RiskMetric.DRAWDOWN,
                current_value=current_drawdown,
                limit_value=self.risk_limits.max_drawdown,
                severity='critical' if current_drawdown > self.risk_limits.max_drawdown * 1.5 else 'warning',
                action_required='Reduce positions or hedge portfolio',
                affected_positions=list(positions.keys())
            ))
        
        # Check exposure limit
        max_exposure = max(positions.values()) if positions else 0
        if max_exposure > self.risk_limits.max_exposure:
            max_exposure_ticker = max(positions, key=positions.get)
            violations.append(RiskViolation(
                timestamp=timestamp,
                risk_metric=RiskMetric.EXPOSURE,
                current_value=max_exposure,
                limit_value=self.risk_limits.max_exposure,
                severity='critical' if max_exposure > self.risk_limits.max_exposure * 1.2 else 'warning',
                action_required=f'Reduce position in {max_exposure_ticker}',
                affected_positions=[max_exposure_ticker]
            ))
        
        # Check volatility limit
        current_volatility = risk_metrics.get('volatility', 0)
        if current_volatility > self.risk_limits.target_volatility * 1.2:
            violations.append(RiskViolation(
                timestamp=timestamp,
                risk_metric=RiskMetric.VOLATILITY,
                current_value=current_volatility,
                limit_value=self.risk_limits.target_volatility,
                severity='warning',
                action_required='Rebalance to reduce portfolio volatility',
                affected_positions=list(positions.keys())
            ))
        
        # Check VaR limit
        var_95 = abs(risk_metrics.get('var_95', 0))
        if var_95 > self.risk_limits.var_limit:
            violations.append(RiskViolation(
                timestamp=timestamp,
                risk_metric=RiskMetric.VAR,
                current_value=var_95,
                limit_value=self.risk_limits.var_limit,
                severity='critical' if var_95 > self.risk_limits.var_limit * 1.5 else 'warning',
                action_required='Reduce portfolio risk or add hedging',
                affected_positions=list(positions.keys())
            ))
        
        # Check sector concentration
        sector_exposure = self._calculate_sector_exposure(positions)
        max_sector_exposure = max(sector_exposure.values()) if sector_exposure else 0
        if max_sector_exposure > self.risk_limits.sector_limit:
            max_sector = max(sector_exposure, key=sector_exposure.get)
            sector_tickers = [ticker for ticker, weight in positions.items() 
                            if self.sector_classifications.get(ticker) == max_sector]
            violations.append(RiskViolation(
                timestamp=timestamp,
                risk_metric=RiskMetric.EXPOSURE,
                current_value=max_sector_exposure,
                limit_value=self.risk_limits.sector_limit,
                severity='warning',
                action_required=f'Reduce exposure to {max_sector} sector',
                affected_positions=sector_tickers
            ))
        
        return violations
    
    def generate_rebalancing_actions(self,
                                   current_positions: Dict[str, float],
                                   target_positions: Dict[str, float],
                                   risk_violations: List[RiskViolation]) -> List[RebalancingAction]:
        """
        Generate rebalancing actions to address risk violations and move toward target
        """
        actions = []
        
        # Handle risk violations first (higher priority)
        for violation in risk_violations:
            if violation.risk_metric == RiskMetric.EXPOSURE:
                # Reduce over-exposed positions
                for ticker in violation.affected_positions:
                    if ticker in current_positions:
                        current_weight = current_positions[ticker]
                        target_weight = min(current_weight * 0.8, self.risk_limits.max_exposure)
                        
                        if target_weight < current_weight:
                            actions.append(RebalancingAction(
                                action_type='sell',
                                ticker=ticker,
                                current_weight=current_weight,
                                target_weight=target_weight,
                                trade_amount=current_weight - target_weight,
                                priority=5 if violation.severity == 'critical' else 4,
                                reason=f"Risk violation: {violation.risk_metric.value}"
                            ))
            
            elif violation.risk_metric == RiskMetric.DRAWDOWN:
                # Reduce all positions proportionally
                reduction_factor = 0.9  # Reduce by 10%
                for ticker, current_weight in current_positions.items():
                    target_weight = current_weight * reduction_factor
                    actions.append(RebalancingAction(
                        action_type='sell',
                        ticker=ticker,
                        current_weight=current_weight,
                        target_weight=target_weight,
                        trade_amount=current_weight - target_weight,
                        priority=5,
                        reason="Drawdown limit exceeded"
                    ))
        
        # Regular rebalancing toward target positions
        all_tickers = set(current_positions.keys()) | set(target_positions.keys())
        
        for ticker in all_tickers:
            current_weight = current_positions.get(ticker, 0)
            target_weight = target_positions.get(ticker, 0)
            
            # Only rebalance if difference is significant
            if abs(current_weight - target_weight) > 0.01:  # 1% threshold
                if target_weight > current_weight:
                    action_type = 'buy'
                    trade_amount = target_weight - current_weight
                else:
                    action_type = 'sell'
                    trade_amount = current_weight - target_weight
                
                actions.append(RebalancingAction(
                    action_type=action_type,
                    ticker=ticker,
                    current_weight=current_weight,
                    target_weight=target_weight,
                    trade_amount=trade_amount,
                    priority=2,
                    reason="Regular rebalancing"
                ))
        
        # Sort by priority (highest first)
        actions.sort(key=lambda x: x.priority, reverse=True)
        
        return actions
    
    def simulate_portfolio_returns(self,
                                 initial_positions: Dict[str, float],
                                 market_data: Dict[str, pd.DataFrame],
                                 rebalancing_frequency: str = 'monthly') -> pd.DataFrame:
        """
        Simulate portfolio returns with rebalancing
        """
        if not market_data:
            return pd.DataFrame()
        
        # Initialize simulation
        positions = initial_positions.copy()
        portfolio_values = []
        rebalancing_dates = []
        
        # Get common date range
        all_dates = []
        for ticker, data in market_data.items():
            if not data.empty:
                all_dates.extend(data.index.tolist())
        
        if not all_dates:
            return pd.DataFrame()
        
        start_date = min(all_dates)
        end_date = max(all_dates)
        
        # Generate rebalancing dates
        if rebalancing_frequency == 'daily':
            rebalancing_dates = pd.date_range(start_date, end_date, freq='D')
        elif rebalancing_frequency == 'weekly':
            rebalancing_dates = pd.date_range(start_date, end_date, freq='W')
        elif rebalancing_frequency == 'monthly':
            rebalancing_dates = pd.date_range(start_date, end_date, freq='M')
        else:
            rebalancing_dates = [start_date, end_date]
        
        # Simulate portfolio
        current_date = start_date
        portfolio_value = 1.0  # Start with $1
        
        while current_date <= end_date:
            # Calculate daily returns
            daily_return = 0
            for ticker, weight in positions.items():
                if ticker in market_data and current_date in market_data[ticker].index:
                    asset_return = market_data[ticker].loc[current_date, 'returns']
                    daily_return += weight * asset_return
            
            # Update portfolio value
            portfolio_value *= (1 + daily_return)
            
            # Record portfolio state
            portfolio_values.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'daily_return': daily_return,
                'positions': positions.copy()
            })
            
            # Rebalance if needed
            if current_date in rebalancing_dates:
                # Calculate risk metrics
                risk_metrics = self.calculate_portfolio_risk(positions, market_data)
                
                # Check for violations
                violations = self.check_risk_limits(positions, risk_metrics)
                
                # Generate rebalancing actions
                actions = self.generate_rebalancing_actions(positions, initial_positions, violations)
                
                # Apply rebalancing actions
                for action in actions:
                    if action.ticker in positions:
                        if action.action_type == 'sell':
                            positions[action.ticker] = max(0, positions[action.ticker] - action.trade_amount)
                        elif action.action_type == 'buy':
                            positions[action.ticker] = min(1, positions[action.ticker] + action.trade_amount)
                
                # Normalize weights
                total_weight = sum(positions.values())
                if total_weight > 0:
                    positions = {k: v/total_weight for k, v in positions.items()}
            
            # Move to next date
            current_date += timedelta(days=1)
        
        # Create results DataFrame
        results_df = pd.DataFrame(portfolio_values)
        results_df.set_index('date', inplace=True)
        
        # Calculate additional metrics
        results_df['cumulative_return'] = results_df['portfolio_value'] - 1
        results_df['drawdown'] = self._calculate_drawdown(results_df['portfolio_value'])
        
        return results_df
    
    def _calculate_drawdown(self, portfolio_values: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        return drawdown
    
    def stress_test_portfolio(self,
                            positions: Dict[str, float],
                            market_data: Dict[str, pd.DataFrame],
                            scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Stress test portfolio under various scenarios
        """
        results = {}
        
        for scenario_name, scenario_shocks in scenarios.items():
            # Apply scenario shocks to returns
            shocked_returns = {}
            for ticker, returns in market_data.items():
                if ticker in scenario_shocks:
                    shock = scenario_shocks[ticker]
                    shocked_returns[ticker] = returns * (1 + shock)
                else:
                    shocked_returns[ticker] = returns
            
            # Calculate portfolio performance under scenario
            portfolio_returns = self._calculate_portfolio_returns(positions, shocked_returns)
            
            if not portfolio_returns.empty:
                scenario_metrics = {
                    'total_return': (1 + portfolio_returns).prod() - 1,
                    'volatility': portfolio_returns.std() * np.sqrt(252),
                    'max_drawdown': self._calculate_drawdown((1 + portfolio_returns).cumprod()).min(),
                    'var_95': np.percentile(portfolio_returns, 5),
                    'worst_day': portfolio_returns.min()
                }
                
                results[scenario_name] = scenario_metrics
        
        return results
    
    def optimize_position_sizing(self,
                               positions: Dict[str, float],
                               risk_metrics: Dict[str, float],
                               target_volatility: float) -> Dict[str, float]:
        """
        Optimize position sizes to achieve target volatility
        """
        current_volatility = risk_metrics.get('volatility', 0)
        
        if current_volatility <= 0:
            return positions
        
        # Calculate scaling factor
        scaling_factor = target_volatility / current_volatility
        
        # Apply scaling factor to positions
        optimized_positions = {}
        for ticker, weight in positions.items():
            optimized_positions[ticker] = weight * scaling_factor
        
        # Normalize weights
        total_weight = sum(optimized_positions.values())
        if total_weight > 0:
            optimized_positions = {k: v/total_weight for k, v in optimized_positions.items()}
        
        return optimized_positions
    
    def calculate_risk_attribution(self,
                                 positions: Dict[str, float],
                                 risk_metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Calculate risk attribution for each position
        """
        attribution = {}
        
        for ticker, weight in positions.items():
            # Simplified risk attribution
            # In practice, this would use more sophisticated calculations
            attribution[ticker] = {
                'weight': weight,
                'volatility_contribution': weight * risk_metrics.get('volatility', 0),
                'var_contribution': weight * abs(risk_metrics.get('var_95', 0)),
                'beta_contribution': weight * risk_metrics.get('beta', 1.0)
            }
        
        return attribution
    
    def get_risk_report(self,
                       positions: Dict[str, float],
                       risk_metrics: Dict[str, float],
                       violations: List[RiskViolation]) -> Dict[str, Any]:
        """
        Generate comprehensive risk report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_summary': {
                'total_positions': len(positions),
                'total_weight': sum(positions.values()),
                'largest_position': max(positions.values()) if positions else 0,
                'largest_position_ticker': max(positions, key=positions.get) if positions else None
            },
            'risk_metrics': risk_metrics,
            'risk_limits': {
                'max_drawdown': self.risk_limits.max_drawdown,
                'max_exposure': self.risk_limits.max_exposure,
                'target_volatility': self.risk_limits.target_volatility,
                'var_limit': self.risk_limits.var_limit
            },
            'violations': [
                {
                    'metric': violation.risk_metric.value,
                    'current_value': violation.current_value,
                    'limit_value': violation.limit_value,
                    'severity': violation.severity,
                    'action_required': violation.action_required
                }
                for violation in violations
            ],
            'sector_exposure': self._calculate_sector_exposure(positions),
            'risk_attribution': self.calculate_risk_attribution(positions, risk_metrics)
        }
        
        return report
    
    def save_risk_report(self, report: Dict[str, Any], filename: Optional[str] = None):
        """Save risk report to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"risk_report_{timestamp}.json"
        
        filepath = Path("reports/risk") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        safe_json_save(str(filepath), report)


# Convenience functions
def create_risk_manager(config_path: str = "config/app_config.yaml") -> PortfolioRiskManager:
    """Create a portfolio risk manager instance"""
    return PortfolioRiskManager(config_path)


def calculate_portfolio_risk(positions: Dict[str, float],
                           market_data: Dict[str, pd.DataFrame],
                           config_path: str = "config/app_config.yaml") -> Dict[str, float]:
    """Quick function to calculate portfolio risk"""
    risk_manager = PortfolioRiskManager(config_path)
    return risk_manager.calculate_portfolio_risk(positions, market_data)


def check_risk_limits(positions: Dict[str, float],
                     risk_metrics: Dict[str, float],
                     config_path: str = "config/app_config.yaml") -> List[RiskViolation]:
    """Quick function to check risk limits"""
    risk_manager = PortfolioRiskManager(config_path)
    return risk_manager.check_risk_limits(positions, risk_metrics)


if __name__ == "__main__":
    # Example usage
    risk_manager = PortfolioRiskManager()
    
    # Sample positions
    positions = {
        'AAPL': 0.3,
        'TSLA': 0.25,
        'NVDA': 0.25,
        'MSFT': 0.2
    }
    
    # Sample market data (simplified)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    market_data = {}
    
    for ticker in positions.keys():
        # Generate random returns
        returns = np.random.normal(0.001, 0.02, len(dates))
        market_data[ticker] = pd.DataFrame({
            'returns': returns
        }, index=dates)
    
    # Calculate risk metrics
    risk_metrics = risk_manager.calculate_portfolio_risk(positions, market_data)
    
    print("Portfolio Risk Metrics:")
    for metric, value in risk_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Check risk limits
    violations = risk_manager.check_risk_limits(positions, risk_metrics)
    
    print(f"\nRisk Violations: {len(violations)}")
    for violation in violations:
        print(f"  {violation.risk_metric.value}: {violation.current_value:.4f} > {violation.limit_value:.4f}")
        print(f"    Severity: {violation.severity}")
        print(f"    Action: {violation.action_required}")
    
    # Generate rebalancing actions
    target_positions = {'AAPL': 0.25, 'TSLA': 0.25, 'NVDA': 0.25, 'MSFT': 0.25}
    actions = risk_manager.generate_rebalancing_actions(positions, target_positions, violations)
    
    print(f"\nRebalancing Actions: {len(actions)}")
    for action in actions:
        print(f"  {action.action_type.upper()} {action.ticker}: {action.current_weight:.3f} → {action.target_weight:.3f}")
        print(f"    Priority: {action.priority}, Reason: {action.reason}")
    
    # Simulate portfolio returns
    simulation = risk_manager.simulate_portfolio_returns(positions, market_data, 'monthly')
    
    print(f"\nSimulation Results:")
    print(f"  Final Portfolio Value: {simulation['portfolio_value'].iloc[-1]:.4f}")
    print(f"  Total Return: {simulation['cumulative_return'].iloc[-1]:.4f}")
    print(f"  Max Drawdown: {simulation['drawdown'].min():.4f}")
    
    # Generate risk report
    report = risk_manager.get_risk_report(positions, risk_metrics, violations)
    print(f"\nRisk Report Generated: {len(report)} sections") 