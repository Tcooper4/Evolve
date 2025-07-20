"""
Portfolio Management Example

This example demonstrates the portfolio allocation and risk management modules
with real-world scenarios including:
- Multi-strategy portfolio allocation
- Risk monitoring and limit enforcement
- Dynamic rebalancing
- Stress testing
- Performance analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import portfolio modules
from portfolio.allocator import (
    PortfolioAllocator, 
    AllocationStrategy, 
    AssetMetrics,
    create_allocator
)
from portfolio.risk_manager import (
    PortfolioRiskManager,
    RiskLimits,
    create_risk_manager
)

# Import utility functions
from utils.common_helpers import safe_json_save
from utils.cache_utils import cache_result


class PortfolioManagementExample:
    """
    Comprehensive portfolio management example
    """
    
    def __init__(self):
        """Initialize the example"""
        self.allocator = create_allocator()
        self.risk_manager = create_risk_manager()
        
        # Sample portfolio assets
        self.assets = self._create_sample_assets()
        
        # Market data simulation
        self.market_data = self._simulate_market_data()
        
        # Results storage
        self.allocation_results = {}
        self.risk_analysis = {}
        self.simulation_results = {}
    
    def _create_sample_assets(self):
        """Create sample assets with realistic characteristics"""
        return [
            AssetMetrics(
                ticker="AAPL",
                expected_return=0.12,
                volatility=0.20,
                sharpe_ratio=0.5,
                beta=1.1,
                correlation={"TSLA": 0.3, "NVDA": 0.4, "MSFT": 0.6, "GOOGL": 0.5},
                market_cap=2000000000000,
                sector="Technology",
                sentiment_score=0.6
            ),
            AssetMetrics(
                ticker="TSLA",
                expected_return=0.18,
                volatility=0.35,
                sharpe_ratio=0.46,
                beta=1.8,
                correlation={"AAPL": 0.3, "NVDA": 0.5, "MSFT": 0.2, "GOOGL": 0.3},
                market_cap=800000000000,
                sector="Automotive",
                sentiment_score=0.7
            ),
            AssetMetrics(
                ticker="NVDA",
                expected_return=0.15,
                volatility=0.30,
                sharpe_ratio=0.43,
                beta=1.5,
                correlation={"AAPL": 0.4, "TSLA": 0.5, "MSFT": 0.4, "GOOGL": 0.4},
                market_cap=1200000000000,
                sector="Technology",
                sentiment_score=0.8
            ),
            AssetMetrics(
                ticker="MSFT",
                expected_return=0.10,
                volatility=0.18,
                sharpe_ratio=0.44,
                beta=0.9,
                correlation={"AAPL": 0.6, "TSLA": 0.2, "NVDA": 0.4, "GOOGL": 0.7},
                market_cap=2500000000000,
                sector="Technology",
                sentiment_score=0.5
            ),
            AssetMetrics(
                ticker="GOOGL",
                expected_return=0.11,
                volatility=0.22,
                sharpe_ratio=0.41,
                beta=1.0,
                correlation={"AAPL": 0.5, "TSLA": 0.3, "NVDA": 0.4, "MSFT": 0.7},
                market_cap=1800000000000,
                sector="Technology",
                sentiment_score=0.4
            ),
            AssetMetrics(
                ticker="JPM",
                expected_return=0.08,
                volatility=0.25,
                sharpe_ratio=0.24,
                beta=1.2,
                correlation={"AAPL": 0.2, "TSLA": 0.1, "NVDA": 0.2, "MSFT": 0.3, "GOOGL": 0.2},
                market_cap=400000000000,
                sector="Financial",
                sentiment_score=0.3
            ),
            AssetMetrics(
                ticker="JNJ",
                expected_return=0.07,
                volatility=0.16,
                sharpe_ratio=0.31,
                beta=0.7,
                correlation={"AAPL": 0.1, "TSLA": 0.1, "NVDA": 0.1, "MSFT": 0.2, "GOOGL": 0.1, "JPM": 0.2},
                market_cap=350000000000,
                sector="Healthcare",
                sentiment_score=0.6
            )
        ]
    
    def _simulate_market_data(self, days=252):
        """Simulate realistic market data"""
        dates = pd.date_range('2023-01-01', periods=days, freq='D')
        market_data = {}
        
        # Market index (SPY) for beta calculations
        market_returns = np.random.normal(0.0008, 0.015, days)
        market_data['SPY'] = pd.DataFrame({'returns': market_returns}, index=dates)
        
        # Individual asset returns with correlations
        for asset in self.assets:
            # Base return from asset characteristics
            base_return = asset.expected_return / 252  # Daily return
            base_volatility = asset.volatility / np.sqrt(252)  # Daily volatility
            
            # Generate correlated returns
            if asset.ticker == 'AAPL':
                # Start with AAPL
                returns = np.random.normal(base_return, base_volatility, days)
            else:
                # Correlate with AAPL based on correlation matrix
                aapl_returns = market_data['AAPL']['returns']
                correlation = asset.correlation.get('AAPL', 0.3)
                
                # Generate correlated returns
                noise = np.random.normal(0, base_volatility * np.sqrt(1 - correlation**2), days)
                returns = correlation * aapl_returns + noise + base_return
            
            market_data[asset.ticker] = pd.DataFrame({'returns': returns}, index=dates)
        
        return market_data
    
    def run_allocation_analysis(self):
        """Run comprehensive allocation analysis"""
        print("="*60)
        print("PORTFOLIO ALLOCATION ANALYSIS")
        print("="*60)
        
        # Test all allocation strategies
        strategies = [
            AllocationStrategy.EQUAL_WEIGHT,
            AllocationStrategy.MINIMUM_VARIANCE,
            AllocationStrategy.MAXIMUM_SHARPE,
            AllocationStrategy.RISK_PARITY,
            AllocationStrategy.KELLY_CRITERION,
            AllocationStrategy.BLACK_LITTERMAN,
            AllocationStrategy.MEAN_VARIANCE,
            AllocationStrategy.MAXIMUM_DIVERSIFICATION
        ]
        
        results = {}
        
        for strategy in strategies:
            try:
                print(f"\nTesting {strategy.value.replace('_', ' ').title()}...")
                result = self.allocator.allocate_portfolio(self.assets, strategy)
                results[strategy.value] = result
                
                print(f"  Expected Return: {result.expected_return:.3f}")
                print(f"  Expected Volatility: {result.expected_volatility:.3f}")
                print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
                print(f"  Diversification Ratio: {result.diversification_ratio:.3f}")
                print(f"  Largest Position: {max(result.weights.values()):.3f}")
                
            except Exception as e:
                print(f"  Failed: {e}")
        
        self.allocation_results = results
        return results
    
    def compare_allocation_strategies(self):
        """Compare all allocation strategies"""
        if not self.allocation_results:
            self.run_allocation_analysis()
        
        print("\n" + "="*60)
        print("ALLOCATION STRATEGY COMPARISON")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_data = []
        for strategy_name, result in self.allocation_results.items():
            comparison_data.append({
                'Strategy': strategy_name.replace('_', ' ').title(),
                'Expected Return': result.expected_return,
                'Expected Volatility': result.expected_volatility,
                'Sharpe Ratio': result.sharpe_ratio,
                'Diversification Ratio': result.diversification_ratio,
                'Max Position': max(result.weights.values()),
                'Min Position': min(result.weights.values()),
                'Constraints Satisfied': result.constraints_satisfied
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        # Find best strategies by different metrics
        print(f"\nBest Strategy by Sharpe Ratio: {comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax(), 'Strategy']}")
        print(f"Best Strategy by Return: {comparison_df.loc[comparison_df['Expected Return'].idxmax(), 'Strategy']}")
        print(f"Best Strategy by Volatility: {comparison_df.loc[comparison_df['Expected Volatility'].idxmin(), 'Strategy']}")
        print(f"Best Strategy by Diversification: {comparison_df.loc[comparison_df['Diversification Ratio'].idxmax(), 'Strategy']}")
        
        return comparison_df
    
    def run_risk_analysis(self, strategy_name='maximum_sharpe'):
        """Run comprehensive risk analysis"""
        print("\n" + "="*60)
        print("PORTFOLIO RISK ANALYSIS")
        print("="*60)
        
        if not self.allocation_results:
            self.run_allocation_analysis()
        
        # Get allocation result
        allocation_result = self.allocation_results.get(strategy_name)
        if not allocation_result:
            print(f"Strategy {strategy_name} not found")
            return
        
        positions = allocation_result.weights
        print(f"Analyzing {strategy_name.replace('_', ' ').title()} portfolio...")
        
        # Calculate risk metrics
        risk_metrics = self.risk_manager.calculate_portfolio_risk(positions, self.market_data)
        
        print(f"Portfolio Risk Metrics:")
        for metric, value in risk_metrics.items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Check risk limits
        violations = self.risk_manager.check_risk_limits(positions, risk_metrics)
        
        print(f"\nRisk Limit Violations: {len(violations)}")
        for violation in violations:
            print(f"  {violation.risk_metric.value}: {violation.current_value:.4f} > {violation.limit_value:.4f}")
            print(f"    Severity: {violation.severity}")
            print(f"    Action: {violation.action_required}")
        
        # Calculate risk attribution
        risk_attribution = self.risk_manager.calculate_risk_attribution(positions, risk_metrics)
        
        print(f"\nRisk Attribution:")
        for ticker, attribution in risk_attribution.items():
            print(f"  {ticker}:")
            print(f"    Weight: {attribution['weight']:.3f}")
            print(f"    Volatility Contribution: {attribution['volatility_contribution']:.4f}")
            print(f"    VaR Contribution: {attribution['var_contribution']:.4f}")
        
        # Generate risk report
        risk_report = self.risk_manager.get_risk_report(positions, risk_metrics, violations)
        
        self.risk_analysis[strategy_name] = {
            'risk_metrics': risk_metrics,
            'violations': violations,
            'risk_attribution': risk_attribution,
            'risk_report': risk_report
        }
        
        return self.risk_analysis[strategy_name]
    
    def run_portfolio_simulation(self, strategy_name='maximum_sharpe', rebalancing_freq='monthly'):
        """Run portfolio simulation with rebalancing"""
        print(f"\n" + "="*60)
        print(f"PORTFOLIO SIMULATION ({rebalancing_freq.upper()} REBALANCING)")
        print("="*60)
        
        if not self.allocation_results:
            self.run_allocation_analysis()
        
        # Get initial positions
        allocation_result = self.allocation_results.get(strategy_name)
        if not allocation_result:
            print(f"Strategy {strategy_name} not found")
            return
        
        initial_positions = allocation_result.weights
        print(f"Simulating {strategy_name.replace('_', ' ').title()} portfolio...")
        
        # Run simulation
        simulation = self.risk_manager.simulate_portfolio_returns(
            initial_positions, 
            self.market_data, 
            rebalancing_freq
        )
        
        # Calculate performance metrics
        final_value = simulation['portfolio_value'].iloc[-1]
        total_return = simulation['cumulative_return'].iloc[-1]
        max_drawdown = simulation['drawdown'].min()
        volatility = simulation['daily_return'].std() * np.sqrt(252)
        sharpe_ratio = (simulation['daily_return'].mean() * 252) / volatility if volatility > 0 else 0
        
        print(f"Simulation Results:")
        print(f"  Final Portfolio Value: ${final_value:.4f}")
        print(f"  Total Return: {total_return:.2%}")
        print(f"  Annualized Volatility: {volatility:.2%}")
        print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"  Maximum Drawdown: {max_drawdown:.2%}")
        
        # Calculate rolling metrics
        rolling_vol = simulation['daily_return'].rolling(30).std() * np.sqrt(252)
        rolling_sharpe = (simulation['daily_return'].rolling(30).mean() * 252) / rolling_vol
        rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)
        
        self.simulation_results[strategy_name] = {
            'simulation': simulation,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'rolling_vol': rolling_vol,
            'rolling_sharpe': rolling_sharpe
        }
        
        return self.simulation_results[strategy_name]
    
    def run_stress_testing(self, strategy_name='maximum_sharpe'):
        """Run stress testing scenarios"""
        print(f"\n" + "="*60)
        print("STRESS TESTING")
        print("="*60)
        
        if not self.allocation_results:
            self.run_allocation_analysis()
        
        positions = self.allocation_results[strategy_name].weights
        
        # Define stress scenarios
        scenarios = {
            'Market Crash (-30%)': {
                'AAPL': -0.3, 'TSLA': -0.4, 'NVDA': -0.35, 'MSFT': -0.25, 
                'GOOGL': -0.3, 'JPM': -0.2, 'JNJ': -0.15
            },
            'Tech Bubble Burst': {
                'AAPL': -0.4, 'TSLA': -0.5, 'NVDA': -0.45, 'MSFT': -0.35, 
                'GOOGL': -0.4, 'JPM': -0.1, 'JNJ': -0.05
            },
            'Financial Crisis': {
                'AAPL': -0.2, 'TSLA': -0.3, 'NVDA': -0.25, 'MSFT': -0.15, 
                'GOOGL': -0.2, 'JPM': -0.4, 'JNJ': -0.1
            },
            'Tech Rally (+30%)': {
                'AAPL': 0.3, 'TSLA': 0.4, 'NVDA': 0.35, 'MSFT': 0.25, 
                'GOOGL': 0.3, 'JPM': 0.1, 'JNJ': 0.05
            },
            'Volatility Spike': {
                'AAPL': 0.1, 'TSLA': -0.1, 'NVDA': 0.05, 'MSFT': -0.05, 
                'GOOGL': 0.1, 'JPM': -0.15, 'JNJ': 0.02
            }
        }
        
        stress_results = self.risk_manager.stress_test_portfolio(positions, self.market_data, scenarios)
        
        print("Stress Test Results:")
        for scenario_name, metrics in stress_results.items():
            print(f"\n  {scenario_name}:")
            print(f"    Total Return: {metrics['total_return']:.2%}")
            print(f"    Volatility: {metrics['volatility']:.2%}")
            print(f"    Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"    VaR (95%): {metrics['var_95']:.2%}")
            print(f"    Worst Day: {metrics['worst_day']:.2%}")
        
        return stress_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print(f"\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Portfolio Management Analysis', fontsize=16, fontweight='bold')
        
        # 1. Allocation Strategy Comparison
        if self.allocation_results:
            ax1 = axes[0, 0]
            strategies = list(self.allocation_results.keys())
            returns = [self.allocation_results[s].expected_return for s in strategies]
            volatilities = [self.allocation_results[s].expected_volatility for s in strategies]
            
            scatter = ax1.scatter(volatilities, returns, s=100, alpha=0.7)
            ax1.set_xlabel('Expected Volatility')
            ax1.set_ylabel('Expected Return')
            ax1.set_title('Risk-Return Profile by Strategy')
            ax1.grid(True, alpha=0.3)
            
            # Add strategy labels
            for i, strategy in enumerate(strategies):
                ax1.annotate(strategy.replace('_', ' ').title(), 
                           (volatilities[i], returns[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 2. Portfolio Weights Comparison
        if self.allocation_results:
            ax2 = axes[0, 1]
            strategy_names = ['Equal Weight', 'Max Sharpe', 'Risk Parity', 'Kelly']
            selected_strategies = ['equal_weight', 'maximum_sharpe', 'risk_parity', 'kelly_criterion']
            
            x = np.arange(len(self.assets))
            width = 0.2
            
            for i, (strategy, name) in enumerate(zip(selected_strategies, strategy_names)):
                if strategy in self.allocation_results:
                    weights = list(self.allocation_results[strategy].weights.values())
                    ax2.bar(x + i*width, weights, width, label=name, alpha=0.8)
            
            ax2.set_xlabel('Assets')
            ax2.set_ylabel('Weight')
            ax2.set_title('Portfolio Weights by Strategy')
            ax2.set_xticks(x + width * 1.5)
            ax2.set_xticklabels([asset.ticker for asset in self.assets], rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Portfolio Simulation
        if self.simulation_results:
            ax3 = axes[1, 0]
            strategy_name = 'maximum_sharpe'
            if strategy_name in self.simulation_results:
                simulation = self.simulation_results[strategy_name]['simulation']
                ax3.plot(simulation.index, simulation['portfolio_value'], linewidth=2)
                ax3.set_xlabel('Date')
                ax3.set_ylabel('Portfolio Value')
                ax3.set_title('Portfolio Value Over Time')
                ax3.grid(True, alpha=0.3)
        
        # 4. Risk Attribution
        if self.risk_analysis:
            ax4 = axes[1, 1]
            strategy_name = 'maximum_sharpe'
            if strategy_name in self.risk_analysis:
                risk_attribution = self.risk_analysis[strategy_name]['risk_attribution']
                tickers = list(risk_attribution.keys())
                volatility_contrib = [risk_attribution[t]['volatility_contribution'] for t in tickers]
                
                bars = ax4.bar(tickers, volatility_contrib, alpha=0.8)
                ax4.set_xlabel('Assets')
                ax4.set_ylabel('Volatility Contribution')
                ax4.set_title('Risk Attribution')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('portfolio_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'portfolio_analysis.png'")
    
    def generate_report(self):
        """Generate comprehensive portfolio report"""
        print(f"\n" + "="*60)
        print("GENERATING PORTFOLIO REPORT")
        print("="*60)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_summary': {
                'total_assets': len(self.assets),
                'sectors': list(set(asset.sector for asset in self.assets)),
                'total_market_cap': sum(asset.market_cap for asset in self.assets)
            },
            'allocation_analysis': {},
            'risk_analysis': {},
            'simulation_results': {},
            'recommendations': []
        }
        
        # Add allocation results
        for strategy_name, result in self.allocation_results.items():
            report['allocation_analysis'][strategy_name] = {
                'expected_return': result.expected_return,
                'expected_volatility': result.expected_volatility,
                'sharpe_ratio': result.sharpe_ratio,
                'diversification_ratio': result.diversification_ratio,
                'weights': result.weights,
                'constraints_satisfied': result.constraints_satisfied
            }
        
        # Add risk analysis
        for strategy_name, analysis in self.risk_analysis.items():
            report['risk_analysis'][strategy_name] = {
                'risk_metrics': analysis['risk_metrics'],
                'violations_count': len(analysis['violations']),
                'risk_attribution': analysis['risk_attribution']
            }
        
        # Add simulation results
        for strategy_name, results in self.simulation_results.items():
            report['simulation_results'][strategy_name] = {
                'final_value': results['final_value'],
                'total_return': results['total_return'],
                'max_drawdown': results['max_drawdown'],
                'volatility': results['volatility'],
                'sharpe_ratio': results['sharpe_ratio']
            }
        
        # Generate recommendations
        best_sharpe = max(self.allocation_results.keys(), 
                         key=lambda k: self.allocation_results[k].sharpe_ratio)
        best_return = max(self.allocation_results.keys(), 
                         key=lambda k: self.allocation_results[k].expected_return)
        best_vol = min(self.allocation_results.keys(), 
                      key=lambda k: self.allocation_results[k].expected_volatility)
        
        report['recommendations'] = [
            f"Best Sharpe Ratio Strategy: {best_sharpe.replace('_', ' ').title()}",
            f"Best Return Strategy: {best_return.replace('_', ' ').title()}",
            f"Lowest Volatility Strategy: {best_vol.replace('_', ' ').title()}",
            "Consider dynamic rebalancing for risk management",
            "Monitor sector concentration and adjust if needed",
            "Implement stop-loss mechanisms for drawdown protection"
        ]
        
        # Save report
        safe_json_save('portfolio_management_report.json', report)
        print("Portfolio report saved as 'portfolio_management_report.json'")
        
        return report
    
    def run_complete_analysis(self):
        """Run complete portfolio analysis"""
        print("Starting comprehensive portfolio analysis...")
        
        # 1. Allocation analysis
        self.run_allocation_analysis()
        
        # 2. Strategy comparison
        self.compare_allocation_strategies()
        
        # 3. Risk analysis for best strategy
        self.run_risk_analysis('maximum_sharpe')
        
        # 4. Portfolio simulation
        self.run_portfolio_simulation('maximum_sharpe', 'monthly')
        
        # 5. Stress testing
        self.run_stress_testing('maximum_sharpe')
        
        # 6. Create visualizations
        self.create_visualizations()
        
        # 7. Generate report
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Files generated:")
        print("- portfolio_analysis.png (visualizations)")
        print("- portfolio_management_report.json (detailed report)")
        
        return report


def main():
    """Main function to run the portfolio management example"""
    print("Portfolio Management Example")
    print("="*60)
    
    # Create example instance
    example = PortfolioManagementExample()
    
    # Run complete analysis
    report = example.run_complete_analysis()
    
    # Print summary
    print("\nSUMMARY:")
    print(f"- Analyzed {len(example.assets)} assets across {len(example.allocation_results)} strategies")
    print(f"- Best Sharpe Ratio: {max(r.sharpe_ratio for r in example.allocation_results.values()):.3f}")
    print(f"- Best Expected Return: {max(r.expected_return for r in example.allocation_results.values()):.3f}")
    print(f"- Lowest Volatility: {min(r.expected_volatility for r in example.allocation_results.values()):.3f}")
    
    return report


if __name__ == "__main__":
    main()
