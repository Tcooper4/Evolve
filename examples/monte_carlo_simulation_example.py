"""
Monte Carlo Simulation Example

This script demonstrates how to use the Monte Carlo simulation functionality
for portfolio backtesting and risk analysis.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trading.backtesting.monte_carlo import (
    MonteCarloSimulator,
    MonteCarloConfig,
    run_monte_carlo_analysis
)


def generate_realistic_returns(
    n_days: int = 252,
    mean_return: float = 0.0005,
    volatility: float = 0.02,
    seed: int = 42
) -> pd.Series:
    """Generate realistic returns with some market-like characteristics."""
    np.random.seed(seed)
    
    # Generate base returns
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    base_returns = np.random.normal(mean_return, volatility, n_days)
    
    # Add some volatility clustering (GARCH-like effect)
    for i in range(1, n_days):
        if abs(base_returns[i-1]) > 2 * volatility:  # High volatility period
            base_returns[i] *= 1.5
    
    # Add some mean reversion
    for i in range(1, n_days):
        if base_returns[i-1] > 2 * volatility:
            base_returns[i] -= 0.1 * volatility
        elif base_returns[i-1] < -2 * volatility:
            base_returns[i] += 0.1 * volatility
    
    returns = pd.Series(base_returns, index=dates)
    return returns


def demonstrate_basic_simulation():
    """Demonstrate basic Monte Carlo simulation."""
    print("ðŸŽ² Basic Monte Carlo Simulation")
    print("=" * 50)
    
    # Generate sample returns
    returns = generate_realistic_returns(n_days=252, mean_return=0.0005, volatility=0.02)
    
    print(f"Generated {len(returns)} days of returns")
    print(f"Mean return: {returns.mean():.4f}")
    print(f"Volatility: {returns.std():.4f}")
    
    # Run simulation
    results = run_monte_carlo_analysis(
        returns=returns,
        initial_capital=10000.0,
        n_simulations=1000,
        bootstrap_method="historical",
        plot_results=False
    )
    
    # Display results
    stats = results['summary_statistics']
    print(f"\nðŸ“Š Simulation Results:")
    print(f"Mean Final Value: ${stats['mean_final_value']:,.2f}")
    print(f"5th Percentile: ${stats['final_p5']:,.2f}")
    print(f"95th Percentile: ${stats['final_p95']:,.2f}")
    print(f"Probability of Loss: {stats['probability_of_loss']:.2%}")
    print(f"95% VaR: {stats['var_95']:.2%}")
    
    return results


def demonstrate_bootstrap_methods():
    """Demonstrate different bootstrap methods."""
    print("\nðŸ”„ Bootstrap Methods Comparison")
    print("=" * 50)
    
    # Generate returns
    returns = generate_realistic_returns(n_days=252)
    
    # Test different bootstrap methods
    methods = ["historical", "block", "parametric"]
    
    for method in methods:
        print(f"\nðŸ“ˆ Testing {method.upper()} bootstrap:")
        
        config = MonteCarloConfig(
            n_simulations=500,  # Fewer for faster comparison
            bootstrap_method=method,
            block_size=20 if method == "block" else 20
        )
        
        simulator = MonteCarloSimulator(config)
        simulator.simulate_portfolio_paths(returns, 10000.0, 500)
        simulator.calculate_percentiles()
        
        stats = simulator.get_summary_statistics()
        print(f"  Mean Final Value: ${stats['mean_final_value']:,.2f}")
        print(f"  Volatility: {stats['std_total_return']:.2%}")
        print(f"  95% VaR: {stats['var_95']:.2%}")


def demonstrate_confidence_intervals():
    """Demonstrate confidence interval calculations."""
    print("\nðŸ“Š Confidence Intervals Analysis")
    print("=" * 50)
    
    # Generate returns
    returns = generate_realistic_returns(n_days=252)
    
    # Test different confidence levels
    confidence_levels = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    
    config = MonteCarloConfig(
        n_simulations=1000,
        confidence_levels=confidence_levels
    )
    
    simulator = MonteCarloSimulator(config)
    simulator.simulate_portfolio_paths(returns, 10000.0, 1000)
    percentiles = simulator.calculate_percentiles(confidence_levels)
    
    print("Final Portfolio Values by Percentile:")
    final_values = percentiles.iloc[-1]
    
    for level in confidence_levels:
        percentile_key = f'P{int(level * 100)}'
        if percentile_key in final_values:
            value = final_values[percentile_key]
            return_pct = (value - 10000.0) / 10000.0
            print(f"  {percentile_key:>4}: ${value:>8,.2f} ({return_pct:>+6.2%})")


def demonstrate_risk_analysis():
    """Demonstrate comprehensive risk analysis."""
    print("\nâš ï¸ Risk Analysis")
    print("=" * 50)
    
    # Generate returns with different characteristics
    scenarios = [
        ("Conservative", 0.0003, 0.015),  # Low return, low volatility
        ("Moderate", 0.0005, 0.020),      # Medium return, medium volatility
        ("Aggressive", 0.0008, 0.030),    # High return, high volatility
    ]
    
    for scenario_name, mean_ret, vol in scenarios:
        print(f"\nðŸ“ˆ {scenario_name} Portfolio:")
        
        returns = generate_realistic_returns(n_days=252, mean_return=mean_ret, volatility=vol)
        
        results = run_monte_carlo_analysis(
            returns=returns,
            initial_capital=10000.0,
            n_simulations=1000,
            plot_results=False
        )
        
        stats = results['summary_statistics']
        
        print(f"  Expected Return: {mean_ret:.2%}")
        print(f"  Volatility: {vol:.2%}")
        print(f"  Mean Final Value: ${stats['mean_final_value']:,.2f}")
        print(f"  Probability of Loss: {stats['probability_of_loss']:.2%}")
        print(f"  95% VaR: {stats['var_95']:.2%}")
        print(f"  Max Drawdown (Median): {stats.get('max_drawdown_p50', 0):.2%}")


def demonstrate_visualization():
    """Demonstrate visualization capabilities."""
    print("\nðŸŽ¨ Visualization Demo")
    print("=" * 50)
    
    # Generate returns
    returns = generate_realistic_returns(n_days=252)
    
    # Create simulator
    config = MonteCarloConfig(n_simulations=1000)
    simulator = MonteCarloSimulator(config)
    
    # Run simulation
    simulator.simulate_portfolio_paths(returns, 10000.0, 1000)
    simulator.calculate_percentiles()
    
    # Create visualization
    fig = simulator.plot_simulation_results(
        figsize=(14, 10),
        show_paths=True,
        n_paths_to_show=100,
        alpha_paths=0.1,
        confidence_bands=True,
        save_path="monte_carlo_simulation.png"
    )
    
    print("âœ… Visualization created and saved as 'monte_carlo_simulation.png'")
    
    # Display some key statistics
    stats = simulator.get_summary_statistics()
    print(f"\nðŸ“Š Key Statistics:")
    print(f"  Mean Final Value: ${stats['mean_final_value']:,.2f}")
    print(f"  Best Case: ${stats['max_final_value']:,.2f}")
    print(f"  Worst Case: ${stats['min_final_value']:,.2f}")
    print(f"  Volatility: {stats['std_total_return']:.2%}")


def demonstrate_performance_comparison():
    """Demonstrate performance comparison between different strategies."""
    print("\nðŸ† Performance Comparison")
    print("=" * 50)
    
    # Generate different return series (simulating different strategies)
    strategies = {
        "Buy and Hold": generate_realistic_returns(mean_return=0.0005, volatility=0.02),
        "Conservative": generate_realistic_returns(mean_return=0.0003, volatility=0.015),
        "Aggressive": generate_realistic_returns(mean_return=0.0008, volatility=0.030),
    }
    
    results_comparison = {}
    
    for strategy_name, returns in strategies.items():
        print(f"\nðŸ“ˆ Analyzing {strategy_name}:")
        
        results = run_monte_carlo_analysis(
            returns=returns,
            initial_capital=10000.0,
            n_simulations=500,
            plot_results=False
        )
        
        stats = results['summary_statistics']
        results_comparison[strategy_name] = {
            'mean_final_value': stats['mean_final_value'],
            'mean_total_return': stats['mean_total_return'],
            'std_total_return': stats['std_total_return'],
            'var_95': stats['var_95'],
            'probability_of_loss': stats['probability_of_loss']
        }
        
        print(f"  Mean Return: {stats['mean_total_return']:.2%}")
        print(f"  Volatility: {stats['std_total_return']:.2%}")
        print(f"  Sharpe Ratio: {stats['mean_total_return'] / stats['std_total_return']:.2f}")
        print(f"  95% VaR: {stats['var_95']:.2%}")
    
    # Create comparison table
    print(f"\nðŸ“Š Strategy Comparison:")
    comparison_df = pd.DataFrame(results_comparison).T
    comparison_df['Sharpe_Ratio'] = comparison_df['mean_total_return'] / comparison_df['std_total_return']
    
    print(comparison_df.round(4))


def demonstrate_parameter_sensitivity():
    """Demonstrate sensitivity to different parameters."""
    print("\nðŸ”¬ Parameter Sensitivity Analysis")
    print("=" * 50)
    
    # Test different simulation parameters
    n_simulations_list = [100, 500, 1000, 2000]
    initial_capitals = [5000, 10000, 20000, 50000]
    
    returns = generate_realistic_returns(n_days=252)
    
    print("Testing different numbers of simulations:")
    for n_sim in n_simulations_list:
        results = run_monte_carlo_analysis(
            returns=returns,
            initial_capital=10000.0,
            n_simulations=n_sim,
            plot_results=False
        )
        
        stats = results['summary_statistics']
        print(f"  {n_sim:>4} simulations: Mean=${stats['mean_final_value']:>8,.0f}, VaR={stats['var_95']:>6.2%}")
    
    print("\nTesting different initial capitals:")
    for capital in initial_capitals:
        results = run_monte_carlo_analysis(
            returns=returns,
            initial_capital=capital,
            n_simulations=1000,
            plot_results=False
        )
        
        stats = results['summary_statistics']
        total_return = (stats['mean_final_value'] - capital) / capital
        print(f"  ${capital:>6,.0f}: Final=${stats['mean_final_value']:>8,.0f}, Return={total_return:>6.2%}")


def main():
    """Main demonstration function."""
    print("ðŸŽ¯ Monte Carlo Simulation Examples")
    print("=" * 60)
    print()
    
    try:
        # Run demonstrations
        demonstrate_basic_simulation()
        demonstrate_bootstrap_methods()
        demonstrate_confidence_intervals()
        demonstrate_risk_analysis()
        demonstrate_visualization()
        demonstrate_performance_comparison()
        demonstrate_parameter_sensitivity()
        
        print("\nâœ… All demonstrations completed successfully!")
        print("\nðŸ’¡ Next steps:")
        print("  1. Run the Streamlit dashboard: streamlit run pages/Monte_Carlo_Simulation.py")
        print("  2. Use MonteCarloSimulator in your backtesting scripts")
        print("  3. Experiment with different bootstrap methods")
        print("  4. Analyze risk metrics for your strategies")
        
    except Exception as e:
        print(f"âŒ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
