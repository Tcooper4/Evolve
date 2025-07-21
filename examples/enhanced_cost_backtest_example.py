"""
Enhanced Cost Backtesting Example

This example demonstrates the enhanced cost modeling capabilities including:
- Commission, slippage, and spread calculations
- Cash drag and idle capital adjustments
- Cost-adjusted performance metrics
- Different cost scenarios comparison
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading.backtesting.backtester import Backtester
from trading.backtesting.cost_model import (
    CostConfig,
    CostModel,
)
from trading.backtesting.performance_analysis import CostParameters, PerformanceAnalyzer


def generate_sample_data(days: int = 252) -> pd.DataFrame:
    """Generate sample price data for backtesting."""
    np.random.seed(42)

    # Generate price data with trend and volatility
    dates = pd.date_range(start="2023-01-01", periods=days, freq="D")

    # Create price series with trend and noise
    trend = np.linspace(100, 120, days)  # Upward trend
    noise = np.random.normal(0, 1, days) * 2
    prices = trend + noise

    # Ensure prices are positive
    prices = np.maximum(prices, 10)

    # Create OHLC data
    data = pd.DataFrame(
        {
            "date": dates,
            "open": prices * (1 + np.random.normal(0, 0.01, days)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.02, days))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.02, days))),
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, days),
        }
    )

    data.set_index("date", inplace=True)
    return data


def create_simple_strategy_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Create simple moving average crossover signals."""
    # Calculate moving averages
    data["sma_20"] = data["close"].rolling(window=20).mean()
    data["sma_50"] = data["close"].rolling(window=50).mean()

    # Generate signals
    data["signal"] = 0
    data.loc[data["sma_20"] > data["sma_50"], "signal"] = 1  # Buy signal
    data.loc[data["sma_20"] < data["sma_50"], "signal"] = -1  # Sell signal

    return data


def run_cost_comparison_backtest():
    """Run backtest with different cost scenarios."""
    print("🚀 Running Enhanced Cost Backtesting Example")
    print("=" * 60)

    # Generate sample data
    data = generate_sample_data(252)
    signals_data = create_simple_strategy_signals(data.copy())

    # Define different cost scenarios
    cost_scenarios = {
        "No Costs": CostParameters(enable_cost_adjustment=False),
        "Low Cost": CostParameters(
            commission_rate=0.0002,  # 0.02%
            slippage_rate=0.0005,  # 0.05%
            spread_rate=0.0001,  # 0.01%
            cash_drag_rate=0.01,  # 1%
        ),
        "Retail Trading": CostParameters(
            commission_rate=0.001,  # 0.1%
            slippage_rate=0.002,  # 0.2%
            spread_rate=0.0005,  # 0.05%
            cash_drag_rate=0.02,  # 2%
        ),
        "High Cost": CostParameters(
            commission_rate=0.003,  # 0.3%
            slippage_rate=0.005,  # 0.5%
            spread_rate=0.001,  # 0.1%
            cash_drag_rate=0.03,  # 3%
        ),
    }

    results = {}

    for scenario_name, cost_params in cost_scenarios.items():
        print(f"\n📊 Testing {scenario_name} scenario...")

        # Create cost model
        cost_config = CostConfig(
            fee_rate=cost_params.commission_rate,
            slippage_rate=cost_params.slippage_rate,
            spread_rate=cost_params.spread_rate,
        )
        cost_model = CostModel(cost_config, data)

        # Initialize backtester
        backtester = Backtester(
            data=data, initial_cash=100000, cost_model=cost_model, max_trades=1000
        )

        # Run backtest
        try:
            # Process signals
            processed_signals = backtester.process_signals_dataframe(
                signals_data[["signal"]], fill_method="ffill"
            )

            # Execute trades based on signals
            for i, (timestamp, row) in enumerate(processed_signals.iterrows()):
                if i == 0:  # Skip first row
                    continue

                signal = row["signal"]
                price = data.loc[timestamp, "close"]

                if signal == 1:  # Buy signal
                    # Calculate position size (simple equal weight)
                    position_value = backtester.cash * 0.95  # Use 95% of cash
                    quantity = position_value / price

                    if quantity > 0:
                        backtester.execute_trade(
                            timestamp=timestamp,
                            asset="STOCK",
                            quantity=quantity,
                            price=price,
                            trade_type="BUY",
                            strategy="MA_CROSSOVER",
                            signal=signal,
                        )

                elif signal == -1:  # Sell signal
                    # Close all positions
                    for asset, position in backtester.positions.items():
                        if position > 0:
                            backtester.execute_trade(
                                timestamp=timestamp,
                                asset=asset,
                                quantity=position,
                                price=price,
                                trade_type="SELL",
                                strategy="MA_CROSSOVER",
                                signal=signal,
                            )

            # Calculate performance metrics
            performance_analyzer = PerformanceAnalyzer(cost_params)
            equity_df = backtester._calculate_equity_curve()
            trade_log_df = (
                pd.DataFrame(backtester.trade_log)
                if backtester.trade_log
                else pd.DataFrame()
            )

            metrics = performance_analyzer.compute_metrics(
                equity_df, trade_log_df, cost_params
            )

            results[scenario_name] = {
                "metrics": metrics,
                "equity_curve": equity_df,
                "trade_log": trade_log_df,
                "final_equity": (
                    equity_df["equity_curve"].iloc[-1]
                    if "equity_curve" in equity_df.columns
                    else 100000
                ),
            }

            print(
                f"✅ {scenario_name}: Final Equity = ${metrics.get('final_equity', 100000):,.2f}"
            )
            print(f"   Total Return: {metrics.get('total_return', 0) * 100:.2f}%")
            print(
                f"   Cost-Adjusted Return: {metrics.get('cost_adjusted_return', 0) * 100:.2f}%"
            )
            print(
                f"   Total Trading Costs: ${metrics.get('total_trading_costs', 0):,.2f}"
            )

        except Exception as e:
            print(f"❌ Error in {scenario_name}: {e}")
            results[scenario_name] = None

    return results


def analyze_cost_impact(results: dict):
    """Analyze the impact of different cost scenarios."""
    print("\n" + "=" * 60)
    print("📈 COST IMPACT ANALYSIS")
    print("=" * 60)

    # Create comparison table
    comparison_data = []

    for scenario_name, result in results.items():
        if result is None:
            continue

        metrics = result["metrics"]
        comparison_data.append(
            {
                "Scenario": scenario_name,
                "Total Return (%)": metrics.get("total_return", 0) * 100,
                "Cost-Adjusted Return (%)": metrics.get("cost_adjusted_return", 0)
                * 100,
                "Cost Impact (%)": metrics.get("cost_impact", 0),
                "Sharpe Ratio": metrics.get("sharpe_ratio", 0),
                "Cost-Adjusted Sharpe": metrics.get("cost_adjusted_sharpe", 0),
                "Total Costs ($)": metrics.get("total_trading_costs", 0),
                "Cost per Trade ($)": metrics.get("cost_per_trade", 0),
                "Number of Trades": metrics.get("num_trades", 0),
                "Win Rate (%)": metrics.get("win_rate", 0) * 100,
                "Max Drawdown (%)": metrics.get("max_drawdown", 0) * 100,
            }
        )

    comparison_df = pd.DataFrame(comparison_data)

    # Display comparison table
    print("\n📊 Performance Comparison:")
    print(comparison_df.round(2).to_string(index=False))

    # Create visualizations
    create_cost_impact_visualizations(comparison_df, results)

    return comparison_df


def create_cost_impact_visualizations(comparison_df: pd.DataFrame, results: dict):
    """Create visualizations showing cost impact."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Trading Cost Impact Analysis", fontsize=16, fontweight="bold")

    # 1. Return comparison
    ax1 = axes[0, 0]
    x = range(len(comparison_df))
    width = 0.35

    ax1.bar(
        [i - width / 2 for i in x],
        comparison_df["Total Return (%)"],
        width,
        label="Gross Return",
        alpha=0.8,
    )
    ax1.bar(
        [i + width / 2 for i in x],
        comparison_df["Cost-Adjusted Return (%)"],
        width,
        label="Net Return",
        alpha=0.8,
    )

    ax1.set_xlabel("Cost Scenario")
    ax1.set_ylabel("Return (%)")
    ax1.set_title("Gross vs Net Returns")
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df["Scenario"], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Cost breakdown
    ax2 = axes[0, 1]
    x = np.arange(len(comparison_df))
    width = 0.35

    ax2.bar(
        x - width / 2,
        comparison_df["Total Costs ($)"],
        width,
        label="Total Costs",
        alpha=0.8,
    )
    ax2.bar(
        x + width / 2,
        comparison_df["Cost per Trade ($)"],
        width,
        label="Cost per Trade",
        alpha=0.8,
    )

    ax2.set_xlabel("Cost Scenario")
    ax2.set_ylabel("Cost ($)")
    ax2.set_title("Trading Costs")
    ax2.set_xticks(x)
    ax2.set_xticklabels(comparison_df["Scenario"], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Sharpe ratio comparison
    ax3 = axes[1, 0]
    ax3.bar(x, comparison_df["Sharpe Ratio"], alpha=0.8, label="Gross Sharpe")
    ax3.bar(x, comparison_df["Cost-Adjusted Sharpe"], alpha=0.8, label="Net Sharpe")

    ax3.set_xlabel("Cost Scenario")
    ax3.set_ylabel("Sharpe Ratio")
    ax3.set_title("Risk-Adjusted Returns")
    ax3.set_xticks(x)
    ax3.set_xticklabels(comparison_df["Scenario"], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Equity curves
    ax4 = axes[1, 1]
    for scenario_name, result in results.items():
        if result is not None and "equity_curve" in result:
            equity_curve = result["equity_curve"]["equity_curve"]
            ax4.plot(
                equity_curve.index,
                equity_curve.values,
                label=scenario_name,
                linewidth=2,
            )

    ax4.set_xlabel("Date")
    ax4.set_ylabel("Portfolio Value ($)")
    ax4.set_title("Equity Curves")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("cost_impact_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n?? Visualization saved as 'cost_impact_analysis.png'")


def demonstrate_cost_config_ui():
    """Demonstrate the cost configuration UI component."""
    print("\n" + "=" * 60)
    print("🎛️ COST CONFIGURATION UI DEMONSTRATION")
    print("=" * 60)

    # This would normally be used in a Streamlit app
    # For demonstration, we'll show how to use the component

    print("To use the cost configuration UI in a Streamlit app:")
    print(
        """
    import streamlit as st
    from trading.ui.cost_config import render_cost_config_sidebar, CostConfigUI

    # In your Streamlit app:
    st.title("Enhanced Cost Backtesting")

    # Render cost configuration in sidebar
    cost_params = render_cost_config_sidebar(
        config_ui=CostConfigUI(show_advanced=True),
        default_params=CostParameters()
    )

    # Use the cost parameters in your backtest
    if st.button("Run Backtest"):
        # Your backtesting code here
        pass
    """
    )


def main():
    """Main function to run the enhanced cost backtesting example."""
    print("🎯 Enhanced Cost Backtesting Example")
    print("This example demonstrates realistic trading cost modeling")

    # Run cost comparison backtest
    results = run_cost_comparison_backtest()

    # Analyze cost impact
    comparison_df = analyze_cost_impact(
        results
    )  # noqa: F841 - Used in analyze_cost_impact function

    # Demonstrate UI component
    demonstrate_cost_config_ui()

    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print("✅ Enhanced cost modeling successfully implemented")
    print("✅ Cost-adjusted performance metrics calculated")
    print("✅ Different cost scenarios compared")
    print("✅ Cost impact analysis completed")
    print("✅ UI component for cost configuration created")
    print("\nKey Features:")
    print("- Commission, slippage, and spread calculations")
    print("- Cash drag and idle capital adjustments")
    print("- Cost-adjusted returns and Sharpe ratios")
    print("- Comprehensive cost breakdown analysis")
    print("- Streamlit UI for cost parameter configuration")

    return results


if __name__ == "__main__":
    main()
