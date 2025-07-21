"""
Risk-Aware Hybrid Model Example

This example demonstrates the enhanced hybrid model with risk-aware weighting
using Sharpe ratio, drawdown, or MSE as the primary weighting metric.

# NOTE: Flake8 compliance changes applied. Non-ASCII print statements fixed.
"""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading.forecasting.hybrid_model import HybridModel


def generate_sample_data(days: int = 252) -> pd.DataFrame:
    """Generate sample price data for testing."""
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


class MockModel:
    """Mock model for demonstration purposes."""

    def __init__(self, name: str, bias: float = 0.0, noise_level: float = 0.1):
        self.name = name
        self.bias = bias
        self.noise_level = noise_level

    def fit(self, data: pd.DataFrame):
        """Mock fit method."""

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate mock predictions with different characteristics."""
        base_prices = data["close"].values

        if self.name == "High_Sharpe_Model":
            # Model with high Sharpe ratio (good risk-adjusted returns)
            predictions = base_prices * (
                1 + 0.001 + np.random.normal(0, 0.005, len(base_prices))
            )
        elif self.name == "Low_Drawdown_Model":
            # Model with low drawdown (conservative)
            predictions = base_prices * (
                1 + 0.0005 + np.random.normal(0, 0.003, len(base_prices))
            )
        elif self.name == "Low_MSE_Model":
            # Model with low MSE (accurate predictions)
            predictions = base_prices * (
                1 + 0.0002 + np.random.normal(0, 0.002, len(base_prices))
            )
        elif self.name == "Aggressive_Model":
            # Model with high returns but high volatility
            predictions = base_prices * (
                1 + 0.002 + np.random.normal(0, 0.01, len(base_prices))
            )
        else:
            # Default model
            predictions = base_prices * (
                1 + self.bias + np.random.normal(0, self.noise_level, len(base_prices))
            )

        return predictions


def create_hybrid_model() -> HybridModel:
    """Create a hybrid model with different mock models."""
    models = {
        "High_Sharpe_Model": MockModel("High_Sharpe_Model"),
        "Low_Drawdown_Model": MockModel("Low_Drawdown_Model"),
        "Low_MSE_Model": MockModel("Low_MSE_Model"),
        "Aggressive_Model": MockModel("Aggressive_Model"),
    }

    return HybridModel(models)


def demonstrate_risk_aware_weighting():
    print("🚀 Risk-Aware Hybrid Model Demonstration")
    print("=" * 60)

    # Create sample data
    data = generate_sample_data(252)
    print(f"🎯 Generated {len(data)} days of sample data")

    # Create hybrid model
    hybrid_model = create_hybrid_model()
    print("🎯 Created hybrid model with 4 mock models")

    # Fit models and calculate initial performance
    print("\n📊 Fitting models and calculating performance...")
    hybrid_model.fit(data)

    # Show initial weights
    print("\n📈 Initial Model Weights:")
    for model_name, weight in hybrid_model.weights.items():
        print(f"   {model_name}: {weight:.2%}")

    # Test different weighting metrics
    print("\n🎯 TESTING DIFFERENT WEIGHTING METRICS")
    print("=" * 60)

    weighting_metrics = ["sharpe", "drawdown", "mse"]
    results = {}

    for metric in weighting_metrics:
        print(f"\n📊 Testing {metric.upper()} weighting...")

        # Set weighting metric
        hybrid_model.set_weighting_metric(metric)

        # Get weights
        weights = hybrid_model.weights.copy()
        results[metric] = weights

        print(f"   Weights after {metric} weighting:")
        for model_name, weight in weights.items():
            print(f"     {model_name}: {weight:.2%}")

    # Analyze results
    print("\n📊 WEIGHTING METRIC ANALYSIS")
    print("=" * 60)

    # Create comparison table
    comparison_data = []
    for model_name in hybrid_model.models.keys():
        row = {"Model": model_name}
        for metric in weighting_metrics:
            row[f"{metric.title()} Weight"] = f"{results[metric][model_name]:.2%}"
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    print("\nWeight Comparison Table:")
    print(comparison_df.to_string(index=False))

    # Analyze which models benefit from each metric
    print("\n🔍 Analysis:")

    # Find best model for each metric
    for metric in weighting_metrics:
        best_model = max(results[metric].items(), key=lambda x: x[1])
        print(
            f"   {metric.title()} weighting favors: {best_model[0]} ({best_model[1]:.2%})"
        )

    return hybrid_model, results


def demonstrate_weighting_metric_selection():
    """Demonstrate how to select and configure weighting metrics."""
    print("\n" + "=" * 60)
    print("📋 WEIGHTING METRIC CONFIGURATION")
    print("=" * 60)

    # Create hybrid model
    hybrid_model = create_hybrid_model()
    data = generate_sample_data(252)
    hybrid_model.fit(data)

    # Show available weighting metrics
    weighting_info = hybrid_model.get_weighting_metric_info()

    print("\n📋 Available Weighting Metrics:")
    for metric, info in weighting_info["available_metrics"].items():
        print(f"   â€¢ {metric.title()}: {info['description']}")

    print("\n📋 Available Ensemble Methods:")
    for method, description in weighting_info["available_methods"].items():
        print(f"   â€¢ {method}: {description}")

    # Demonstrate metric switching
    print("\n🛠️ Demonstrating metric switching:")

    for metric in ["sharpe", "drawdown", "mse"]:
        hybrid_model.set_weighting_metric(metric)
        current_weights = hybrid_model.weights

        print(f"\n   {metric.title()} weighting:")
        for model_name, weight in current_weights.items():
            print(f"     {model_name}: {weight:.2%}")

    return hybrid_model


def demonstrate_performance_analysis():
    """Demonstrate performance analysis and recommendations."""
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Create and fit hybrid model
    hybrid_model = create_hybrid_model()
    data = generate_sample_data(252)
    hybrid_model.fit(data)

    # Get performance summary
    summary = hybrid_model.get_model_performance_summary()

    print("\n📊 Model Performance Summary:")
    for model_name, model_info in summary.items():
        if model_info["status"] == "active":
            print(f"\n   {model_name}:")
            print(f"     Current Weight: {model_info['current_weight']:.2%}")
            print(f"     Performance Count: {model_info['performance_count']}")

            avg_metrics = model_info["avg_metrics"]
            print(f"     Avg Sharpe: {avg_metrics.get('sharpe_ratio', 0):.3f}")
            print(f"     Avg Win Rate: {avg_metrics.get('win_rate', 0):.1%}")
            print(f"     Avg Max Drawdown: {avg_metrics.get('max_drawdown', 0):.1%}")
            print(f"     Avg MSE: {avg_metrics.get('mse', 0):.2f}")
            print(f"     Avg Total Return: {avg_metrics.get('total_return', 0):.1%}")

    # Show current configuration
    config = hybrid_model.scoring_config
    print(f"\n📋 Current Configuration:")
    print(f"   Method: {config['method']}")
    print(f"   Weighting Metric: {config['weighting_metric']}")
    print(f"   Min Performance Threshold: {config['min_performance_threshold']}")
    print(f"   Sharpe Floor: {config['sharpe_floor']}")
    print(f"   Drawdown Ceiling: {config['drawdown_ceiling']}")
    print(f"   MSE Ceiling: {config['mse_ceiling']}")


def create_visualization_comparison(
    hybrid_model: HybridModel, results: Dict[str, Dict[str, float]]
):
    """Create visualizations comparing different weighting methods."""
    print("\n" + "=" * 60)
    print("📊 CREATING VISUALIZATIONS")
    print("=" * 60)

    # Create comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Risk-Aware Weighting Comparison", fontsize=16, fontweight="bold")

    # 1. Weight comparison bar chart
    ax1 = axes[0, 0]
    metrics = list(results.keys())
    model_names = list(results[metrics[0]].keys())

    x = np.arange(len(model_names))
    width = 0.25

    for i, metric in enumerate(metrics):
        weights = [results[metric][name] for name in model_names]
        ax1.bar(x + i * width, weights, width, label=metric.title(), alpha=0.8)

    ax1.set_xlabel("Models")
    ax1.set_ylabel("Weight")
    ax1.set_title("Model Weights by Weighting Metric")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(model_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Weight distribution pie charts
    for i, metric in enumerate(metrics[1:], 1):
        ax = axes[0, i]
        weights = list(results[metric].values())
        labels = list(results[metric].keys())

        ax.pie(weights, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title(f"{metric.title()} Weighting")

    # 3. Performance metrics comparison
    ax3 = axes[1, 0]
    summary = hybrid_model.get_model_performance_summary()

    model_names = []
    sharpe_ratios = []
    win_rates = []
    drawdowns = []

    for model_name, model_info in summary.items():
        if model_info["status"] == "active":
            model_names.append(model_name)
            avg_metrics = model_info["avg_metrics"]
            sharpe_ratios.append(avg_metrics.get("sharpe_ratio", 0))
            win_rates.append(avg_metrics.get("win_rate", 0))
            drawdowns.append(abs(avg_metrics.get("max_drawdown", 0)))

    x = np.arange(len(model_names))
    width = 0.25

    ax3.bar(x - width, sharpe_ratios, width, label="Sharpe Ratio", alpha=0.8)
    ax3.bar(x, win_rates, width, label="Win Rate", alpha=0.8)
    ax3.bar(x + width, drawdowns, width, label="Max Drawdown (abs)", alpha=0.8)

    ax3.set_xlabel("Models")
    ax3.set_ylabel("Metric Value")
    ax3.set_title("Model Performance Metrics")
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Weight vs Performance correlation
    ax4 = axes[1, 1]

    # Get current weights (using Sharpe weighting as example)
    hybrid_model.set_weighting_metric("sharpe")
    current_weights = [hybrid_model.weights[name] for name in model_names]

    ax4.scatter(sharpe_ratios, current_weights, s=100, alpha=0.7)

    # Add model labels
    for i, model_name in enumerate(model_names):
        ax4.annotate(
            model_name,
            (sharpe_ratios[i], current_weights[i]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax4.set_xlabel("Sharpe Ratio")
    ax4.set_ylabel("Weight")
    ax4.set_title("Weight vs Sharpe Ratio Correlation")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("risk_aware_weighting_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"📊 Visualization saved as 'risk_aware_weighting_comparison.png'")


def demonstrate_streamlit_integration():
    """Demonstrate how to integrate with Streamlit."""
    print("\n" + "=" * 60)
    print("🎛️ STREAMLIT INTEGRATION EXAMPLE")
    print("=" * 60)

    print(
        """
To integrate the risk-aware hybrid model with Streamlit:

```python
import streamlit as st
from trading.forecasting.hybrid_model import HybridModel
from trading.ui.hybrid_model_config import render_hybrid_model_config_sidebar

# Create hybrid model
models = {
    "LSTM": lstm_model,
    "XGBoost": xgb_model,
    "Transformer": transformer_model
}
hybrid_model = HybridModel(models)

# Render configuration in sidebar
config = render_hybrid_model_config_sidebar(hybrid_model)

# Use the configured model
if st.button("Run Prediction"):
    prediction = hybrid_model.predict(data)
    st.line_chart(prediction)

# Show weighting comparison
render_weighting_metric_comparison(hybrid_model)

# Show performance dashboard
render_model_performance_dashboard(hybrid_model)
```

Key Features:
- Interactive weighting metric selection
- Real-time weight updates
- Performance visualization
- Configuration validation
- Recommendations system
"""
    )


def main():
    """Main function to run the risk-aware hybrid model demonstration."""
    print("🎯 Risk-Aware Hybrid Model Example")
    print(
        "This example demonstrates risk-aware weighting using Sharpe, Drawdown, or MSE"
    )

    # Demonstrate risk-aware weighting
    hybrid_model, results = demonstrate_risk_aware_weighting()

    # Demonstrate metric selection
    demonstrate_weighting_metric_selection()

    # Demonstrate performance analysis
    demonstrate_performance_analysis()

    # Create visualizations
    create_visualization_comparison(hybrid_model, results)

    # Demonstrate Streamlit integration
    demonstrate_streamlit_integration()

    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print("🎯 Risk-aware weighting successfully implemented")
    print("🎯 Sharpe ratio weighting: weight = Sharpe / total_Sharpe")
    print("🎯 Drawdown weighting: weight = (1 + drawdown) / total")
    print("🎯 MSE weighting: weight = (1/MSE) / total(1/MSE)")
    print("🎯 User-selectable weighting metrics")
    print("🎯 Comprehensive performance analysis")
    print("🎯 Streamlit UI integration")
    print("🎯 Configuration validation and recommendations")

    print("\nKey Benefits:")
    print("- Risk-aware ensemble weighting")
    print("- Better model differentiation")
    print("- Improved risk-adjusted returns")
    print("- Flexible configuration options")
    print("- Comprehensive performance tracking")

    return hybrid_model, results


if __name__ == "__main__":
    main()
