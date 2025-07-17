"""
Model Performance Logging Example

This script demonstrates how to use the model performance logging functionality
to track and analyze model performance across different tickers and time periods.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memory.model_log import (
    log_model_performance,
    get_model_performance_history,
    get_best_models,
    get_available_tickers,
    get_available_models,
    clear_model_performance_log
)


def generate_sample_performance_data():
    """Generate realistic sample performance data for demonstration."""
    
    # Define model types and their typical performance characteristics
    model_types = {
        "LSTM": {
            "sharpe_range": (1.2, 2.5),
            "mse_range": (0.015, 0.045),
            "drawdown_range": (-0.20, -0.05),
            "return_range": (0.15, 0.40),
            "win_rate_range": (0.55, 0.80),
            "accuracy_range": (0.60, 0.85)
        },
        "XGBoost": {
            "sharpe_range": (1.5, 2.8),
            "mse_range": (0.012, 0.035),
            "drawdown_range": (-0.18, -0.04),
            "return_range": (0.20, 0.45),
            "win_rate_range": (0.60, 0.85),
            "accuracy_range": (0.65, 0.90)
        },
        "Transformer": {
            "sharpe_range": (1.3, 2.6),
            "mse_range": (0.014, 0.040),
            "drawdown_range": (-0.22, -0.06),
            "return_range": (0.18, 0.42),
            "win_rate_range": (0.58, 0.82),
            "accuracy_range": (0.62, 0.88)
        },
        "RandomForest": {
            "sharpe_range": (1.1, 2.2),
            "mse_range": (0.018, 0.050),
            "drawdown_range": (-0.25, -0.08),
            "return_range": (0.12, 0.35),
            "win_rate_range": (0.52, 0.75),
            "accuracy_range": (0.58, 0.82)
        },
        "SVM": {
            "sharpe_range": (1.0, 2.0),
            "mse_range": (0.020, 0.055),
            "drawdown_range": (-0.28, -0.10),
            "return_range": (0.10, 0.30),
            "win_rate_range": (0.50, 0.72),
            "accuracy_range": (0.55, 0.78)
        }
    }
    
    tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"]
    model_versions = ["v1", "v2", "v3", "v4"]
    
    performance_data = []
    
    # Generate data for each ticker
    for ticker in tickers:
        # Generate 3-5 models per ticker
        num_models = random.randint(3, 5)
        
        for i in range(num_models):
            # Select random model type
            model_type = random.choice(list(model_types.keys()))
            model_version = random.choice(model_versions)
            model_name = f"{model_type}_{model_version}"
            
            # Get performance ranges for this model type
            ranges = model_types[model_type]
            
            # Generate performance metrics
            sharpe = random.uniform(*ranges["sharpe_range"])
            mse = random.uniform(*ranges["mse_range"])
            drawdown = random.uniform(*ranges["drawdown_range"])
            total_return = random.uniform(*ranges["return_range"])
            win_rate = random.uniform(*ranges["win_rate_range"])
            accuracy = random.uniform(*ranges["accuracy_range"])
            
            # Add some variation based on ticker
            if ticker in ["TSLA", "NVDA"]:  # More volatile stocks
                sharpe *= 0.9
                drawdown *= 1.2
            elif ticker in ["AAPL", "MSFT"]:  # More stable stocks
                sharpe *= 1.1
                drawdown *= 0.8
            
            performance_data.append({
                "model_name": model_name,
                "ticker": ticker,
                "sharpe": round(sharpe, 3),
                "mse": round(mse, 4),
                "drawdown": round(drawdown, 3),
                "total_return": round(total_return, 3),
                "win_rate": round(win_rate, 3),
                "accuracy": round(accuracy, 3),
                "notes": f"{model_type} model for {ticker} - {model_version}"
            })
    
    return performance_data


def demonstrate_logging():
    """Demonstrate the logging functionality."""
    print("🚀 Model Performance Logging Demonstration")
    print("=" * 50)
    
    # Clear existing data
    print("🗑️ Clearing existing data...")
    clear_model_performance_log()
    
    # Generate and log sample data
    print("📊 Generating sample performance data...")
    sample_data = generate_sample_performance_data()
    
    for i, data in enumerate(sample_data, 1):
        print(f"Logging performance {i}/{len(sample_data)}: {data['model_name']} for {data['ticker']}")
        log_model_performance(**data)
    
    print(f"✅ Successfully logged {len(sample_data)} performance records!")
    print()


def demonstrate_analysis():
    """Demonstrate the analysis functionality."""
    print("📈 Performance Analysis Demonstration")
    print("=" * 50)
    
    # Get available data
    tickers = get_available_tickers()
    print(f"📊 Available tickers: {', '.join(tickers)}")
    
    # Analyze each ticker
    for ticker in tickers[:3]:  # Show first 3 tickers
        print(f"\n🎯 Analysis for {ticker}:")
        
        # Get best models for this ticker
        best_models = get_best_models(ticker)
        
        if best_models:
            print("🏆 Best models:")
            for metric, data in best_models.items():
                if data.get("model"):
                    metric_name = metric.replace("best_", "").replace("_", " ").title()
                    value = data["value"]
                    
                    if metric == "best_mse":
                        formatted_value = f"{value:.4f}"
                    elif metric in ["best_total_return", "best_win_rate", "best_accuracy", "best_drawdown"]:
                        formatted_value = f"{value:.1%}"
                    else:
                        formatted_value = f"{value:.3f}"
                    
                    print(f"  • {metric_name}: {data['model']} ({formatted_value})")
        
        # Get performance history
        history = get_model_performance_history(ticker=ticker)
        if not history.empty:
            print(f"📈 Performance history: {len(history)} records")
            
            # Show average metrics
            avg_sharpe = history['sharpe'].mean()
            avg_mse = history['mse'].mean()
            print(f"  • Average Sharpe: {avg_sharpe:.3f}")
            print(f"  • Average MSE: {avg_mse:.4f}")
    
    print()


def demonstrate_filtering():
    """Demonstrate filtering and querying functionality."""
    print("🔍 Filtering and Querying Demonstration")
    print("=" * 50)
    
    # Get all performance history
    all_history = get_model_performance_history()
    print(f"📊 Total performance records: {len(all_history)}")
    
    # Filter by specific model
    lstm_history = get_model_performance_history(model_name="LSTM_v1")
    print(f"📈 LSTM_v1 records: {len(lstm_history)}")
    
    if not lstm_history.empty:
        print("LSTM_v1 performance summary:")
        print(f"  • Average Sharpe: {lstm_history['sharpe'].mean():.3f}")
        print(f"  • Average MSE: {lstm_history['mse'].mean():.4f}")
        print(f"  • Best Sharpe: {lstm_history['sharpe'].max():.3f}")
        print(f"  • Best MSE: {lstm_history['mse'].min():.4f}")
    
    # Filter by recent data
    recent_history = get_model_performance_history(days_back=7)
    print(f"🕒 Recent records (7 days): {len(recent_history)}")
    
    print()


def demonstrate_model_comparison():
    """Demonstrate model comparison functionality."""
    print("⚖️ Model Comparison Demonstration")
    print("=" * 50)
    
    # Get all available models
    all_models = set()
    tickers = get_available_tickers()
    
    for ticker in tickers:
        models = get_available_models(ticker)
        all_models.update(models)
    
    print(f"🤖 Available models: {', '.join(sorted(all_models))}")
    
    # Compare models across all tickers
    all_history = get_model_performance_history()
    
    if not all_history.empty:
        # Group by model and calculate average performance
        model_performance = all_history.groupby('model_name').agg({
            'sharpe': ['mean', 'std', 'count'],
            'mse': ['mean', 'std'],
            'total_return': ['mean', 'std'],
            'win_rate': ['mean', 'std']
        }).round(4)
        
        print("\n📊 Model Performance Summary:")
        print(model_performance)
        
        # Find best overall model by Sharpe ratio
        best_sharpe_model = all_history.loc[all_history['sharpe'].idxmax()]
        print(f"\n🏆 Best Sharpe Ratio: {best_sharpe_model['model_name']} ({best_sharpe_model['ticker']}) - {best_sharpe_model['sharpe']:.3f}")
        
        # Find best overall model by MSE
        best_mse_model = all_history.loc[all_history['mse'].idxmin()]
        print(f"🎯 Best MSE: {best_mse_model['model_name']} ({best_mse_model['ticker']}) - {best_mse_model['mse']:.4f}")
    
    print()


def main():
    """Main demonstration function."""
    print("🎯 Model Performance Logging System Demo")
    print("=" * 60)
    print()
    
    try:
        # Run demonstrations
        demonstrate_logging()
        demonstrate_analysis()
        demonstrate_filtering()
        demonstrate_model_comparison()
        
        print("✅ All demonstrations completed successfully!")
        print("\n💡 Next steps:")
        print("  1. Run the Streamlit dashboard: streamlit run pages/Model_Performance_Dashboard.py")
        print("  2. Use log_model_performance() in your model training scripts")
        print("  3. Analyze performance trends with get_model_performance_history()")
        print("  4. Find best models with get_best_models()")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 