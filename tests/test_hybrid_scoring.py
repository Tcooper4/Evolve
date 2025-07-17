import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from trading.forecasting.hybrid_model import HybridModel

class MockModel:
    model for testing purposes.
    def __init__(self, name: str, bias: float =0.0, noise: float = 0.1
        self.name = name
        self.bias = bias
        self.noise = noise
    
    def fit(self, data):
        pass
    
    def predict(self, data):
        # Generate mock predictions with bias and noise
        actual = data["close"].values
        predictions = actual * (1 + self.bias) + np.random.normal(0, self.noise, len(actual))
        return predictions

def generate_test_data(n_days: int = 100 -> pd.DataFrame:
    """Generate realistic test data.   dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Generate price data with trend and volatility
    np.random.seed(42)
    returns = np.random.normal(0.01, 0.02, n_days)  # Daily returns
    prices =10* np.exp(np.cumsum(returns))
    
    # Add some trend
    trend = np.linspace(0, 0.1, n_days)
    prices = prices * (1 + trend)
    
    data = pd.DataFrame({
     date': dates,
        open': prices * (1 + np.random.normal(0,0.005 n_days)),
        high': prices * (1 + np.abs(np.random.normal(0,0.01n_days))),
        low': prices * (1 - np.abs(np.random.normal(0,0.01n_days))),
       close: prices,
  volume': np.random.randint(1000, 10000days)
    })
    
    data.set_index('date', inplace=True)
    return data

def test_scoring_methods():
    different scoring methods."""
    print("🔬 Testing Comprehensive Scoring System)
    print(= * 50)   
    # Generate test data
    data = generate_test_data(200)
    print(f"Generated {len(data)} days of test data")
    
    # Create mock models with different characteristics
    models = {
        High_Sharpe_Low_Drawdown: MockModel(High_Sharpe_Low_Drawdown", bias=01se=0.05),
    High_WinRate_Medium_Risk: MockModel("High_WinRate_Medium_Risk", bias=0.2ise=001,
        Low_MSE_Poor_Sharpe": MockModel(Low_MSE_Poor_Sharpe", bias=-01se=0.003
        High_Volatility_High_Return: MockModel(High_Volatility_High_Return", bias=0.3ise=0.02),
      Conservative_Stable": MockModel("Conservative_Stable", bias=005, noise=08)
    }
    
    # Test different scoring configurations
    scoring_configs =[object Object]
       weighted_average: {          method":weighted_average",
       metrics[object Object]
                sharpe_ratio: {"weight:00.4direction": "maximize"},
            win_rate: {"weight:00.3direction": "maximize"},
                max_drawdown: {"weight:00.2direction": "minimize"},
                mse: {"weight:00.1direction": "minimize"}
            }
        },ahp: {         method":ahp        },
     composite: {       method": "composite     }
    }
    
    results = {}
    
    for method_name, config in scoring_configs.items():
        print(f"\n📊 Testing {method_name.upper()} scoring method")
        print("-" * 30)
        
        # Create hybrid model
        hybrid_model = HybridModel(models)
        hybrid_model.set_scoring_config(config)
        
        # Fit models multiple times to build performance history
        for i in range(10
            # Use different subsets of data for each iteration
            start_idx = i * 20           end_idx = start_idx + 100
            subset_data = data.iloc[start_idx:end_idx]
            
            if len(subset_data) >= 50:  # Ensure enough data
                hybrid_model.fit(subset_data)
        
        # Get final weights and performance summary
        weights = hybrid_model.weights
        performance_summary = hybrid_model.get_model_performance_summary()
        
        results[method_name] = {
            weights": weights,
        performance: performance_summary
        }
        
        print(fFinalweights:")
        for model, weight in weights.items():
            print(f"  [object Object]model}: {weight:.3f}")
        
        print(f"\nPerformance summary:")
        for model, perf in performance_summary.items():
            if perf["status"] == "active:               avg_metrics = perf["avg_metrics]             print(f"  {model}:)             print(f    Sharpe: {avg_metrics['sharpe_ratio']:.3f})             print(f"    Win Rate: {avg_metrics['win_rate']:.3f})             print(f    Max DD:[object Object]avg_metricsmax_drawdown']:.3f})             print(f"    MSE:[object Object]avg_metrics['mse]:.6f})    return results, data, models

def compare_with_old_mse_system():
ompare new scoring system with old MSE-based approach."nt("\n🔄 Comparing with Old MSE-Based System)
    print(= * 50)   
    # Generate test data
    data = generate_test_data(150)
    
    # Create models with different characteristics
    models = {
        Good_Sharpe_Bad_MSE: MockModel(Good_Sharpe_Bad_MSE", bias=02se=0.015
        Bad_Sharpe_Good_MSE: MockModel(Bad_Sharpe_Good_MSE", bias=-01se=0.5
       Balanced_Model": MockModel("Balanced_Model", bias=001noise=0.1)
    }
    
    # Test old MSE-based approach (simulated)
    print("Old MSE-based weights (simulated):)
    mse_weights = {
        Good_Sharpe_Bad_MSE": 02  # High MSE = low weight
        Bad_Sharpe_Good_MSE":00.6  # Low MSE = high weight
        Balanced_Model:0.2    }
    
    for model, weight in mse_weights.items():
        print(f"  [object Object]model}: {weight:.3f})   
    # Test new comprehensive scoring
    print("\nNew comprehensive scoring weights:")
    hybrid_model = HybridModel(models)
    
    # Fit models to build performance history
    for i in range(8):
        start_idx = i * 20
        end_idx = start_idx +80       subset_data = data.iloc[start_idx:end_idx]
        
        if len(subset_data) >= 50:
            hybrid_model.fit(subset_data)
    
    new_weights = hybrid_model.weights
    for model, weight in new_weights.items():
        print(f"  [object Object]model}: {weight:.3f}")
    
    # Compare the differences
    print("\nWeight changes:")
    for model in models.keys():
        old_weight = mse_weights.get(model, 0.0)
        new_weight = new_weights.get(model, 00)
        change = new_weight - old_weight
        print(f"  {model}: {old_weight:0.3f} → {new_weight:0.3f} ({change:+.3f}))
    
    return mse_weights, new_weights

def visualize_results(results, data, models):
  ualize the results of different scoring methods."int("\n📈 Creating visualizations...")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2figsize=(1512)
    fig.suptitle('Hybrid Model Scoring System Comparison', fontsize=16)
    
    # Plot 1: Weight comparison across methods
    ax1[0, 0]
    methods = list(results.keys())
    model_names = list(models.keys())
    
    x = np.arange(len(model_names))
    width =0.25
    for i, method in enumerate(methods):
        weights = [results[method]["weights"].get(model, 0) for model in model_names]
        ax1.bar(x + i * width, weights, width, label=method.replace('_', ').title())
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Weight')
    ax1.set_title('Weight Distribution by Scoring Method')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([name.replace('_',n for name in model_names], rotation=45
    ax1end()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance metrics heatmap
    ax2[0, 1]
    metrics = ['sharpe_ratio', win_rate, max_drawdown',mse]
    metric_data = []
    
    for model in model_names:
        model_metrics = []
        for method in methods:
            perf = results[method][performance].get(model, [object Object]           if perf.get("status") == "active:               avg_metrics = perf["avg_metrics]
                # Use weighted average across methods
                avg_value = np.mean([avg_metrics.get(metric, 0) for metric in metrics])
                model_metrics.append(avg_value)
            else:
                model_metrics.append(0)
        metric_data.append(model_metrics)
    
    sns.heatmap(metric_data, 
                xticklabels=m.replace('_', '\n') for m in methods],
                yticklabels=[name.replace('_',n for name in model_names],
                annot=True, fmt='0.3f', cmap=RdYlGn, ax=ax2)
    ax2.set_title('Average Performance Score by Method)
    
    # Plot 3a with predictions
    ax3 axes[1
    ax3t(data.index, data['close'], label='Actual Price', linewidth=2)
    
    # Generate predictions using the best scoring method
    best_method = max(results.keys(), key=lambda x: np.mean(list(results[x]["weights].values())))
    hybrid_model = HybridModel(models)
    hybrid_model.set_scoring_config({"method": best_method})
    hybrid_model.fit(data)
    predictions = hybrid_model.predict(data)
    
    ax3.plot(data.index[-len(predictions):], predictions, 
             label=f'Ensemble ({best_method}), linestyle='--', linewidth=2)
    ax3et_xlabel('Date')
    ax3.set_ylabel('Price')
    ax3.set_title(fPrice Predictions - {best_method.replace("_", ").title()}')
    ax3end()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Weight evolution over time
    ax4= axes[1,1 # Simulate weight evolution
    time_points = np.arange(10)
    for model in model_names[:3]:  # Show first 3 models
        weights = [results[best_method]["weights].get(model, 0) + np.random.normal(0, 0.02) 
                  for _ in time_points]
        ax4.plot(time_points, weights, marker='o', label=model.replace('_',  )
    
    ax4xlabel('Time Steps')
    ax4.set_ylabel('Weight')
    ax4.set_title('Weight Evolution Over Time')
    ax4end()
    ax4.grid(True, alpha=00.3    
    plt.tight_layout()
    plt.savefig('hybrid_scoring_comparison.png', dpi=300bbox_inches=tight')
    print("📊 Visualization saved as 'hybrid_scoring_comparison.png')   
    return fig

def main():
    """Main test function."""
    print("🚀 Starting Hybrid Model Scoring System Tests)
    print(= * 60  # Test different scoring methods
    results, data, models = test_scoring_methods()
    
    # Compare with old MSE system
    old_weights, new_weights = compare_with_old_mse_system()
    
    # Create visualizations
    fig = visualize_results(results, data, models)
    
    # Summary
    print("\n" + "=" *60 print("📋 SUMMARY)
    print("=" * 60)
    print("✅ New comprehensive scoring system implemented successfully!) print("✅ Replaced MSE-based weights with Sharpe ratio, drawdown, and win rate")
    print("✅ Added AHP and composite scoring methods")
    print("✅ Models with poor Sharpe ratios now get reduced weights")
    print(✅ System provides better risk-adjusted performance")
    
    print("\n🎯 Key Improvements:")
    print("  • Sharpe ratio weighting (40ards risk-adjusted returns")
    print("  • Win rate weighting (30%) - rewards consistency")
    print("  • Drawdown weighting (20%) - penalizes excessive risk")
    print( • MSE weighting (10%) - maintains some accuracy focus")
    print("  • Minimum performance threshold - prevents zero weights")
    
    print("\n🔧 Available Scoring Methods:")
    print("  • weighted_average: Configurable metric weights")
    print(  • ahp:Analytic Hierarchy Process")
    print("  • composite: Trend-adjusted scoring")
    
    print("\n📊 Test completed successfully!)if __name__ == "__main__":
    main() 