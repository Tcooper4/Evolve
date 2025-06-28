"""
Optimizer Dashboard.

This module provides a Streamlit dashboard for the optimization framework,
allowing users to:
1. Select optimization methods (Grid, Bayesian, Genetic)
2. Configure optimization parameters
3. Visualize optimization results
4. Compare different optimization runs
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
import os

# Import from consolidated trading.optimization module
from trading.optimization import OptimizerFactory, StrategyOptimizer, OptimizationVisualizer
from trading.agents.strategy_switcher import StrategySwitcher
from trading.utils.memory_logger import MemoryLogger

def load_strategy_data() -> pd.DataFrame:
    """Load strategy performance data."""
    # TODO: Implement data loading
    return pd.DataFrame()

def main():
    st.title("Strategy Optimizer")
    
    # Initialize components
    strategy_switcher = StrategySwitcher()
    memory_logger = MemoryLogger()
    optimizer = StrategyOptimizer(strategy_switcher, memory_logger)
    
    # Sidebar configuration
    st.sidebar.header("Optimization Settings")
    
    # Strategy selection
    strategy = st.sidebar.selectbox(
        "Select Strategy",
        ["RSI", "MACD", "Bollinger", "SMA"]
    )
    
    # Optimizer selection
    optimizer_type = st.sidebar.selectbox(
        "Select Optimizer",
        optimizer.get_available_optimizers()
    )
    
    # Parameter space configuration
    st.sidebar.subheader("Parameter Space")
    param_space = optimizer.get_strategy_param_space(strategy)
    
    # Display parameter ranges
    for param, range_info in param_space.items():
        st.sidebar.text(f"{param}: {range_info}")
    
    # Optimization settings
    st.sidebar.subheader("Optimization Settings")
    
    if optimizer_type == "Grid":
        n_jobs = st.sidebar.slider("Number of Jobs", 1, 8, 4)
        settings = {"n_jobs": n_jobs}
        
    elif optimizer_type == "Bayesian":
        n_initial_points = st.sidebar.slider("Initial Random Points", 5, 20, 10)
        n_iterations = st.sidebar.slider("Optimization Iterations", 10, 100, 50)
        settings = {
            "n_initial_points": n_initial_points,
            "n_iterations": n_iterations
        }
        
    elif optimizer_type == "Genetic":
        population_size = st.sidebar.slider("Population Size", 20, 200, 100)
        n_generations = st.sidebar.slider("Number of Generations", 10, 100, 50)
        mutation_prob = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.2)
        crossover_prob = st.sidebar.slider("Crossover Probability", 0.0, 1.0, 0.8)
        settings = {
            "population_size": population_size,
            "n_generations": n_generations,
            "mutation_prob": mutation_prob,
            "crossover_prob": crossover_prob
        }
    
    # Load training data
    data = load_strategy_data()
    
    # Run optimization
    if st.sidebar.button("Start Optimization"):
        with st.spinner("Running optimization..."):
            results = optimizer.optimize_strategy(
                strategy=strategy,
                optimizer_type=optimizer_type,
                param_space=param_space,
                training_data=data,
                **settings
            )
            
            # Display results
            OptimizationVisualizer.display_optimization_summary(results)
            
            # Save results
            if st.sidebar.button("Save Results"):
                save_path = f"optimization_results/{strategy}_{optimizer_type}.json"
                optimizer.save_optimization_results(results, save_path)
                st.success(f"Results saved to {save_path}")
    
    # Load previous results
    st.sidebar.subheader("Load Previous Results")
    result_files = [f for f in os.listdir("optimization_results") 
                   if f.endswith(".json")]
    
    if result_files:
        selected_file = st.sidebar.selectbox(
            "Select Results File",
            result_files
        )
        
        if st.sidebar.button("Load Results"):
            results = optimizer.load_optimization_results(
                f"optimization_results/{selected_file}"
            )
            OptimizationVisualizer.display_optimization_summary(results)

if __name__ == "__main__":
    main() 