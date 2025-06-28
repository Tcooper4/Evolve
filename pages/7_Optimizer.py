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
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Import from consolidated trading.optimization module
try:
    from trading.optimization import OptimizerFactory, StrategyOptimizer, OptimizationVisualizer
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Optimization module import failed: {e}")
    OPTIMIZATION_AVAILABLE = False
    st.session_state["status"] = "fallback activated"

# Import AgentHub for unified agent routing
try:
    from core.agent_hub import AgentHub
    AGENT_HUB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"AgentHub import failed: {e}")
    AGENT_HUB_AVAILABLE = False

from trading.agents.strategy_switcher import StrategySwitcher
from trading.utils.memory_logger import MemoryLogger

def load_strategy_data() -> pd.DataFrame:
    """Load strategy performance data."""
    # TODO: Implement data loading
    return pd.DataFrame()

def main():
    st.title("Strategy Optimizer")
    
    # Initialize AgentHub if available
    if AGENT_HUB_AVAILABLE and 'agent_hub' not in st.session_state:
        st.session_state['agent_hub'] = AgentHub()
    
    # Natural Language Input Section
    if AGENT_HUB_AVAILABLE:
        st.subheader("🤖 AI Agent Interface")
        st.markdown("Ask for optimization help or request specific optimizations:")
        
        user_prompt = st.text_area(
            "What optimization would you like to perform?",
            placeholder="e.g., 'Optimize RSI strategy for AAPL' or 'Find best parameters for MACD strategy'",
            height=100
        )
        
        if st.button("🚀 Process with AI Agent"):
            if user_prompt:
                with st.spinner("Processing optimization request..."):
                    try:
                        agent_hub = st.session_state['agent_hub']
                        response = agent_hub.route(user_prompt)
                        
                        st.subheader("🤖 AI Response")
                        st.write(response['content'])
                        
                        if response['type'] == 'fallback':
                            st.warning("Using fallback optimization interface")
                        
                    except Exception as e:
                        st.error(f"Failed to process request: {e}")
                        logger.error(f"AgentHub error: {e}")
            else:
                st.warning("Please enter a prompt to process.")
        
        st.divider()
    
    # Check if optimization is available
    if not OPTIMIZATION_AVAILABLE:
        st.error("Optimization module not available")
        st.info("Please check that the trading.optimization module is properly installed.")
        return
    
    # Initialize components
    try:
        strategy_switcher = StrategySwitcher()
        memory_logger = MemoryLogger()
        
        # Create optimizer config
        optimizer_config = {
            "name": "strategy_optimizer",
            "optimizer_type": "bayesian",
            "n_initial_points": 10,
            "n_iterations": 50,
            "primary_metric": "sharpe_ratio"
        }
        optimizer = StrategyOptimizer(optimizer_config)
        
        logger.info("StrategyOptimizer initialized successfully")
        
    except Exception as e:
        st.error(f"Failed to initialize optimization components: {e}")
        logger.error(f"Optimization initialization error: {e}")
        st.session_state["status"] = "fallback activated"
        return
    
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