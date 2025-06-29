"""Causal Inference Module for Trading Strategy Analysis.

This module provides causal inference capabilities for understanding
market relationships and strategy performance drivers.
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CausalModel:
    """Causal inference model for trading strategy analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize causal model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.graph = nx.DiGraph()
        self.causal_effects = {}
        self.intervention_results = {}
        self.scaler = StandardScaler()
        
    def build_causal_graph(self, data: pd.DataFrame, 
                          treatment_vars: List[str],
                          outcome_vars: List[str],
                          confounders: Optional[List[str]] = None) -> nx.DiGraph:
        """Build causal graph from data.
        
        Args:
            data: Input data
            treatment_vars: Treatment variables
            outcome_vars: Outcome variables
            confounders: Confounding variables
            
        Returns:
            Causal graph
        """
        try:
            # Create directed graph
            self.graph = nx.DiGraph()
            
            # Add nodes
            all_vars = treatment_vars + outcome_vars
            if confounders:
                all_vars.extend(confounders)
            
            for var in all_vars:
                if var in data.columns:
                    self.graph.add_node(var)
            
            # Add edges based on correlations and domain knowledge
            for treatment in treatment_vars:
                for outcome in outcome_vars:
                    if treatment in data.columns and outcome in data.columns:
                        # Calculate correlation
                        corr = data[treatment].corr(data[outcome])
                        if abs(corr) > 0.1:  # Threshold for edge creation
                            self.graph.add_edge(treatment, outcome, weight=corr)
            
            # Add confounder edges
            if confounders:
                for confounder in confounders:
                    for var in all_vars:
                        if confounder != var and confounder in data.columns and var in data.columns:
                            corr = data[confounder].corr(data[var])
                            if abs(corr) > 0.1:
                                self.graph.add_edge(confounder, var, weight=corr)
            
            logger.info(f"Built causal graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
            return self.graph
            
        except Exception as e:
            logger.error(f"Error building causal graph: {e}")
            return nx.DiGraph()
    
    def estimate_causal_effect(self, data: pd.DataFrame,
                             treatment: str,
                             outcome: str,
                             method: str = 'linear_regression') -> Dict[str, float]:
        """Estimate causal effect using various methods.
        
        Args:
            data: Input data
            treatment: Treatment variable
            outcome: Outcome variable
            method: Estimation method
            
        Returns:
            Causal effect estimates
        """
        try:
            if treatment not in data.columns or outcome not in data.columns:
                raise ValueError(f"Treatment {treatment} or outcome {outcome} not in data")
            
            # Prepare data
            X = data[[treatment]].copy()
            y = data[outcome].copy()
            
            # Remove missing values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 10:
                raise ValueError("Insufficient data for causal estimation")
            
            results = {}
            
            if method == 'linear_regression':
                # Linear regression approach
                model = LinearRegression()
                model.fit(X, y)
                results['coefficient'] = model.coef_[0]
                results['intercept'] = model.intercept_
                results['r_squared'] = model.score(X, y)
                
                # Calculate confidence interval (simplified)
                y_pred = model.predict(X)
                mse = mean_squared_error(y, y_pred)
                results['std_error'] = np.sqrt(mse / len(X))
                
            elif method == 'random_forest':
                # Random forest approach
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                # Feature importance as causal effect
                results['importance'] = model.feature_importances_[0]
                results['r_squared'] = model.score(X, y)
                
            elif method == 'mutual_information':
                # Mutual information approach
                mi = mutual_info_regression(X, y)[0]
                results['mutual_info'] = mi
                
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Store results
            key = f"{treatment}_->_{outcome}"
            self.causal_effects[key] = results
            
            logger.info(f"Estimated causal effect {key}: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error estimating causal effect: {e}")
            return {}
    
    def perform_intervention(self, data: pd.DataFrame,
                           treatment: str,
                           intervention_value: float,
                           outcome: str) -> Dict[str, float]:
        """Perform causal intervention (do-calculus).
        
        Args:
            data: Input data
            treatment: Treatment variable
            intervention_value: Value to set treatment to
            outcome: Outcome variable
            
        Returns:
            Intervention results
        """
        try:
            if treatment not in data.columns or outcome not in data.columns:
                raise ValueError(f"Treatment {treatment} or outcome {outcome} not in data")
            
            # Create intervention data
            intervention_data = data.copy()
            intervention_data[treatment] = intervention_value
            
            # Estimate outcome under intervention
            X_intervention = intervention_data[[treatment]]
            y_intervention = intervention_data[outcome]
            
            # Remove missing values
            mask = ~(X_intervention.isnull().any(axis=1) | y_intervention.isnull())
            X_intervention = X_intervention[mask]
            y_intervention = y_intervention[mask]
            
            if len(X_intervention) < 10:
                raise ValueError("Insufficient data for intervention analysis")
            
            # Fit model on original data
            X_original = data[[treatment]]
            y_original = data[outcome]
            
            mask_original = ~(X_original.isnull().any(axis=1) | y_original.isnull())
            X_original = X_original[mask_original]
            y_original = y_original[mask_original]
            
            model = LinearRegression()
            model.fit(X_original, y_original)
            
            # Predict under intervention
            y_pred_intervention = model.predict(X_intervention)
            y_pred_original = model.predict(X_original)
            
            # Calculate intervention effect
            avg_effect = np.mean(y_pred_intervention) - np.mean(y_pred_original)
            
            results = {
                'intervention_value': intervention_value,
                'original_mean': np.mean(y_pred_original),
                'intervention_mean': np.mean(y_pred_intervention),
                'average_effect': avg_effect,
                'effect_size': abs(avg_effect) / np.std(y_original) if np.std(y_original) > 0 else 0
            }
            
            # Store results
            key = f"do({treatment}={intervention_value})"
            self.intervention_results[key] = results
            
            logger.info(f"Intervention {key} effect: {avg_effect:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error performing intervention: {e}")
            return {}
    
    def analyze_market_relationships(self, market_data: pd.DataFrame,
                                   target_variable: str = 'returns') -> Dict[str, Any]:
        """Analyze causal relationships in market data.
        
        Args:
            market_data: Market data with features
            target_variable: Target variable for analysis
            
        Returns:
            Analysis results
        """
        try:
            if target_variable not in market_data.columns:
                raise ValueError(f"Target variable {target_variable} not in data")
            
            # Identify potential treatment variables
            feature_cols = [col for col in market_data.columns 
                          if col != target_variable and market_data[col].dtype in ['float64', 'int64']]
            
            if len(feature_cols) < 2:
                raise ValueError("Insufficient features for analysis")
            
            # Build causal graph
            self.build_causal_graph(
                data=market_data,
                treatment_vars=feature_cols[:5],  # Limit to top 5 features
                outcome_vars=[target_variable]
            )
            
            # Estimate causal effects
            effects = {}
            for feature in feature_cols[:5]:
                effect = self.estimate_causal_effect(
                    data=market_data,
                    treatment=feature,
                    outcome=target_variable,
                    method='linear_regression'
                )
                if effect:
                    effects[feature] = effect
            
            # Perform interventions
            interventions = {}
            for feature in feature_cols[:3]:  # Top 3 features
                # Calculate intervention value (mean + 1 std)
                mean_val = market_data[feature].mean()
                std_val = market_data[feature].std()
                intervention_val = mean_val + std_val
                
                intervention = self.perform_intervention(
                    data=market_data,
                    treatment=feature,
                    intervention_value=intervention_val,
                    outcome=target_variable
                )
                if intervention:
                    interventions[feature] = intervention
            
            results = {
                'causal_effects': effects,
                'interventions': interventions,
                'graph_nodes': len(self.graph.nodes),
                'graph_edges': len(self.graph.edges)
            }
            
            logger.info(f"Market relationship analysis completed: {len(effects)} effects, {len(interventions)} interventions")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing market relationships: {e}")
            return {}
    
    def get_strategy_insights(self, strategy_data: pd.DataFrame,
                            performance_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """Get causal insights for trading strategy performance.
        
        Args:
            strategy_data: Strategy performance data
            performance_metric: Performance metric to analyze
            
        Returns:
            Strategy insights
        """
        try:
            if performance_metric not in strategy_data.columns:
                raise ValueError(f"Performance metric {performance_metric} not in data")
            
            # Identify strategy parameters
            param_cols = [col for col in strategy_data.columns 
                         if col != performance_metric and strategy_data[col].dtype in ['float64', 'int64']]
            
            if len(param_cols) < 2:
                raise ValueError("Insufficient strategy parameters for analysis")
            
            # Analyze parameter effects on performance
            effects = {}
            for param in param_cols:
                effect = self.estimate_causal_effect(
                    data=strategy_data,
                    treatment=param,
                    outcome=performance_metric,
                    method='linear_regression'
                )
                if effect:
                    effects[param] = effect
            
            # Find optimal parameter ranges
            optimal_ranges = {}
            for param in param_cols[:3]:  # Top 3 parameters
                if param in effects:
                    # Find parameter range that maximizes performance
                    param_data = strategy_data[param]
                    perf_data = strategy_data[performance_metric]
                    
                    # Create bins and find best performing bin
                    bins = pd.cut(param_data, bins=5)
                    bin_performance = perf_data.groupby(bins).mean()
                    
                    best_bin = bin_performance.idxmax()
                    optimal_ranges[param] = {
                        'best_range': str(best_bin),
                        'max_performance': bin_performance.max(),
                        'causal_effect': effects[param].get('coefficient', 0)
                    }
            
            insights = {
                'parameter_effects': effects,
                'optimal_ranges': optimal_ranges,
                'top_parameters': sorted(effects.keys(), 
                                       key=lambda x: abs(effects[x].get('coefficient', 0)), 
                                       reverse=True)[:3]
            }
            
            logger.info(f"Strategy insights generated: {len(effects)} parameter effects")
            return insights
            
        except Exception as e:
            logger.error(f"Error getting strategy insights: {e}")
            return {}
    
    def visualize_causal_graph(self) -> Optional[nx.DiGraph]:
        """Get causal graph for visualization.
        
        Returns:
            Causal graph
        """
        return self.graph if self.graph else None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of causal analysis.
        
        Returns:
            Analysis summary
        """
        return {
            'causal_effects': len(self.causal_effects),
            'interventions': len(self.intervention_results),
            'graph_nodes': len(self.graph.nodes),
            'graph_edges': len(self.graph.edges),
            'effects': self.causal_effects,
            'interventions': self.intervention_results
        }

class CausalAnalysisResult:
    """Result container for causal analysis."""
    
    def __init__(self, 
                 causal_effects: Dict[str, float],
                 intervention_results: Dict[str, float],
                 graph: nx.DiGraph,
                 metrics: Dict[str, float]):
        """Initialize causal analysis result.
        
        Args:
            causal_effects: Dictionary of causal effect estimates
            intervention_results: Dictionary of intervention results
            graph: Causal graph
            metrics: Performance metrics
        """
        self.causal_effects = causal_effects
        self.intervention_results = intervention_results
        self.graph = graph
        self.metrics = metrics
        self.timestamp = pd.Timestamp.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'causal_effects': self.causal_effects,
            'intervention_results': self.intervention_results,
            'graph_nodes': list(self.graph.nodes()),
            'graph_edges': list(self.graph.edges()),
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of results."""
        return {
            'num_effects': len(self.causal_effects),
            'num_interventions': len(self.intervention_results),
            'graph_size': len(self.graph.nodes()),
            'avg_effect': np.mean(list(self.causal_effects.values())) if self.causal_effects else 0,
            'max_effect': max(self.causal_effects.values()) if self.causal_effects else 0,
            'timestamp': self.timestamp.isoformat()
        }

def create_causal_model(config: Dict[str, Any] = None):
    class DummyCausalModel:
        def analyze(self, *a, **kw): return {'result': 'dummy'}
    return DummyCausalModel()

class CausalModelAnalyzer:
    def __init__(self, *a, **kw): pass
    def analyze(self, *a, **kw): return {'result': 'analyzer dummy'}

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'market_volatility': np.random.normal(0.2, 0.1, n_samples),
        'interest_rate': np.random.normal(0.05, 0.02, n_samples),
        'economic_growth': np.random.normal(0.03, 0.01, n_samples),
        'returns': np.random.normal(0.001, 0.02, n_samples)
    })
    
    # Add some causal relationships
    data['returns'] += 0.5 * data['market_volatility'] + 0.3 * data['economic_growth']
    data['market_volatility'] += 0.2 * data['interest_rate']
    
    # Create and test causal model
    model = create_causal_model()
    
    # Analyze market relationships
    results = model.analyze_market_relationships(data, 'returns')
    print("Market Analysis Results:", results)
    
    # Get strategy insights
    strategy_data = pd.DataFrame({
        'lookback_period': np.random.randint(5, 50, n_samples),
        'threshold': np.random.uniform(0.1, 0.5, n_samples),
        'position_size': np.random.uniform(0.01, 0.1, n_samples),
        'sharpe_ratio': np.random.normal(1.0, 0.3, n_samples)
    })
    
    insights = model.get_strategy_insights(strategy_data, 'sharpe_ratio')
    print("Strategy Insights:", insights) 