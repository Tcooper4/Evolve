"""Causal Inference Engine for Market Driver Analysis.

This module provides causal inference capabilities using DoWhy and CausalNex
to identify causal relationships between market factors and price movements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# Try to import causal inference libraries
try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    CausalModel = None

try:
    import causallearn
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    CAUSALLEARN_AVAILABLE = True
except ImportError:
    CAUSALLEARN_AVAILABLE = False
    pc = fisherz = None

logger = logging.getLogger(__name__)

class CausalDriverAnalyzer:
    """Main causal driver analysis engine."""
    
    def __init__(self):
        """Initialize the causal analyzer."""
        self.causal_models = {}
        self.analysis_results = {}
        
    def create_market_dataset(self, price_data: pd.DataFrame, 
                            macro_data: Optional[pd.DataFrame] = None,
                            sentiment_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create dataset for causal analysis."""
        dataset = price_data.copy()
        
        # Add technical indicators
        dataset['returns'] = dataset['Close'].pct_change()
        dataset['volatility'] = dataset['returns'].rolling(20).std()
        dataset['ma_5'] = dataset['Close'].rolling(5).mean()
        dataset['ma_20'] = dataset['Close'].rolling(20).mean()
        dataset['rsi'] = self._calculate_rsi(dataset['Close'])
        dataset['volume_ma'] = dataset['Volume'].rolling(20).mean()
        dataset['volume_ratio'] = dataset['Volume'] / dataset['volume_ma']
        
        # Add macro data if available
        if macro_data is not None:
            for col in macro_data.columns:
                if col not in dataset.columns:
                    dataset[col] = macro_data[col]
        
        # Add sentiment data if available
        if sentiment_data is not None:
            for col in sentiment_data.columns:
                if col not in dataset.columns:
                    dataset[col] = sentiment_data[col]
        
        # Remove NaN values
        dataset = dataset.dropna()
        
        return dataset
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def identify_causal_structure(self, data: pd.DataFrame, 
                                target_variable: str = 'returns',
                                method: str = 'pc') -> Dict[str, Any]:
        """Identify causal structure using constraint-based methods."""
        if not CAUSALLEARN_AVAILABLE:
            logger.error("CausalLearn not available")
            return {}
        
        try:
            # Prepare data for causal discovery
            numeric_data = data.select_dtypes(include=[np.number])
            numeric_data = numeric_data.dropna()
            
            # Run PC algorithm
            if method == 'pc':
                cg = pc(numeric_data.values, alpha=0.05, indep_test=fisherz)
                
                # Convert to adjacency matrix
                adj_matrix = cg.G.graph
                
                # Create causal graph structure
                variables = list(numeric_data.columns)
                causal_structure = {
                    'variables': variables,
                    'adjacency_matrix': adj_matrix.tolist(),
                    'edges': self._extract_edges(adj_matrix, variables),
                    'method': method
                }
                
                logger.info(f"Identified causal structure with {len(causal_structure['edges'])} edges")
                return causal_structure
            
            else:
                logger.error(f"Unknown method: {method}")
                return {}
                
        except Exception as e:
            logger.error(f"Error in causal structure identification: {e}")
            return {}
    
    def _extract_edges(self, adj_matrix: np.ndarray, variables: List[str]) -> List[Dict]:
        """Extract edges from adjacency matrix."""
        edges = []
        for i in range(len(variables)):
            for j in range(len(variables)):
                if adj_matrix[i, j] != 0:
                    edges.append({
                        'from': variables[i],
                        'to': variables[j],
                        'type': 'directed' if adj_matrix[i, j] == 1 else 'undirected'
                    })
        return edges
    
    def estimate_causal_effects(self, data: pd.DataFrame,
                              treatment: str,
                              outcome: str,
                              common_causes: List[str] = None) -> Dict[str, Any]:
        """Estimate causal effects using DoWhy."""
        if not DOWHY_AVAILABLE:
            logger.error("DoWhy not available")
            return {}
        
        try:
            # Prepare data
            analysis_data = data[[treatment, outcome] + (common_causes or [])].dropna()
            
            # Create causal model
            model = CausalModel(
                data=analysis_data,
                treatment=treatment,
                outcome=outcome,
                common_causes=common_causes or []
            )
            
            # Identify causal effect
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            
            # Estimate causal effect
            estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
            
            # Refute results
            refutation_results = {}
            
            # Placebo treatment test
            placebo_refuter = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
            refutation_results['placebo_test'] = {
                'new_effect': placebo_refuter.new_effect,
                'p_value': placebo_refuter.refutation_result['p_value']
            }
            
            # Data subset test
            subset_refuter = model.refute_estimate(identified_estimand, estimate, method_name="data_subset_refuter")
            refutation_results['subset_test'] = {
                'new_effect': subset_refuter.new_effect,
                'p_value': subset_refuter.refutation_result['p_value']
            }
            
            results = {
                'treatment': treatment,
                'outcome': outcome,
                'common_causes': common_causes,
                'estimated_effect': estimate.value,
                'confidence_intervals': estimate.get_confidence_intervals(),
                'refutation_results': refutation_results,
                'model_summary': str(estimate)
            }
            
            logger.info(f"Estimated causal effect: {treatment} -> {outcome} = {estimate.value:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in causal effect estimation: {e}")
            return {}
    
    def analyze_market_drivers(self, data: pd.DataFrame,
                             target_variable: str = 'returns',
                             key_drivers: List[str] = None) -> Dict[str, Any]:
        """Comprehensive market driver analysis."""
        
        if key_drivers is None:
            key_drivers = ['volatility', 'volume_ratio', 'rsi', 'ma_5', 'ma_20']
        
        analysis_results = {
            'causal_structure': {},
            'causal_effects': {},
            'driver_importance': {},
            'summary': {}
        }
        
        # Identify causal structure
        causal_structure = self.identify_causal_structure(data, target_variable)
        analysis_results['causal_structure'] = causal_structure
        
        # Estimate causal effects for each driver
        for driver in key_drivers:
            if driver in data.columns:
                effect = self.estimate_causal_effects(
                    data=data,
                    treatment=driver,
                    outcome=target_variable,
                    common_causes=[d for d in key_drivers if d != driver]
                )
                analysis_results['causal_effects'][driver] = effect
        
        # Calculate driver importance
        driver_importance = {}
        for driver, effect in analysis_results['causal_effects'].items():
            if effect and 'estimated_effect' in effect:
                driver_importance[driver] = abs(effect['estimated_effect'])
        
        # Sort by importance
        sorted_drivers = sorted(driver_importance.items(), key=lambda x: x[1], reverse=True)
        analysis_results['driver_importance'] = dict(sorted_drivers)
        
        # Create summary
        analysis_results['summary'] = {
            'total_drivers_analyzed': len(key_drivers),
            'significant_drivers': len([d for d in driver_importance.values() if d > 0.01]),
            'top_driver': sorted_drivers[0][0] if sorted_drivers else None,
            'analysis_date': pd.Timestamp.now().isoformat()
        }
        
        return analysis_results
    
    def create_causal_graph(self, causal_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualization-ready causal graph."""
        if not causal_structure or 'edges' not in causal_structure:
            return {}
        
        nodes = []
        edges = []
        
        # Create nodes
        for variable in causal_structure.get('variables', []):
            nodes.append({
                'id': variable,
                'label': variable,
                'type': 'variable'
            })
        
        # Create edges
        for edge in causal_structure.get('edges', []):
            edges.append({
                'source': edge['from'],
                'target': edge['to'],
                'type': edge['type']
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'method': causal_structure.get('method', 'unknown')
            }
        }
    
    def get_driver_confidence(self, causal_effects: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for causal drivers."""
        confidence_scores = {}
        
        for driver, effect in causal_effects.items():
            if not effect:
                confidence_scores[driver] = 0.0
                continue
            
            # Calculate confidence based on multiple factors
            effect_magnitude = abs(effect.get('estimated_effect', 0))
            refutation_results = effect.get('refutation_results', {})
            
            # Check refutation tests
            placebo_p_value = refutation_results.get('placebo_test', {}).get('p_value', 1.0)
            subset_p_value = refutation_results.get('subset_test', {}).get('p_value', 1.0)
            
            # Calculate confidence score (0-1)
            confidence = 0.0
            
            # Effect magnitude contribution (0-0.4)
            if effect_magnitude > 0.01:
                confidence += min(0.4, effect_magnitude * 10)
            
            # Placebo test contribution (0-0.3)
            if placebo_p_value < 0.05:
                confidence += 0.3
            elif placebo_p_value < 0.1:
                confidence += 0.15
            
            # Subset test contribution (0-0.3)
            if subset_p_value < 0.05:
                confidence += 0.3
            elif subset_p_value < 0.1:
                confidence += 0.15
            
            confidence_scores[driver] = min(1.0, confidence)
        
        return confidence_scores
    
    def save_analysis(self, analysis_results: Dict[str, Any], 
                     filepath: str = "reports/causal_analysis.json") -> bool:
        """Save causal analysis results."""
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            logger.info(f"Saved causal analysis to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving causal analysis: {e}")
            return False

# Global causal analyzer instance
causal_analyzer = CausalDriverAnalyzer()

def get_causal_analyzer() -> CausalDriverAnalyzer:
    """Get the global causal analyzer instance."""
    return causal_analyzer 