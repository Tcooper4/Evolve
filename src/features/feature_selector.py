"""
Feature Selector

Selects features using various methods including SHAP and permutation importance
for explainability and transparency.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
# Try to import scikit-learn
try:
    from sklearn.feature_selection import (
        SelectKBest, f_regression, f_classif, mutual_info_regression, 
        mutual_info_classif, RFE, SelectFromModel
    )
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print("âš ï¸ scikit-learn not available. Disabling feature selection capabilities.")
    print(f"   Missing: {e}")
    SelectKBest = None
    f_regression = None
    f_classif = None
    mutual_info_regression = None
    mutual_info_classif = None
    RFE = None
    SelectFromModel = None
    RandomForestRegressor = None
    RandomForestClassifier = None
    Lasso = None
    Ridge = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False
import warnings

logger = logging.getLogger(__name__)

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")


@dataclass
class FeatureScore:
    """Feature importance score with metadata."""
    feature_name: str
    score: float
    rank: int
    method: str
    metadata: Dict[str, Any]


class FeatureSelector:
    """Feature selector with multiple methods and explainability."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the feature selector.
        
        Args:
            config: Configuration dictionary
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not available. Cannot create FeatureSelector.")
            
        self.config = config or {}
        self.selection_methods = {
            "correlation": self._correlation_selection,
            "mutual_info": self._mutual_info_selection,
            "f_score": self._f_score_selection,
            "lasso": self._lasso_selection,
            "random_forest": self._random_forest_selection,
            "shap": self._shap_selection,
            "permutation": self._permutation_selection,
            "rfe": self._rfe_selection,
        }
        self.logger = logging.getLogger(__name__)
        self.feature_scores = {}
        
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "random_forest",
        n_features: Optional[int] = None,
        task_type: str = "regression",
        **kwargs
    ) -> Dict[str, float]:
        """
        Select features using specified method.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Selection method
            n_features: Number of features to select (None for all)
            task_type: 'regression' or 'classification'
            **kwargs: Additional method-specific parameters
            
        Returns:
            Dictionary mapping feature names to scores
        """
        if method not in self.selection_methods:
            raise ValueError(f"Unknown selection method: {method}")
            
        if X.empty or y.empty:
            raise ValueError("Input data is empty")
            
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
            
        # Handle NaN values
        X_clean, y_clean = self._handle_nan_values(X, y)
        
        # Select features
        feature_scores = self.selection_methods[method](
            X_clean, y_clean, n_features, task_type, **kwargs
        )
        
        # Store scores
        self.feature_scores[method] = feature_scores
        
        # Convert to simple dict if requested
        if kwargs.get("return_simple_dict", True):
            return {name: score.score for name, score in feature_scores.items()}
        
        return feature_scores
        
    def _handle_nan_values(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle NaN values in features and target."""
        # Remove rows with NaN in target
        valid_indices = ~y.isna()
        X_clean = X[valid_indices].copy()
        y_clean = y[valid_indices].copy()
        
        # Fill NaN in features with median
        X_clean = X_clean.fillna(X_clean.median())
        
        return X_clean, y_clean
        
    def _correlation_selection(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_features: Optional[int],
        task_type: str,
        **kwargs
    ) -> Dict[str, FeatureScore]:
        """Select features based on correlation with target."""
        scores = {}
        
        for column in X.columns:
            correlation = abs(X[column].corr(y))
            scores[column] = FeatureScore(
                feature_name=column,
                score=correlation,
                rank=0,  # Will be set later
                method="correlation",
                metadata={"correlation_type": "absolute"}
            )
            
        return self._rank_and_limit_features(scores, n_features)
        
    def _mutual_info_selection(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_features: Optional[int],
        task_type: str,
        **kwargs
    ) -> Dict[str, FeatureScore]:
        """Select features based on mutual information."""
        if task_type == "regression":
            mi_scores = mutual_info_regression(X, y, random_state=42)
        else:
            mi_scores = mutual_info_classif(X, y, random_state=42)
            
        scores = {}
        for i, column in enumerate(X.columns):
            scores[column] = FeatureScore(
                feature_name=column,
                score=mi_scores[i],
                rank=0,
                method="mutual_info",
                metadata={"task_type": task_type}
            )
            
        return self._rank_and_limit_features(scores, n_features)
        
    def _f_score_selection(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_features: Optional[int],
        task_type: str,
        **kwargs
    ) -> Dict[str, FeatureScore]:
        """Select features based on F-score."""
        if task_type == "regression":
            f_scores, _ = f_regression(X, y)
        else:
            f_scores, _ = f_classif(X, y)
            
        scores = {}
        for i, column in enumerate(X.columns):
            scores[column] = FeatureScore(
                feature_name=column,
                score=f_scores[i],
                rank=0,
                method="f_score",
                metadata={"task_type": task_type}
            )
            
        return self._rank_and_limit_features(scores, n_features)
        
    def _lasso_selection(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_features: Optional[int],
        task_type: str,
        **kwargs
    ) -> Dict[str, FeatureScore]:
        """Select features using Lasso regularization."""
        alpha = kwargs.get("alpha", 0.01)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if task_type == "regression":
            model = Lasso(alpha=alpha, random_state=42)
        else:
            # For classification, use Ridge as Lasso is for regression
            model = Ridge(alpha=alpha, random_state=42)
            
        model.fit(X_scaled, y)
        
        scores = {}
        for i, column in enumerate(X.columns):
            scores[column] = FeatureScore(
                feature_name=column,
                score=abs(model.coef_[i]),
                rank=0,
                method="lasso",
                metadata={"alpha": alpha, "task_type": task_type}
            )
            
        return self._rank_and_limit_features(scores, n_features)
        
    def _random_forest_selection(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_features: Optional[int],
        task_type: str,
        **kwargs
    ) -> Dict[str, FeatureScore]:
        """Select features using Random Forest importance."""
        n_estimators = kwargs.get("n_estimators", 100)
        
        if task_type == "regression":
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            
        model.fit(X, y)
        
        scores = {}
        for i, column in enumerate(X.columns):
            scores[column] = FeatureScore(
                feature_name=column,
                score=model.feature_importances_[i],
                rank=0,
                method="random_forest",
                metadata={"n_estimators": n_estimators, "task_type": task_type}
            )
            
        return self._rank_and_limit_features(scores, n_features)
        
    def _shap_selection(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_features: Optional[int],
        task_type: str,
        **kwargs
    ) -> Dict[str, FeatureScore]:
        """Select features using SHAP values."""
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available, falling back to random forest")
            return self._random_forest_selection(X, y, n_features, task_type, **kwargs)
            
        n_estimators = kwargs.get("n_estimators", 100)
        sample_size = kwargs.get("sample_size", min(1000, len(X)))
        
        # Sample data for SHAP computation
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]
        else:
            X_sample = X
            y_sample = y
            
        if task_type == "regression":
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            
        model.fit(X_sample, y_sample)
        
        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # For classification, use mean of absolute SHAP values
        if task_type == "classification" and len(shap_values) > 1:
            shap_values = np.mean(np.abs(shap_values), axis=0)
        elif task_type == "classification":
            shap_values = np.abs(shap_values)
            
        # Calculate mean absolute SHAP values
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        scores = {}
        for i, column in enumerate(X.columns):
            scores[column] = FeatureScore(
                feature_name=column,
                score=mean_shap[i],
                rank=0,
                method="shap",
                metadata={
                    "n_estimators": n_estimators,
                    "task_type": task_type,
                    "sample_size": len(X_sample)
                }
            )
            
        return self._rank_and_limit_features(scores, n_features)
        
    def _permutation_selection(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_features: Optional[int],
        task_type: str,
        **kwargs
    ) -> Dict[str, FeatureScore]:
        """Select features using permutation importance."""
        n_repeats = kwargs.get("n_repeats", 10)
        n_estimators = kwargs.get("n_estimators", 100)
        
        if task_type == "regression":
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            
        model.fit(X, y)
        
        # Calculate permutation importance
        from sklearn.inspection import permutation_importance
        perm_importance = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=42
        )
        
        scores = {}
        for i, column in enumerate(X.columns):
            scores[column] = FeatureScore(
                feature_name=column,
                score=perm_importance.importances_mean[i],
                rank=0,
                method="permutation",
                metadata={
                    "n_repeats": n_repeats,
                    "n_estimators": n_estimators,
                    "task_type": task_type,
                    "std": perm_importance.importances_std[i]
                }
            )
            
        return self._rank_and_limit_features(scores, n_features)
        
    def _rfe_selection(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_features: Optional[int],
        task_type: str,
        **kwargs
    ) -> Dict[str, FeatureScore]:
        """Select features using Recursive Feature Elimination."""
        if n_features is None:
            n_features = len(X.columns) // 2
            
        if task_type == "regression":
            model = RandomForestRegressor(n_estimators=50, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            
        rfe = RFE(estimator=model, n_features_to_select=n_features)
        rfe.fit(X, y)
        
        scores = {}
        for i, column in enumerate(X.columns):
            scores[column] = FeatureScore(
                feature_name=column,
                score=1.0 if rfe.support_[i] else 0.0,
                rank=rfe.ranking_[i],
                method="rfe",
                metadata={
                    "n_features_selected": n_features,
                    "task_type": task_type,
                    "selected": rfe.support_[i]
                }
            )
            
        return scores
        
    def _rank_and_limit_features(
        self, 
        scores: Dict[str, FeatureScore], 
        n_features: Optional[int]
    ) -> Dict[str, FeatureScore]:
        """Rank features by score and limit to n_features."""
        # Sort by score (descending)
        sorted_features = sorted(scores.items(), key=lambda x: x[1].score, reverse=True)
        
        # Update ranks
        for rank, (feature_name, feature_score) in enumerate(sorted_features):
            feature_score.rank = rank + 1
            
        # Limit to n_features if specified
        if n_features is not None:
            sorted_features = sorted_features[:n_features]
            
        return dict(sorted_features)
        
    def select_features_multiple_methods(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        methods: List[str] = None,
        n_features: Optional[int] = None,
        task_type: str = "regression",
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """Select features using multiple methods and compare results."""
        if methods is None:
            methods = ["correlation", "mutual_info", "random_forest", "shap"]
            
        results = {}
        
        for method in methods:
            try:
                self.logger.info(f"Running {method} feature selection")
                scores = self.select_features(
                    X, y, method, n_features, task_type, 
                    return_simple_dict=True, **kwargs
                )
                results[method] = scores
            except Exception as e:
                self.logger.error(f"Error in {method} feature selection: {e}")
                continue
                
        return results
        
    def get_feature_importance_summary(
        self, 
        method: str = "random_forest"
    ) -> pd.DataFrame:
        """Get a summary of feature importance for a method."""
        if method not in self.feature_scores:
            raise ValueError(f"No scores available for method: {method}")
            
        scores = self.feature_scores[method]
        
        summary_data = []
        for feature_name, feature_score in scores.items():
            summary_data.append({
                "feature_name": feature_name,
                "score": feature_score.score,
                "rank": feature_score.rank,
                "method": feature_score.method,
                **feature_score.metadata
            })
            
        return pd.DataFrame(summary_data).sort_values("rank")
        
    def plot_feature_importance(
        self, 
        method: str = "random_forest",
        top_n: int = 20
    ) -> None:
        """Plot feature importance for a method."""
        try:
            import matplotlib.pyplot as plt
            
            summary = self.get_feature_importance_summary(method)
            top_features = summary.head(top_n)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_features["score"])
            plt.yticks(range(len(top_features)), top_features["feature_name"])
            plt.xlabel("Importance Score")
            plt.title(f"Feature Importance - {method.upper()}")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            self.logger.warning("matplotlib not available for plotting")
            
    def get_consensus_features(
        self, 
        methods: List[str] = None,
        min_rank: int = 10
    ) -> List[str]:
        """Get features that are important across multiple methods."""
        if methods is None:
            methods = list(self.feature_scores.keys())
            
        feature_ranks = {}
        
        for method in methods:
            if method in self.feature_scores:
                scores = self.feature_scores[method]
                for feature_name, feature_score in scores.items():
                    if feature_name not in feature_ranks:
                        feature_ranks[feature_name] = []
                    feature_ranks[feature_name].append(feature_score.rank)
                    
        # Find features that rank well across methods
        consensus_features = []
        for feature_name, ranks in feature_ranks.items():
            if len(ranks) >= len(methods) // 2 and all(rank <= min_rank for rank in ranks):
                consensus_features.append(feature_name)
                
        return sorted(consensus_features)
        
    def compare_methods(self) -> pd.DataFrame:
        """Compare feature rankings across different methods."""
        if not self.feature_scores:
            return pd.DataFrame()
            
        comparison_data = []
        
        for method, scores in self.feature_scores.items():
            for feature_name, feature_score in scores.items():
                comparison_data.append({
                    "feature_name": feature_name,
                    "method": method,
                    "score": feature_score.score,
                    "rank": feature_score.rank
                })
                
        df = pd.DataFrame(comparison_data)
        
        # Pivot to show ranks across methods
        rank_pivot = df.pivot(index="feature_name", columns="method", values="rank")
        score_pivot = df.pivot(index="feature_name", columns="method", values="score")
        
        return {
            "ranks": rank_pivot,
            "scores": score_pivot,
            "raw_data": df
        }
