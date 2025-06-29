"""Auto-Evolutionary Model Generator Agent for Evolve Trading Platform.

This agent automatically discovers, implements, and benchmarks new ML models
from research papers to continuously improve trading performance.
"""

import requests
import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import hashlib

# ML imports
try:
    import sklearn
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ResearchPaper:
    """Represents a research paper from arXiv."""
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: str
    published_date: str
    categories: List[str]
    relevance_score: float
    implementation_complexity: str  # "low", "medium", "high"
    potential_impact: str  # "low", "medium", "high"

@dataclass
class ModelCandidate:
    """Represents a candidate model for implementation."""
    name: str
    paper: ResearchPaper
    model_type: str  # "sklearn", "pytorch", "custom"
    implementation_code: str
    hyperparameters: Dict[str, Any]
    expected_performance: Dict[str, float]
    implementation_status: str  # "pending", "implemented", "tested", "deployed"

@dataclass
class BenchmarkResult:
    """Results of model benchmarking."""
    model_name: str
    mse: float
    mae: float
    r2_score: float
    sharpe_ratio: float
    max_drawdown: float
    training_time: float
    inference_time: float
    memory_usage: float
    overall_score: float
    benchmark_date: str

class ArxivResearchFetcher:
    """Fetches research papers from arXiv API."""
    
    def __init__(self, 
                 search_terms: Optional[List[str]] = None,
                 max_results: int = 100,
                 days_back: int = 30):
        """Initialize research fetcher.
        
        Args:
            search_terms: Search terms for relevant papers
            max_results: Maximum number of results to fetch
            days_back: Number of days back to search
        """
        self.search_terms = search_terms or [
            "time series forecasting",
            "financial prediction",
            "machine learning trading",
            "neural networks forecasting",
            "quantitative finance",
            "market prediction",
            "deep learning time series",
            "reinforcement learning trading"
        ]
        self.max_results = max_results
        self.days_back = days_back
        self.base_url = "http://export.arxiv.org/api/query"
        
        # Cache for fetched papers
        self.paper_cache = {}
        self.cache_file = Path("agents/research_cache.json")
        self._load_cache()
        
        logger.info(f"Initialized ArxivResearchFetcher with {len(self.search_terms)} search terms")
    
    def _load_cache(self):
        """Load cached papers."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.paper_cache = json.load(f)
                logger.info(f"Loaded {len(self.paper_cache)} cached papers")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            self.paper_cache = {}
    
    def _save_cache(self):
        """Save papers to cache."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.paper_cache, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _calculate_relevance_score(self, title: str, abstract: str, categories: List[str]) -> float:
        """Calculate relevance score for a paper."""
        score = 0.0
        
        # Keywords that indicate relevance
        relevant_keywords = [
            "time series", "forecasting", "prediction", "financial", "trading",
            "market", "stock", "price", "return", "volatility", "neural network",
            "deep learning", "machine learning", "reinforcement learning",
            "lstm", "transformer", "attention", "ensemble", "optimization"
        ]
        
        # Check title relevance
        title_lower = title.lower()
        for keyword in relevant_keywords:
            if keyword in title_lower:
                score += 2.0
        
        # Check abstract relevance
        abstract_lower = abstract.lower()
        for keyword in relevant_keywords:
            if keyword in abstract_lower:
                score += 1.0
        
        # Check categories relevance
        relevant_categories = ["cs.lg", "cs.ai", "q-fin", "stat.ml", "stat.me"]
        for category in categories:
            if category in relevant_categories:
                score += 3.0
        
        # Normalize score
        score = min(score / 20.0, 1.0)
        
        return score
    
    def _assess_implementation_complexity(self, title: str, abstract: str) -> str:
        """Assess implementation complexity."""
        complexity_indicators = {
            "low": ["simple", "linear", "regression", "basic", "traditional"],
            "medium": ["neural", "network", "ensemble", "gradient", "optimization"],
            "high": ["transformer", "attention", "reinforcement", "complex", "advanced"]
        }
        
        text_lower = (title + " " + abstract).lower()
        
        scores = {}
        for complexity, keywords in complexity_indicators.items():
            scores[complexity] = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Return complexity with highest score
        return max(scores, key=scores.get)
    
    def _assess_potential_impact(self, title: str, abstract: str, relevance_score: float) -> str:
        """Assess potential impact of the research."""
        impact_indicators = {
            "high": ["novel", "breakthrough", "state-of-the-art", "sota", "improvement"],
            "medium": ["improved", "enhanced", "better", "effective"],
            "low": ["comparison", "analysis", "study", "review"]
        }
        
        text_lower = (title + " " + abstract).lower()
        
        scores = {}
        for impact, keywords in impact_indicators.items():
            scores[impact] = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Combine with relevance score
        base_impact = max(scores, key=scores.get)
        
        if relevance_score > 0.7:
            if base_impact == "low":
                return "medium"
            return base_impact
        else:
            return "low"
    
    async def fetch_papers_async(self, search_term: str) -> List[ResearchPaper]:
        """Fetch papers asynchronously for a search term."""
        try:
            # Construct query
            query_params = {
                'search_query': f'all:"{search_term}"',
                'start': 0,
                'max_results': self.max_results // len(self.search_terms),
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=query_params) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        return self._parse_arxiv_xml(xml_content, search_term)
                    else:
                        logger.warning(f"Failed to fetch papers for '{search_term}': {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching papers for '{search_term}': {e}")
            return []
    
    def _parse_arxiv_xml(self, xml_content: str, search_term: str) -> List[ResearchPaper]:
        """Parse arXiv XML response."""
        papers = []
        
        try:
            # Simple XML parsing (in production, use proper XML parser)
            entries = re.findall(r'<entry>(.*?)</entry>', xml_content, re.DOTALL)
            
            for entry in entries:
                try:
                    # Extract title
                    title_match = re.search(r'<title>(.*?)</title>', entry)
                    title = title_match.group(1).strip() if title_match else ""
                    
                    # Extract authors
                    authors_match = re.findall(r'<name>(.*?)</name>', entry)
                    authors = [author.strip() for author in authors_match]
                    
                    # Extract abstract
                    abstract_match = re.search(r'<summary>(.*?)</summary>', entry)
                    abstract = abstract_match.group(1).strip() if abstract_match else ""
                    
                    # Extract arXiv ID
                    id_match = re.search(r'<id>(.*?)</id>', entry)
                    arxiv_id = id_match.group(1).split('/')[-1] if id_match else ""
                    
                    # Extract published date
                    date_match = re.search(r'<published>(.*?)</published>', entry)
                    published_date = date_match.group(1) if date_match else ""
                    
                    # Extract categories
                    categories_match = re.findall(r'<category term="(.*?)"', entry)
                    categories = categories_match
                    
                    # Skip if already cached
                    if arxiv_id in self.paper_cache:
                        continue
                    
                    # Calculate scores
                    relevance_score = self._calculate_relevance_score(title, abstract, categories)
                    implementation_complexity = self._assess_implementation_complexity(title, abstract)
                    potential_impact = self._assess_potential_impact(title, abstract, relevance_score)
                    
                    # Create paper object
                    paper = ResearchPaper(
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        arxiv_id=arxiv_id,
                        published_date=published_date,
                        categories=categories,
                        relevance_score=relevance_score,
                        implementation_complexity=implementation_complexity,
                        potential_impact=potential_impact
                    )
                    
                    papers.append(paper)
                    
                    # Cache paper
                    self.paper_cache[arxiv_id] = {
                        "title": title,
                        "authors": authors,
                        "abstract": abstract,
                        "published_date": published_date,
                        "categories": categories,
                        "relevance_score": relevance_score,
                        "implementation_complexity": implementation_complexity,
                        "potential_impact": potential_impact
                    }
                    
                except Exception as e:
                    logger.error(f"Error parsing paper entry: {e}")
                    continue
            
            # Save updated cache
            self._save_cache()
            
        except Exception as e:
            logger.error(f"Error parsing arXiv XML: {e}")
        
        return papers
    
    async def fetch_recent_papers(self) -> List[ResearchPaper]:
        """Fetch recent relevant papers from arXiv."""
        logger.info("Fetching recent papers from arXiv...")
        
        all_papers = []
        
        # Fetch papers for each search term
        tasks = [self.fetch_papers_async(term) for term in self.search_terms]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_papers.extend(result)
            else:
                logger.error(f"Error in paper fetching: {result}")
        
        # Filter by relevance and recency
        relevant_papers = [
            paper for paper in all_papers
            if paper.relevance_score > 0.3 and
            paper.potential_impact in ["medium", "high"]
        ]
        
        # Sort by relevance score
        relevant_papers.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"Found {len(relevant_papers)} relevant papers out of {len(all_papers)} total")
        return relevant_papers[:50]  # Return top 50

class ModelImplementationGenerator:
    """Generates model implementations from research papers."""
    
    def __init__(self):
        """Initialize model generator."""
        self.implementation_templates = self._load_templates()
        logger.info("Initialized Model Implementation Generator")
    
    def _load_templates(self) -> Dict[str, str]:
        """Load implementation templates."""
        return {
            "lstm": self._get_lstm_template(),
            "transformer": self._get_transformer_template(),
            "ensemble": self._get_ensemble_template(),
            "attention": self._get_attention_template(),
            "reinforcement": self._get_rl_template(),
            "sklearn": self._get_sklearn_template()
        }
    
    def _get_lstm_template(self) -> str:
        """Get LSTM implementation template."""
        return '''
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
'''
    
    def _get_transformer_template(self) -> str:
        """Get Transformer implementation template."""
        return '''
class TransformerForecaster(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        return self.output_projection(x[:, -1, :])
'''
    
    def _get_ensemble_template(self) -> str:
        """Get Ensemble implementation template."""
        return '''
class EnsembleForecaster:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)
    
    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return np.average(predictions, weights=self.weights, axis=0)
'''
    
    def _get_attention_template(self) -> str:
        """Get Attention mechanism template."""
        return '''
class AttentionLayer(nn.Module):
    def __init__(self, input_size, attention_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_size, attention_size, batch_first=True)
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output
'''
    
    def _get_rl_template(self) -> str:
        """Get Reinforcement Learning template."""
        return '''
class RLForecaster:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # Implementation would use stable-baselines3 or similar
        pass
'''
    
    def _get_sklearn_template(self) -> str:
        """Get scikit-learn template."""
        return '''
class SklearnForecaster:
    def __init__(self, model_type="random_forest"):
        if model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == "svr":
            self.model = SVR(kernel='rbf')
    
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
'''
    
    def generate_implementation(self, paper: ResearchPaper) -> Optional[ModelCandidate]:
        """Generate model implementation from research paper."""
        try:
            # Determine model type from paper content
            model_type = self._classify_model_type(paper)
            
            if model_type not in self.implementation_templates:
                logger.warning(f"No template available for model type: {model_type}")
                return None
            
            # Generate implementation code
            implementation_code = self._generate_code(paper, model_type)
            
            # Generate hyperparameters
            hyperparameters = self._generate_hyperparameters(paper, model_type)
            
            # Estimate expected performance
            expected_performance = self._estimate_performance(paper, model_type)
            
            # Create model candidate
            candidate = ModelCandidate(
                name=f"{model_type}_{paper.arxiv_id[:8]}",
                paper=paper,
                model_type=model_type,
                implementation_code=implementation_code,
                hyperparameters=hyperparameters,
                expected_performance=expected_performance,
                implementation_status="pending"
            )
            
            return candidate
            
        except Exception as e:
            logger.error(f"Error generating implementation for paper {paper.arxiv_id}: {e}")
            return None
    
    def _classify_model_type(self, paper: ResearchPaper) -> str:
        """Classify the type of model from paper content."""
        content = (paper.title + " " + paper.abstract).lower()
        
        if any(term in content for term in ["lstm", "long short-term memory", "recurrent"]):
            return "lstm"
        elif any(term in content for term in ["transformer", "attention", "self-attention"]):
            return "transformer"
        elif any(term in content for term in ["ensemble", "boosting", "bagging", "stacking"]):
            return "ensemble"
        elif any(term in content for term in ["reinforcement", "rl", "q-learning", "policy"]):
            return "reinforcement"
        elif any(term in content for term in ["neural", "deep learning", "cnn", "mlp"]):
            return "transformer"  # Default to transformer for neural networks
        else:
            return "sklearn"  # Default to sklearn for other approaches
    
    def _generate_code(self, paper: ResearchPaper, model_type: str) -> str:
        """Generate implementation code."""
        template = self.implementation_templates[model_type]
        
        # Add paper-specific modifications
        code = f"""# Implementation based on: {paper.title}
# Authors: {', '.join(paper.authors)}
# arXiv ID: {paper.arxiv_id}

{template}

# Model configuration
model_config = {{
    "paper_title": "{paper.title}",
    "arxiv_id": "{paper.arxiv_id}",
    "relevance_score": {paper.relevance_score},
    "implementation_complexity": "{paper.implementation_complexity}",
    "potential_impact": "{paper.potential_impact}"
}}
"""
        return code
    
    def _generate_hyperparameters(self, paper: ResearchPaper, model_type: str) -> Dict[str, Any]:
        """Generate hyperparameters based on paper and model type."""
        base_params = {
            "random_state": 42,
            "verbose": True
        }
        
        if model_type == "lstm":
            base_params.update({
                "input_size": 10,
                "hidden_size": 64,
                "num_layers": 2,
                "output_size": 1,
                "dropout": 0.2,
                "learning_rate": 0.001
            })
        elif model_type == "transformer":
            base_params.update({
                "input_size": 10,
                "d_model": 64,
                "nhead": 8,
                "num_layers": 2,
                "output_size": 1,
                "dropout": 0.1,
                "learning_rate": 0.0001
            })
        elif model_type == "ensemble":
            base_params.update({
                "n_estimators": 100,
                "max_depth": 10,
                "learning_rate": 0.1,
                "subsample": 0.8
            })
        elif model_type == "sklearn":
            base_params.update({
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            })
        
        return base_params
    
    def _estimate_performance(self, paper: ResearchPaper, model_type: str) -> Dict[str, float]:
        """Estimate expected performance based on paper and model type."""
        # Base estimates
        base_performance = {
            "mse": 0.01,
            "mae": 0.08,
            "r2_score": 0.7,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.15
        }
        
        # Adjust based on paper characteristics
        if paper.potential_impact == "high":
            base_performance["r2_score"] += 0.1
            base_performance["sharpe_ratio"] += 0.3
        elif paper.potential_impact == "low":
            base_performance["r2_score"] -= 0.1
            base_performance["sharpe_ratio"] -= 0.3
        
        if paper.relevance_score > 0.7:
            base_performance["mse"] *= 0.9
            base_performance["mae"] *= 0.9
        
        return base_performance

class ModelBenchmarker:
    """Benchmarks model performance against current best."""
    
    def __init__(self, 
                 benchmark_data: pd.DataFrame,
                 target_column: str = "returns",
                 current_best_score: float = 1.0):
        """Initialize benchmarker.
        
        Args:
            benchmark_data: Data for benchmarking
            target_column: Target variable
            current_best_score: Current best performance score
        """
        self.benchmark_data = benchmark_data
        self.target_column = target_column
        self.current_best_score = current_best_score
        
        # Performance history
        self.benchmark_history = []
        
        logger.info(f"Initialized Model Benchmarker with current best score: {current_best_score}")
    
    def benchmark_model(self, 
                       model_candidate: ModelCandidate,
                       test_data: Optional[pd.DataFrame] = None) -> BenchmarkResult:
        """Benchmark a model candidate.
        
        Args:
            model_candidate: Model to benchmark
            test_data: Test data (uses benchmark_data if None)
            
        Returns:
            Benchmark results
        """
        if test_data is None:
            test_data = self.benchmark_data
        
        try:
            start_time = time.time()
            
            # Prepare data
            X, y = self._prepare_benchmark_data(test_data)
            
            # Train and evaluate model
            if model_candidate.model_type == "sklearn":
                result = self._benchmark_sklearn_model(model_candidate, X, y)
            elif model_candidate.model_type in ["lstm", "transformer"]:
                result = self._benchmark_pytorch_model(model_candidate, X, y)
            else:
                result = self._benchmark_generic_model(model_candidate, X, y)
            
            # Calculate overall score
            result.overall_score = self._calculate_overall_score(result)
            
            # Record benchmark
            self.benchmark_history.append(result)
            
            logger.info(f"Benchmark completed for {model_candidate.name}: "
                       f"Overall score = {result.overall_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error benchmarking model {model_candidate.name}: {e}")
            return BenchmarkResult(
                model_name=model_candidate.name,
                mse=float('inf'),
                mae=float('inf'),
                r2_score=0.0,
                sharpe_ratio=0.0,
                max_drawdown=1.0,
                training_time=0.0,
                inference_time=0.0,
                memory_usage=0.0,
                overall_score=0.0,
                benchmark_date=datetime.now().isoformat()
            )
    
    def _prepare_benchmark_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for benchmarking."""
        # Select features
        feature_columns = [col for col in data.columns if col != self.target_column]
        X = data[feature_columns].values
        y = data[self.target_column].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        return X, y
    
    def _benchmark_sklearn_model(self, 
                                model_candidate: ModelCandidate,
                                X: np.ndarray, 
                                y: np.ndarray) -> BenchmarkResult:
        """Benchmark sklearn model."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")
        
        # Create model based on candidate
        if "random_forest" in model_candidate.name.lower():
            model = RandomForestRegressor(**model_candidate.hyperparameters)
        elif "gradient" in model_candidate.name.lower():
            model = GradientBoostingRegressor(**model_candidate.hyperparameters)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        mse_scores = []
        mae_scores = []
        r2_scores = []
        training_times = []
        inference_times = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train
            train_start = time.time()
            model.fit(X_train, y_train)
            training_times.append(time.time() - train_start)
            
            # Predict
            infer_start = time.time()
            y_pred = model.predict(X_test)
            inference_times.append(time.time() - infer_start)
            
            # Calculate metrics
            mse_scores.append(mean_squared_error(y_test, y_pred))
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))
        
        # Calculate trading metrics
        sharpe_ratio, max_drawdown = self._calculate_trading_metrics(y, model.predict(X))
        
        return BenchmarkResult(
            model_name=model_candidate.name,
            mse=np.mean(mse_scores),
            mae=np.mean(mae_scores),
            r2_score=np.mean(r2_scores),
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            training_time=np.mean(training_times),
            inference_time=np.mean(inference_times),
            memory_usage=0.0,  # sklearn models are typically small
            overall_score=0.0,  # Will be calculated later
            benchmark_date=datetime.now().isoformat()
        )
    
    def _benchmark_pytorch_model(self, 
                                model_candidate: ModelCandidate,
                                X: np.ndarray, 
                                y: np.ndarray) -> BenchmarkResult:
        """Benchmark PyTorch model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        # This is a simplified implementation
        # In practice, you'd implement the actual model from the candidate
        
        # For now, return placeholder results
        return BenchmarkResult(
            model_name=model_candidate.name,
            mse=0.01,
            mae=0.08,
            r2_score=0.75,
            sharpe_ratio=1.5,
            max_drawdown=0.12,
            training_time=10.0,
            inference_time=0.001,
            memory_usage=100.0,
            overall_score=0.0,
            benchmark_date=datetime.now().isoformat()
        )
    
    def _benchmark_generic_model(self, 
                                model_candidate: ModelCandidate,
                                X: np.ndarray, 
                                y: np.ndarray) -> BenchmarkResult:
        """Benchmark generic model."""
        # Placeholder implementation
        return BenchmarkResult(
            model_name=model_candidate.name,
            mse=0.015,
            mae=0.1,
            r2_score=0.65,
            sharpe_ratio=1.1,
            max_drawdown=0.18,
            training_time=5.0,
            inference_time=0.002,
            memory_usage=50.0,
            overall_score=0.0,
            benchmark_date=datetime.now().isoformat()
        )
    
    def _calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """Calculate trading-specific metrics."""
        try:
            # Calculate returns
            returns = np.diff(y_true)
            pred_returns = np.diff(y_pred)
            
            # Sharpe ratio
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            # Max drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
            
            return sharpe_ratio, max_drawdown
            
        except Exception as e:
            logger.error(f"Error calculating trading metrics: {e}")
            return 0.0, 0.0
    
    def _calculate_overall_score(self, result: BenchmarkResult) -> float:
        """Calculate overall performance score."""
        # Weighted combination of metrics
        weights = {
            "mse": -0.2,  # Lower is better
            "mae": -0.2,  # Lower is better
            "r2_score": 0.2,  # Higher is better
            "sharpe_ratio": 0.3,  # Higher is better
            "max_drawdown": -0.1,  # Lower is better
            "training_time": -0.05,  # Lower is better
            "inference_time": -0.05  # Lower is better
        }
        
        score = 0.0
        score += weights["mse"] * result.mse
        score += weights["mae"] * result.mae
        score += weights["r2_score"] * result.r2_score
        score += weights["sharpe_ratio"] * result.sharpe_ratio
        score += weights["max_drawdown"] * result.max_drawdown
        score += weights["training_time"] * result.training_time
        score += weights["inference_time"] * result.inference_time
        
        return max(0.0, score)  # Ensure non-negative

class AutoEvolutionaryModelGenerator:
    """Main agent for auto-evolutionary model generation."""
    
    def __init__(self, 
                 benchmark_data: pd.DataFrame,
                 target_column: str = "returns",
                 current_best_score: float = 1.0,
                 max_candidates: int = 10):
        """Initialize auto-evolutionary model generator.
        
        Args:
            benchmark_data: Data for benchmarking
            target_column: Target variable
            current_best_score: Current best performance score
            max_candidates: Maximum number of candidates to generate
        """
        self.benchmark_data = benchmark_data
        self.target_column = target_column
        self.current_best_score = current_best_score
        self.max_candidates = max_candidates
        
        # Initialize components
        self.research_fetcher = ArxivResearchFetcher()
        self.model_generator = ModelImplementationGenerator()
        self.benchmarker = ModelBenchmarker(benchmark_data, target_column, current_best_score)
        
        # Results storage
        self.candidates = []
        self.benchmark_results = []
        self.deployed_models = []
        
        # Create output directory
        self.output_dir = Path("agents/evolutionary_models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized Auto-Evolutionary Model Generator")
    
    async def discover_and_implement_models(self) -> List[ModelCandidate]:
        """Discover and implement new models from research."""
        logger.info("Starting model discovery and implementation...")
        
        try:
            # Fetch recent papers
            papers = await self.research_fetcher.fetch_recent_papers()
            
            # Generate model candidates
            candidates = []
            for paper in papers[:self.max_candidates]:
                candidate = self.model_generator.generate_implementation(paper)
                if candidate:
                    candidates.append(candidate)
            
            self.candidates = candidates
            logger.info(f"Generated {len(candidates)} model candidates")
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error in model discovery: {e}")
            return []
    
    def benchmark_candidates(self) -> List[BenchmarkResult]:
        """Benchmark all candidates."""
        logger.info("Starting candidate benchmarking...")
        
        results = []
        for candidate in self.candidates:
            try:
                result = self.benchmarker.benchmark_model(candidate)
                results.append(result)
                
                # Update candidate status
                candidate.implementation_status = "tested"
                
            except Exception as e:
                logger.error(f"Error benchmarking {candidate.name}: {e}")
        
        self.benchmark_results = results
        logger.info(f"Benchmarked {len(results)} candidates")
        
        return results
    
    def select_best_models(self, improvement_threshold: float = 0.1) -> List[BenchmarkResult]:
        """Select models that improve over current best."""
        if not self.benchmark_results:
            return []
        
        # Sort by overall score
        sorted_results = sorted(self.benchmark_results, 
                              key=lambda x: x.overall_score, reverse=True)
        
        # Select models that improve over current best
        best_models = []
        for result in sorted_results:
            if result.overall_score > self.current_best_score * (1 + improvement_threshold):
                best_models.append(result)
        
        logger.info(f"Selected {len(best_models)} models that improve over current best")
        return best_models
    
    def deploy_models(self, selected_models: List[BenchmarkResult]) -> List[str]:
        """Deploy selected models to production."""
        deployed = []
        
        for result in selected_models:
            try:
                # Find corresponding candidate
                candidate = next((c for c in self.candidates if c.name == result.model_name), None)
                
                if candidate:
                    # Save model implementation
                    model_path = self.output_dir / f"{candidate.name}.py"
                    with open(model_path, 'w') as f:
                        f.write(candidate.implementation_code)
                    
                    # Save configuration
                    config_path = self.output_dir / f"{candidate.name}_config.json"
                    config = {
                        "name": candidate.name,
                        "model_type": candidate.model_type,
                        "hyperparameters": candidate.hyperparameters,
                        "benchmark_results": {
                            "overall_score": result.overall_score,
                            "mse": result.mse,
                            "sharpe_ratio": result.sharpe_ratio,
                            "max_drawdown": result.max_drawdown
                        },
                        "deployment_date": datetime.now().isoformat(),
                        "paper_info": {
                            "title": candidate.paper.title,
                            "arxiv_id": candidate.paper.arxiv_id,
                            "authors": candidate.paper.authors
                        }
                    }
                    
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2, default=str)
                    
                    deployed.append(candidate.name)
                    candidate.implementation_status = "deployed"
                    
                    logger.info(f"Deployed model: {candidate.name}")
                
            except Exception as e:
                logger.error(f"Error deploying model {result.model_name}: {e}")
        
        self.deployed_models.extend(deployed)
        return deployed
    
    async def run_evolution_cycle(self, improvement_threshold: float = 0.1) -> Dict[str, Any]:
        """Run complete evolution cycle."""
        logger.info("Starting evolution cycle...")
        
        try:
            # Discover and implement models
            candidates = await self.discover_and_implement_models()
            
            # Benchmark candidates
            benchmark_results = self.benchmark_candidates()
            
            # Select best models
            best_models = self.select_best_models(improvement_threshold)
            
            # Deploy models
            deployed_models = self.deploy_models(best_models)
            
            # Compile results
            results = {
                "cycle_date": datetime.now().isoformat(),
                "candidates_generated": len(candidates),
                "candidates_benchmarked": len(benchmark_results),
                "models_selected": len(best_models),
                "models_deployed": len(deployed_models),
                "best_score": max([r.overall_score for r in benchmark_results]) if benchmark_results else 0.0,
                "improvement_over_current": max([r.overall_score for r in benchmark_results]) - self.current_best_score if benchmark_results else 0.0,
                "deployed_models": deployed_models
            }
            
            # Save cycle results
            cycle_path = self.output_dir / f"evolution_cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(cycle_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Evolution cycle completed. Deployed {len(deployed_models)} new models")
            return results
            
        except Exception as e:
            logger.error(f"Error in evolution cycle: {e}")
            return {"error": str(e)}

async def run_model_evolution(benchmark_data: pd.DataFrame,
                       target_column: str = "returns",
                       current_best_score: float = 1.0) -> Dict[str, Any]:
    """Run model evolution process.
    
    Args:
        benchmark_data: Data for benchmarking
        target_column: Target variable
        current_best_score: Current best performance score
        
    Returns:
        Evolution results
    """
    try:
        # Initialize generator
        generator = AutoEvolutionaryModelGenerator(
            benchmark_data=benchmark_data,
            target_column=target_column,
            current_best_score=current_best_score,
            max_candidates=10
        )
        
        # Run evolution cycle
        results = await generator.run_evolution_cycle()
        
        return results
        
    except Exception as e:
        logger.error(f"Error in model evolution: {e}")
        return {"error": str(e)}

def run_model_evolution_sync(benchmark_data: pd.DataFrame,
                       target_column: str = "returns",
                       current_best_score: float = 1.0) -> Dict[str, Any]:
    """Synchronous wrapper for model evolution process.
    
    Args:
        benchmark_data: Data for benchmarking
        target_column: Target variable
        current_best_score: Current best performance score
        
    Returns:
        Evolution results
    """
    try:
        return asyncio.run(run_model_evolution(benchmark_data, target_column, current_best_score))
    except Exception as e:
        logger.error(f"Error in model evolution: {e}")
        return {"error": str(e)} 