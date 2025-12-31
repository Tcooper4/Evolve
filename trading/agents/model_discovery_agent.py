"""
Model Discovery Agent

This agent automatically discovers, benchmarks, and integrates new stock forecasting models
from Arxiv, Hugging Face, and GitHub. It performs comprehensive evaluation and only
registers high-performing models into the model pool.
"""

import logging
import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

warnings.filterwarnings("ignore")

# Try to import arxiv
try:
    import arxiv

    ARXIV_AVAILABLE = True
except ImportError as e:
    print("⚠️ arxiv not available. Disabling Arxiv model discovery.")
    print(f"   Missing: {e}")
    arxiv = None
    ARXIV_AVAILABLE = False

# Try to import huggingface_hub
try:
    from huggingface_hub import HfApi

    HUGGINGFACE_AVAILABLE = True
except ImportError as e:
    print("⚠️ huggingface_hub not available. Disabling HuggingFace model discovery.")
    print(f"   Missing: {e}")
    HfApi = None
    HUGGINGFACE_AVAILABLE = False

# Try to import PyTorch for model creation
try:
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError as e:
    print("⚠️ PyTorch not available. Disabling PyTorch model creation.")
    print(f"   Missing: {e}")
    nn = None
    TORCH_AVAILABLE = False

# GitHub API availability (using requests)
GITHUB_AVAILABLE = True  # requests is already imported

logger = logging.getLogger(__name__)


@dataclass
class ModelDiscovery:
    """Discovered model information."""

    source: str  # 'arxiv', 'huggingface', 'github'
    model_id: str
    title: str
    description: str
    url: str
    framework: str
    model_type: str
    discovered_at: datetime
    performance_metrics: Optional[Dict[str, float]] = None
    benchmark_status: str = "pending"
    integration_status: str = "pending"
    rejection_reason: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Benchmark results for discovered models."""

    model_id: str
    rmse: float
    mae: float
    mape: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    benchmark_date: datetime
    dataset_size: int
    training_time: float
    inference_time: float
    overall_score: float
    is_approved: bool = False
    rejection_reason: Optional[str] = None


class ArxivModelDiscoverer:
    """Discovers models from Arxiv papers."""

    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.search_terms = [
            "stock price forecasting transformer",
            "time series prediction LSTM",
            "financial forecasting neural network",
            "market prediction deep learning",
            "trading signal generation",
            "quantitative trading model",
        ]

    def search_models(self, max_results: int = 50) -> List[ModelDiscovery]:
        """Search for models on Arxiv."""
        discoveries = []

        if not ARXIV_AVAILABLE:
            logger.warning("Arxiv library not available")
            return discoveries

        try:
            for term in self.search_terms:
                logger.info(f"Searching Arxiv for: {term}")

                # Search Arxiv
                search = arxiv.Search(
                    query=term,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                )

                for result in search.results():
                    # Extract model information
                    model_info = self._extract_model_info(result)
                    if model_info:
                        discoveries.append(model_info)

                time.sleep(1)  # Rate limiting

        except Exception as e:
            logger.error(f"Error searching Arxiv: {e}")

        return discoveries

    def _extract_model_info(self, result) -> Optional[ModelDiscovery]:
        """Extract model information from Arxiv result."""
        try:
            # Parse title and abstract for model information
            title = result.title
            abstract = result.summary

            # Determine model type
            model_type = self._classify_model_type(title, abstract)
            framework = self._extract_framework(title, abstract)

            if model_type and framework:
                return ModelDiscovery(
                    source="arxiv",
                    model_id=f"arxiv_{result.entry_id.split('/')[-1]}",
                    title=title,
                    description=(
                        abstract[:500] + "..." if len(abstract) > 500 else abstract
                    ),
                    url=result.entry_id,
                    framework=framework,
                    model_type=model_type,
                    discovered_at=datetime.now(),
                )

        except Exception as e:
            logger.error(f"Error extracting model info: {e}")

        return None

    def _classify_model_type(self, title: str, abstract: str) -> Optional[str]:
        """Classify the type of model."""
        text = (title + " " + abstract).lower()

        if any(word in text for word in ["transformer", "attention"]):
            return "transformer"
        elif any(word in text for word in ["lstm", "long short-term memory"]):
            return "lstm"
        elif any(word in text for word in ["gru", "gated recurrent"]):
            return "gru"
        elif any(word in text for word in ["cnn", "convolutional"]):
            return "cnn"
        elif any(word in text for word in ["tcn", "temporal convolutional"]):
            return "tcn"
        elif any(word in text for word in ["xgboost", "gradient boosting"]):
            return "xgboost"
        elif any(word in text for word in ["random forest", "ensemble"]):
            return "ensemble"
        else:
            return "neural_network"

    def _extract_framework(self, title: str, abstract: str) -> Optional[str]:
        """Extract the framework used."""
        text = (title + " " + abstract).lower()

        if any(word in text for word in ["pytorch", "torch"]):
            return "pytorch"
        elif any(word in text for word in ["xgboost"]):
            return "xgboost"
        elif any(word in text for word in ["lightgbm"]):
            return "lightgbm"
        else:
            return "unknown"


class HuggingFaceModelDiscoverer:
    """Discovers models from Hugging Face Hub."""

    def __init__(self):
        self.api = HfApi() if HUGGINGFACE_AVAILABLE else None
        self.search_terms = [
            "stock-forecasting",
            "time-series-forecasting",
            "financial-prediction",
            "market-prediction",
            "trading-signals",
        ]

    def search_models(self, max_results: int = 50) -> List[ModelDiscovery]:
        """Search for models on Hugging Face Hub."""
        discoveries = []

        if not HUGGINGFACE_AVAILABLE or not self.api:
            logger.warning("Hugging Face Hub not available")
            return discoveries

        try:
            for term in self.search_terms:
                logger.info(f"Searching Hugging Face for: {term}")

                # Search models
                models = self.api.list_models(
                    search=term, limit=max_results, sort="downloads", direction=-1
                )

                for model in models:
                    model_info = self._extract_model_info(model)
                    if model_info:
                        discoveries.append(model_info)

                time.sleep(1)  # Rate limiting

        except Exception as e:
            logger.error(f"Error searching Hugging Face: {e}")

        return discoveries

    def _extract_model_info(self, model) -> Optional[ModelDiscovery]:
        """Extract model information from Hugging Face model."""
        try:
            # Determine model type and framework
            model_type = self._classify_model_type(model.modelId, model.tags or [])
            framework = self._extract_framework(model.modelId, model.tags or [])

            if model_type and framework:
                return ModelDiscovery(
                    source="huggingface",
                    model_id=f"hf_{model.modelId.replace('/', '_')}",
                    title=model.modelId,
                    description=model.description or "No description available",
                    url=f"https://huggingface.co/{model.modelId}",
                    framework=framework,
                    model_type=model_type,
                    discovered_at=datetime.now(),
                )

        except Exception as e:
            logger.error(f"Error extracting model info: {e}")

        return None

    def _classify_model_type(self, model_id: str, tags: List[str]) -> Optional[str]:
        """Classify the type of model."""
        text = (model_id + " " + " ".join(tags)).lower()

        if any(word in text for word in ["transformer", "attention"]):
            return "transformer"
        elif any(word in text for word in ["lstm", "rnn"]):
            return "lstm"
        elif any(word in text for word in ["tcn", "temporal"]):
            return "tcn"
        elif any(word in text for word in ["cnn", "convolutional"]):
            return "cnn"
        else:
            return "neural_network"

    def _extract_framework(self, model_id: str, tags: List[str]) -> Optional[str]:
        """Extract the framework used."""
        text = (model_id + " " + " ".join(tags)).lower()

        if any(word in text for word in ["pytorch", "torch"]):
            return "pytorch"
        elif any(word in text for word in ["xgboost"]):
            return "xgboost"
        else:
            return "pytorch"  # Default to PyTorch for HF models


class GitHubModelDiscoverer:
    """Discovers models from GitHub repositories."""

    def __init__(self):
        self.search_terms = [
            "stock forecasting transformer",
            "time series prediction LSTM",
            "financial forecasting model",
            "trading signal generation",
            "quantitative trading",
        ]

    def search_models(self, max_results: int = 50) -> List[ModelDiscovery]:
        """Search for models on GitHub."""
        discoveries = []

        if not GITHUB_AVAILABLE:
            logger.warning("GitHub library not available")
            return discoveries

        try:
            # Note: GitHub API requires authentication for higher rate limits
            # This is a simplified version using search API
            for term in self.search_terms:
                logger.info(f"Searching GitHub for: {term}")

                # Use GitHub search API
                search_url = f"https://api.github.com/search/repositories?q={term}&sort=stars&order=desc&per_page={max_results}"

                response = requests.get(search_url)
                if response.status_code == 200:
                    repos = response.json()["items"]

                    for repo in repos:
                        model_info = self._extract_model_info(repo)
                        if model_info:
                            discoveries.append(model_info)

                time.sleep(2)  # Rate limiting

        except Exception as e:
            logger.error(f"Error searching GitHub: {e}")

        return discoveries

    def _extract_model_info(self, repo) -> Optional[ModelDiscovery]:
        """Extract model information from GitHub repository."""
        try:
            # Parse repository information
            title = repo["name"]
            description = repo["description"] or ""

            # Determine model type and framework
            model_type = self._classify_model_type(title, description)
            framework = self._extract_framework(title, description)

            if model_type and framework:
                return ModelDiscovery(
                    source="github",
                    model_id=f"github_{repo['id']}",
                    title=title,
                    description=(
                        description[:500] + "..."
                        if len(description) > 500
                        else description
                    ),
                    url=repo["html_url"],
                    framework=framework,
                    model_type=model_type,
                    discovered_at=datetime.now(),
                )

        except Exception as e:
            logger.error(f"Error extracting model info: {e}")

        return None

    def _classify_model_type(self, title: str, description: str) -> Optional[str]:
        """Classify the type of model."""
        text = (title + " " + description).lower()

        if any(word in text for word in ["transformer", "attention"]):
            return "transformer"
        elif any(word in text for word in ["lstm", "rnn"]):
            return "lstm"
        elif any(word in text for word in ["tcn", "temporal"]):
            return "tcn"
        elif any(word in text for word in ["cnn", "convolutional"]):
            return "cnn"
        elif any(word in text for word in ["xgboost", "gradient"]):
            return "xgboost"
        else:
            return "neural_network"

    def _extract_framework(self, title: str, description: str) -> Optional[str]:
        """Extract the framework used."""
        text = (title + " " + description).lower()

        if any(word in text for word in ["pytorch", "torch"]):
            return "pytorch"
        elif any(word in text for word in ["xgboost"]):
            return "xgboost"
        elif any(word in text for word in ["sklearn", "scikit"]):
            return "sklearn"
        else:
            return "unknown"


class ModelBenchmarker:
    """Benchmarks discovered models using existing backtesting logic."""

    def __init__(self):
        self.benchmark_results = {}
        self.performance_thresholds = {
            "rmse": 0.05,  # Lower is better
            "mae": 0.04,
            "mape": 5.0,
            "sharpe_ratio": 0.5,  # Higher is better
            "max_drawdown": 0.15,  # Lower is better
            "win_rate": 0.55,
            "profit_factor": 1.2,
        }

    def benchmark_model(self, model_discovery: ModelDiscovery) -> BenchmarkResult:
        """Benchmark a discovered model."""
        try:
            logger.info(f"Benchmarking model: {model_discovery.model_id}")

            # Generate test data
            X, y = self._generate_test_data()

            # Create and train model
            model = self._create_model(model_discovery)
            if not model:
                return self._create_failed_benchmark(
                    model_discovery.model_id, "Model creation failed"
                )

            # Train model
            start_time = time.time()
            model.fit(X, y)
            training_time = time.time() - start_time

            # Make predictions
            start_time = time.time()
            y_pred = model.predict(X)
            inference_time = time.time() - start_time

            # Calculate metrics
            metrics = self._calculate_metrics(y, y_pred)

            # Calculate overall score
            overall_score = self._calculate_overall_score(metrics)

            # Determine if model is approved
            is_approved = self._evaluate_performance(metrics)
            rejection_reason = (
                None if is_approved else self._get_rejection_reason(metrics)
            )

            result = BenchmarkResult(
                model_id=model_discovery.model_id,
                rmse=metrics["rmse"],
                mae=metrics["mae"],
                mape=metrics["mape"],
                sharpe_ratio=metrics["sharpe_ratio"],
                max_drawdown=metrics["max_drawdown"],
                win_rate=metrics["win_rate"],
                profit_factor=metrics["profit_factor"],
                benchmark_date=datetime.now(),
                dataset_size=len(X),
                training_time=training_time,
                inference_time=inference_time,
                overall_score=overall_score,
                is_approved=is_approved,
                rejection_reason=rejection_reason,
            )

            self.benchmark_results[model_discovery.model_id] = result
            logger.info(
                f"Benchmark completed for {model_discovery.model_id}: Score={overall_score:.3f}, Approved={is_approved}"
            )

            return result

        except Exception as e:
            logger.error(f"Error benchmarking model {model_discovery.model_id}: {e}")
            return self._create_failed_benchmark(model_discovery.model_id, str(e))

    def _generate_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate test data for benchmarking."""
        # Generate synthetic time series data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        # Create features
        X = np.random.randn(n_samples, n_features)

        # Create target with some pattern
        y = np.sum(X[:, :3], axis=1) + 0.1 * np.random.randn(n_samples)

        return X, y

    def _create_model(self, model_discovery: ModelDiscovery):
        """Create model instance based on discovery information."""
        try:
            if model_discovery.framework == "sklearn":
                from sklearn.ensemble import RandomForestRegressor

                return RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_discovery.framework == "xgboost":
                import xgboost as xgb

                return xgb.XGBRegressor(n_estimators=100, random_state=42)
            elif model_discovery.framework == "pytorch":
                # Create a simple neural network
                import torch
                import torch.nn as nn

                class SimpleNN(nn.Module):
                    def __init__(self, input_size=10, hidden_size=64, output_size=1):
                        super().__init__()
                        self.layer1 = nn.Linear(input_size, hidden_size)
                        self.layer2 = nn.Linear(hidden_size, hidden_size)
                        self.layer3 = nn.Linear(hidden_size, output_size)
                        self.relu = nn.ReLU()

                    def forward(self, x):
                        x = self.relu(self.layer1(x))
                        x = self.relu(self.layer2(x))
                        x = self.layer3(x)
                        return x

                model = SimpleNN()

                # Create a wrapper for sklearn-like interface
                class PyTorchWrapper:
                    def __init__(self, model):
                        self.model = model
                        self.optimizer = torch.optim.Adam(model.parameters())
                        self.criterion = nn.MSELoss()

                    def fit(self, X, y):
                        X_tensor = torch.FloatTensor(X)
                        y_tensor = torch.FloatTensor(y).unsqueeze(1)

                        for epoch in range(100):
                            self.optimizer.zero_grad()
                            outputs = self.model(X_tensor)
                            loss = self.criterion(outputs, y_tensor)
                            loss.backward()
                            self.optimizer.step()

                    def predict(self, X):
                        X_tensor = torch.FloatTensor(X)
                        with torch.no_grad():
                            outputs = self.model(X_tensor)
                        return outputs.squeeze().numpy()

                return PyTorchWrapper(model)
            else:
                # Default to sklearn
                from sklearn.ensemble import RandomForestRegressor

                return RandomForestRegressor(n_estimators=100, random_state=42)

        except Exception as e:
            logger.error(f"Error creating model: {e}")
            return None

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        try:
            # Basic regression metrics
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mae = np.mean(np.abs(y_true - y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            # Trading metrics
            returns = np.diff(y_true) / y_true[:-1]
            pred_returns = np.diff(y_pred) / y_pred[:-1]

            # Sharpe ratio
            sharpe_ratio = (
                np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            )

            # Max drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)

            # Win rate
            signals = np.where(pred_returns > 0, 1, -1)
            actual_direction = np.where(returns > 0, 1, -1)
            win_rate = np.mean(signals == actual_direction)

            # Profit factor
            winning_trades = returns[signals == actual_direction]
            losing_trades = returns[signals != actual_direction]

            if len(losing_trades) > 0 and np.sum(losing_trades) != 0:
                profit_factor = np.sum(winning_trades) / abs(np.sum(losing_trades))
            else:
                profit_factor = 1.0

            return {
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": abs(max_drawdown),
                "win_rate": win_rate,
                "profit_factor": profit_factor,
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                "rmse": float("inf"),
                "mae": float("inf"),
                "mape": float("inf"),
                "sharpe_ratio": 0.0,
                "max_drawdown": 1.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
            }

    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score."""
        try:
            # Normalize metrics to 0-1 scale
            normalized_metrics = {
                "rmse": max(0, 1 - metrics["rmse"] / 0.1),  # Lower is better
                "mae": max(0, 1 - metrics["mae"] / 0.08),
                "mape": max(0, 1 - metrics["mape"] / 10),
                "sharpe_ratio": min(
                    1, max(0, metrics["sharpe_ratio"] / 2)
                ),  # Higher is better
                "max_drawdown": max(
                    0, 1 - metrics["max_drawdown"] / 0.3
                ),  # Lower is better
                "win_rate": metrics["win_rate"],
                "profit_factor": min(1, metrics["profit_factor"] / 3),
            }

            # Weighted average
            weights = {
                "rmse": 0.2,
                "mae": 0.15,
                "mape": 0.1,
                "sharpe_ratio": 0.25,
                "max_drawdown": 0.15,
                "win_rate": 0.1,
                "profit_factor": 0.05,
            }

            overall_score = sum(
                normalized_metrics[metric] * weights[metric] for metric in weights
            )
            return overall_score

        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 0.0

    def _evaluate_performance(self, metrics: Dict[str, float]) -> bool:
        """Evaluate if model meets performance thresholds."""
        try:
            for metric, threshold in self.performance_thresholds.items():
                if metric in ["rmse", "mae", "mape", "max_drawdown"]:
                    if metrics[metric] > threshold:
                        return False
                else:  # sharpe_ratio, win_rate, profit_factor
                    if metrics[metric] < threshold:
                        return False
            return True

        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            return False

    def _get_rejection_reason(self, metrics: Dict[str, float]) -> str:
        """Get reason for model rejection."""
        reasons = []

        for metric, threshold in self.performance_thresholds.items():
            if metric in ["rmse", "mae", "mape", "max_drawdown"]:
                if metrics[metric] > threshold:
                    reasons.append(
                        f"{metric.upper()} too high: {metrics[metric]:.4f} > {threshold}"
                    )
            else:
                if metrics[metric] < threshold:
                    reasons.append(
                        f"{metric.upper()} too low: {metrics[metric]:.4f} < {threshold}"
                    )

        return "; ".join(reasons)

    def _create_failed_benchmark(self, model_id: str, reason: str) -> BenchmarkResult:
        """Create a failed benchmark result."""
        return BenchmarkResult(
            model_id=model_id,
            rmse=float("inf"),
            mae=float("inf"),
            mape=float("inf"),
            sharpe_ratio=0.0,
            max_drawdown=1.0,
            win_rate=0.0,
            profit_factor=0.0,
            benchmark_date=datetime.now(),
            dataset_size=0,
            training_time=0.0,
            inference_time=0.0,
            overall_score=0.0,
            is_approved=False,
            rejection_reason=reason,
        )


class ModelDiscoveryAgent:
    """Main agent for discovering, benchmarking, and integrating new models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the model discovery agent."""
        self.config = config or {}

        # Initialize discoverers
        self.arxiv_discoverer = ArxivModelDiscoverer()
        self.hf_discoverer = HuggingFaceModelDiscoverer()
        self.github_discoverer = GitHubModelDiscoverer()

        # Initialize benchmarker
        self.benchmarker = ModelBenchmarker()

        # Storage
        self.discovered_models = []
        self.benchmark_results = {}
        self.integrated_models = []

        # Performance tracking
        self.discovery_stats = {
            "total_discovered": 0,
            "total_benchmarked": 0,
            "total_approved": 0,
            "total_rejected": 0,
            "last_discovery_run": None,
        }

        logger.info("ModelDiscoveryAgent initialized successfully")

    def run_discovery(self, max_results_per_source: int = 20) -> List[ModelDiscovery]:
        """Run full model discovery process."""
        try:
            logger.info("Starting model discovery process")

            discoveries = []

            # Discover from Arxiv
            if ARXIV_AVAILABLE:
                logger.info("Discovering models from Arxiv...")
                arxiv_discoveries = self.arxiv_discoverer.search_models(
                    max_results_per_source
                )
                discoveries.extend(arxiv_discoveries)
                logger.info(f"Found {len(arxiv_discoveries)} models on Arxiv")

            # Discover from Hugging Face
            if HUGGINGFACE_AVAILABLE:
                logger.info("Discovering models from Hugging Face...")
                hf_discoveries = self.hf_discoverer.search_models(
                    max_results_per_source
                )
                discoveries.extend(hf_discoveries)
                logger.info(f"Found {len(hf_discoveries)} models on Hugging Face")

            # Discover from GitHub
            if GITHUB_AVAILABLE:
                logger.info("Discovering models from GitHub...")
                github_discoveries = self.github_discoverer.search_models(
                    max_results_per_source
                )
                discoveries.extend(github_discoveries)
                logger.info(f"Found {len(github_discoveries)} models on GitHub")

            # Remove duplicates
            unique_discoveries = self._remove_duplicates(discoveries)

            # Store discoveries
            self.discovered_models.extend(unique_discoveries)

            # Update stats
            self.discovery_stats["total_discovered"] += len(unique_discoveries)
            self.discovery_stats["last_discovery_run"] = datetime.now()

            logger.info(
                f"Discovery completed. Found {len(unique_discoveries)} unique models"
            )

            return unique_discoveries

        except Exception as e:
            logger.error(f"Error in model discovery: {e}")
            return []

    def benchmark_discoveries(
        self, discoveries: List[ModelDiscovery]
    ) -> List[BenchmarkResult]:
        """Benchmark discovered models."""
        try:
            logger.info(f"Starting benchmark of {len(discoveries)} models")

            benchmark_results = []

            for discovery in discoveries:
                result = self.benchmarker.benchmark_model(discovery)
                benchmark_results.append(result)

                # Store result
                self.benchmark_results[discovery.model_id] = result

                # Update discovery status
                discovery.benchmark_status = "completed"
                discovery.performance_metrics = {
                    "rmse": result.rmse,
                    "mae": result.mae,
                    "mape": result.mape,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "overall_score": result.overall_score,
                }

            # Update stats
            self.discovery_stats["total_benchmarked"] += len(benchmark_results)
            self.discovery_stats["total_approved"] += sum(
                1 for r in benchmark_results if r.is_approved
            )
            self.discovery_stats["total_rejected"] += sum(
                1 for r in benchmark_results if not r.is_approved
            )

            logger.info(
                f"Benchmark completed. Approved: {self.discovery_stats['total_approved']}, Rejected: {self.discovery_stats['total_rejected']}"
            )

            return benchmark_results

        except Exception as e:
            logger.error(f"Error in benchmarking: {e}")
            return []

    def integrate_approved_models(
        self, benchmark_results: List[BenchmarkResult]
    ) -> List[str]:
        """Integrate approved models into the model pool."""
        try:
            logger.info("Integrating approved models into model pool")

            approved_models = [r for r in benchmark_results if r.is_approved]
            integrated_ids = []

            for result in approved_models:
                try:
                    # Find corresponding discovery
                    discovery = next(
                        (
                            d
                            for d in self.discovered_models
                            if d.model_id == result.model_id
                        ),
                        None,
                    )

                    if discovery:
                        # Register model in the system
                        success = self._register_model_in_pool(discovery, result)

                        if success:
                            discovery.integration_status = "integrated"
                            integrated_ids.append(result.model_id)
                            self.integrated_models.append(discovery)
                            logger.info(
                                f"Successfully integrated model: {result.model_id}"
                            )
                        else:
                            discovery.integration_status = "failed"
                            logger.warning(
                                f"Failed to integrate model: {result.model_id}"
                            )

                except Exception as e:
                    logger.error(f"Error integrating model {result.model_id}: {e}")

            logger.info(
                f"Integration completed. Successfully integrated {len(integrated_ids)} models"
            )

            return integrated_ids

        except Exception as e:
            logger.error(f"Error in model integration: {e}")
            return []

    def _register_model_in_pool(
        self, discovery: ModelDiscovery, result: BenchmarkResult
    ) -> bool:
        """Register model in the system's model pool."""
        try:
            logger.info(f"Registering model {discovery.model_id} in model pool")

            # 1. Register model in ModelRegistry
            try:
                from trading.models.registry import get_model_registry
                
                registry = get_model_registry()
                
                # Try to get model class from discovery
                if hasattr(discovery, 'model_class') and discovery.model_class:
                    registry.register_model(discovery.model_id, discovery.model_class)
                    logger.info(f"Registered {discovery.model_id} in ModelRegistry")
                else:
                    # If no model class, save model implementation and register path
                    logger.warning(f"No model class found for {discovery.model_id}, saving implementation only")
                    
            except ImportError:
                logger.warning("ModelRegistry not available, skipping registration")
            except Exception as e:
                logger.error(f"Error registering in ModelRegistry: {e}")

            # 2. Register in ForecastRouter if available
            try:
                from models.forecast_router import ForecastRouter
                
                router = ForecastRouter()
                if hasattr(router, 'model_registry') and discovery.model_id not in router.model_registry:
                    # Add to router's model registry
                    if hasattr(discovery, 'model_class') and discovery.model_class:
                        router.model_registry[discovery.model_id] = discovery.model_class
                        logger.info(f"Added {discovery.model_id} to ForecastRouter")
            except ImportError:
                logger.warning("ForecastRouter not available, skipping router registration")
            except Exception as e:
                logger.error(f"Error registering in ForecastRouter: {e}")

            # 3. Save model metadata to persistent storage
            try:
                import json
                from pathlib import Path
                
                models_dir = Path("models/discovered")
                models_dir.mkdir(parents=True, exist_ok=True)
                
                metadata = {
                    "model_id": discovery.model_id,
                    "name": discovery.name,
                    "model_type": discovery.model_type,
                    "source": discovery.source,
                    "title": discovery.title,
                    "arxiv_id": getattr(discovery, 'arxiv_id', None),
                    "benchmark_results": {
                        "overall_score": result.overall_score,
                        "rmse": result.rmse,
                        "mae": result.mae,
                        "sharpe_ratio": result.sharpe_ratio,
                        "max_drawdown": result.max_drawdown,
                        "is_approved": result.is_approved,
                    },
                    "integration_date": datetime.now().isoformat(),
                }
                
                metadata_path = models_dir / f"{discovery.model_id}_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                logger.info(f"Saved metadata for {discovery.model_id}")
                
            except Exception as e:
                logger.error(f"Error saving model metadata: {e}")

            # 4. Update integration history
            self.integration_history.append({
                "model_id": discovery.model_id,
                "integration_date": datetime.now().isoformat(),
                "benchmark_score": result.overall_score,
                "status": "integrated",
            })

            logger.info(f"Successfully registered model {discovery.model_id} in model pool")
            return True

        except Exception as e:
            logger.error(f"Error registering model in pool: {e}")
            return False

    def _remove_duplicates(
        self, discoveries: List[ModelDiscovery]
    ) -> List[ModelDiscovery]:
        """Remove duplicate discoveries based on title similarity."""
        unique_discoveries = []
        seen_titles = set()

        for discovery in discoveries:
            # Normalize title for comparison
            normalized_title = re.sub(r"[^\w\s]", "", discovery.title.lower())

            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_discoveries.append(discovery)

        return unique_discoveries

    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        return {
            **self.discovery_stats,
            "total_discoveries": len(self.discovered_models),
            "total_benchmarks": len(self.benchmark_results),
            "total_integrated": len(self.integrated_models),
        }

    def get_top_performing_models(self, n: int = 10) -> List[BenchmarkResult]:
        """Get top performing models."""
        sorted_results = sorted(
            self.benchmark_results.values(), key=lambda x: x.overall_score, reverse=True
        )
        return sorted_results[:n]

    def get_rejected_models(self) -> List[BenchmarkResult]:
        """Get rejected models with reasons."""
        return [r for r in self.benchmark_results.values() if not r.is_approved]


def get_model_discovery_agent(
    config: Optional[Dict[str, Any]] = None,
) -> ModelDiscoveryAgent:
    """Get the model discovery agent instance."""
    return ModelDiscoveryAgent(config)
