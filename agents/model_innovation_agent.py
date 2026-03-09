"""
ModelInnovationAgent — lightweight stub for automated model search.

This stub exists so orchestrators and tests can import a working
ModelInnovationAgent without failing even when full AutoML
infrastructure is not yet configured. It exposes a minimal
interface and uses flaml/optuna only when available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import flaml  # type: ignore
    import optuna  # type: ignore

    _AUTO_ML_AVAILABLE = True
except Exception:
    _AUTO_ML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ModelInnovationResult:
    success: bool
    message: str
    best_params: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    search_space: Optional[Dict[str, Any]] = None


class ModelInnovationAgent:
    """
    Minimal stand-in for the full ModelInnovationAgent.

    Responsibilities in this stub:
    - Accept a simple search request (model type, metric).
    - If flaml/optuna are installed, acknowledge availability but
      do not run long searches inside the Streamlit request cycle.
    - Return a structured result that callers can render safely.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.auto_ml_available = _AUTO_ML_AVAILABLE
        logger.info(
            "ModelInnovationAgent initialized (AutoML available=%s)",
            self.auto_ml_available,
        )

    def suggest_hyperparameters(
        self,
        model_type: str = "xgboost",
        metric: str = "mape",
        search_space: Optional[Dict[str, Any]] = None,
    ) -> ModelInnovationResult:
        """
        Return a placeholder hyperparameter suggestion.

        In a future version this can call flaml/optuna to run an
        actual search. For now we keep it deterministic and fast.
        """
        if not self.auto_ml_available:
            msg = (
                "AutoML libraries (flaml/optuna) are not available. "
                "Install them in the environment to enable full model innovation."
            )
            logger.warning(msg)
            return ModelInnovationResult(
                success=False,
                message=msg,
                best_params=None,
                best_score=None,
                search_space=search_space,
            )

        # Very light-weight, no real search to avoid long blocking calls.
        default_space = search_space or {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 4, 5],
            "n_estimators": [50, 100, 200],
        }
        # Pick mid-point values as a sane default "suggestion"
        best_params = {
            "learning_rate": 0.05,
            "max_depth": 4,
            "n_estimators": 100,
            "model_type": model_type,
            "metric": metric,
        }
        msg = (
            "Returned a default hyperparameter suggestion. "
            "Full AutoML search is intentionally disabled in the UI path."
        )
        return ModelInnovationResult(
            success=True,
            message=msg,
            best_params=best_params,
            best_score=None,
            search_space=default_space,
        )


def create_model_innovation_agent(config: Optional[Dict[str, Any]] = None) -> ModelInnovationAgent:
    """Factory used by orchestrator/task_providers."""
    return ModelInnovationAgent(config=config or {})

