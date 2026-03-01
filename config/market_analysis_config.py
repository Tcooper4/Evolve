"""
Market analysis configuration.

Loads config from config/market_analysis_config.yaml and exposes it via
MarketAnalysisConfig for use by config/__init__.py and any code that imports it.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parent / "market_analysis_config.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Could not load market_analysis_config.yaml: %s", e)
        return {}


class MarketAnalysisConfig:
    """
    Wrapper for market analysis settings (market conditions, technical indicators,
    visualization, pipeline). Loads from config/market_analysis_config.yaml.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._path = config_path or _DEFAULT_PATH
        self._data = _load_yaml(self._path)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a top-level key (e.g. 'market_conditions', 'analysis_settings')."""
        return self._data.get(key, default)

    @property
    def market_conditions(self) -> Dict[str, Any]:
        return self._data.get("market_conditions") or {}

    @property
    def analysis_settings(self) -> Dict[str, Any]:
        return self._data.get("analysis_settings") or {}

    @property
    def visualization_settings(self) -> Dict[str, Any]:
        return self._data.get("visualization_settings") or {}

    @property
    def pipeline_settings(self) -> Dict[str, Any]:
        return self._data.get("pipeline_settings") or {}

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)
