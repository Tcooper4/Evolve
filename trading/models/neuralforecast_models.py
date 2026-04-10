"""
NeuralForecast model wrappers (N-BEATS, N-HiTS, PatchTST, TFT).

Autoformer / Informer are not registered in Evolve (see model_registry).
The neuralforecast package is imported lazily on first fit() so importing
this module does not load heavy DL stacks at interpreter startup.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Populated by _lazy_import_neuralforecast()
NeuralForecast: Any = None
NBEATS: Any = None
NHITS: Any = None
PatchTST: Any = None
TFT: Any = None
_NF_IMPORT_TRIED: bool = False
_NF_IMPORT_OK: bool = False


def neuralforecast_installed() -> bool:
    """Cheap check — does not import neuralforecast."""
    try:
        return importlib.util.find_spec("neuralforecast") is not None
    except Exception:
        return False


def _lazy_import_neuralforecast() -> bool:
    """Import neuralforecast once; return True if usable."""
    global NeuralForecast, NBEATS, NHITS, PatchTST, TFT
    global _NF_IMPORT_TRIED, _NF_IMPORT_OK
    if _NF_IMPORT_TRIED:
        return _NF_IMPORT_OK
    _NF_IMPORT_TRIED = True
    try:
        from neuralforecast import NeuralForecast as _NF
        from neuralforecast.models import NBEATS as _NB
        from neuralforecast.models import NHITS as _NH
        from neuralforecast.models import PatchTST as _PT
        from neuralforecast.models import TFT as _TFT

        NeuralForecast = _NF
        NBEATS = _NB
        NHITS = _NH
        PatchTST = _PT
        TFT = _TFT
        _NF_IMPORT_OK = True
        logger.info("[OK] NeuralForecast loaded (lazy)")
    except ImportError as e:
        _NF_IMPORT_OK = False
        logger.warning("NeuralForecast not available: %s", e)
    except Exception as e:
        _NF_IMPORT_OK = False
        logger.warning("NeuralForecast import failed: %s", e)
    return _NF_IMPORT_OK


# Back-compat: truthy if package appears installed (no import).
NEURALFORECAST_AVAILABLE: bool = neuralforecast_installed()


def _strip_router_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """Remove keys passed by ForecastRouter that are not model kwargs."""
    skip = {
        "target_column",
        "date_column",
        "feature_columns",
        "sequence_length",
        "epochs",
        "models",
        "voting_method",
        "weight_window",
    }
    return {k: v for k, v in d.items() if k not in skip}


class NeuralForecastWrapper:
    """Base wrapper for N-BEATS, N-HiTS, PatchTST, TFT."""

    _MODEL_CLASS_KEY: str = ""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        merged = dict(config or {})
        merged.update(kwargs)
        self.model_params = _strip_router_keys(merged)
        self.nf_model: Any = None
        self.fitted: bool = False
        self._last_df: Optional[pd.DataFrame] = None
        # Default horizon matches typical router quick-forecast (overridable via config)
        self._fit_horizon: int = int(self.model_params.get("horizon", 30))

    def _prepare_data(self, data: pd.DataFrame, target_col: str = "close") -> pd.DataFrame:
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.copy()
            data.index = pd.to_datetime(data.index)
        _col_map = {c.lower(): c for c in data.columns}
        _tc = _col_map.get(target_col.lower(), target_col)
        if _tc not in data.columns:
            _tc = data.columns[0]
        y = pd.to_numeric(data[_tc], errors="coerce")
        return pd.DataFrame(
            {
                "unique_id": "1",
                "ds": data.index,
                "y": y.values.astype(float),
            }
        )

    @staticmethod
    def _infer_freq(df: pd.DataFrame) -> str:
        try:
            if len(df["ds"]) >= 3:
                d0 = df["ds"].iloc[1] - df["ds"].iloc[0]
                days = abs(d0.total_seconds()) / 86400.0
                if 5.5 <= days <= 7.5:
                    return "W"
                if 0.9 <= days <= 1.5:
                    return "D"
                if 2.5 <= days <= 3.5:
                    return "D"
        except Exception:
            pass
        return "D"

    def _get_model_class(self) -> Type:
        key = self._MODEL_CLASS_KEY
        mapping = {
            "NBEATS": NBEATS,
            "NHITS": NHITS,
            "PatchTST": PatchTST,
            "TFT": TFT,
        }
        cls = mapping.get(key)
        if cls is None:
            raise RuntimeError(f"Unknown NeuralForecast model key: {key}")
        return cls

    def fit(self, train_data: pd.DataFrame, target_col: str = "close", **fit_params: Any) -> Any:
        if not _lazy_import_neuralforecast():
            raise ImportError(
                "NeuralForecast is not installed. Install with: pip install neuralforecast"
            )
        try:
            df = self._prepare_data(train_data, target_col=target_col)
            self._last_df = df
            horizon = int(
                fit_params.get("horizon")
                or self.model_params.get("horizon")
                or self._fit_horizon
            )
            horizon = max(1, min(horizon, 64))
            self._fit_horizon = horizon
            input_size = int(self.model_params.get("input_size", max(horizon * 2, 14)))
            input_size = min(input_size, max(len(df) - horizon - 1, horizon + 1))
            max_steps = int(self.model_params.get("max_steps", 100))
            batch_size = int(self.model_params.get("batch_size", 32))
            freq = self.model_params.get("freq") or self._infer_freq(df)
            model_class = self._get_model_class()
            extra: Dict[str, Any] = {}
            if self._MODEL_CLASS_KEY == "PatchTST":
                pl = int(self.model_params.get("patch_len", max(horizon // 2, 4)))
                extra["patch_len"] = max(2, pl)
            m = model_class(
                h=horizon,
                input_size=input_size,
                max_steps=max_steps,
                batch_size=batch_size,
                **extra,
            )
            self.nf_model = NeuralForecast(models=[m], freq=freq)
            self.nf_model.fit(df)
            self.fitted = True
            return self
        except ImportError:
            raise
        except Exception as e:
            raise RuntimeError(f"{self._MODEL_CLASS_KEY} fit failed: {e}") from e

    def predict(self, steps: Optional[int] = None) -> np.ndarray:
        if not self.fitted or self.nf_model is None or self._last_df is None:
            raise RuntimeError("Model must be fitted before prediction")
        h = int(steps or self._fit_horizon)
        h = max(1, h)
        try:
            forecast_df = self.nf_model.predict(df=self._last_df, h=h)
        except TypeError:
            forecast_df = self.nf_model.predict(self._last_df)
        except Exception as e:
            raise RuntimeError(f"{self._MODEL_CLASS_KEY} predict failed: {e}") from e
        return self._extract_yhat(forecast_df, h)

    def _extract_yhat(self, forecast_df: pd.DataFrame, h: int) -> np.ndarray:
        skip = {"unique_id", "ds"}
        cols = [c for c in forecast_df.columns if c not in skip]
        if not cols:
            raise ValueError("No forecast columns in NeuralForecast output")
        key = self._MODEL_CLASS_KEY
        preferred = [c for c in cols if key and key.upper() in c.upper()]
        use_col = preferred[0] if preferred else cols[0]
        arr = np.asarray(forecast_df[use_col].values, dtype=float).ravel()
        if arr.size >= h:
            return arr[:h]
        return arr

    def forecast(
        self, data: pd.DataFrame, horizon: int = 30, **kwargs: Any
    ) -> Dict[str, Any]:
        try:
            if not _lazy_import_neuralforecast():
                return {
                    "forecast": np.array([], dtype=float),
                    "error": "NeuralForecast not installed",
                }
            hz = max(1, int(horizon))
            # NF models fix h at construction — refit if requested horizon changed
            if (not self.fitted) or self._fit_horizon != hz:
                self.fitted = False
                self.nf_model = None
                self.fit(data, horizon=hz, **kwargs)
            arr = self.predict(steps=hz)
            return {"forecast": np.asarray(arr, dtype=float).ravel()}
        except ImportError as e:
            logger.warning("%s forecast skipped: %s", self._MODEL_CLASS_KEY, e)
            return {"forecast": np.array([], dtype=float), "error": str(e)}
        except Exception as e:
            logger.warning("%s forecast failed: %s", self._MODEL_CLASS_KEY, e)
            return {"forecast": np.array([], dtype=float), "error": str(e)}


class TFTModel(NeuralForecastWrapper):
    """Temporal Fusion Transformer (NeuralForecast)."""

    _MODEL_CLASS_KEY = "TFT"


class NBEATSModel(NeuralForecastWrapper):
    """N-BEATS (NeuralForecast)."""

    _MODEL_CLASS_KEY = "NBEATS"


class PatchTSTModel(NeuralForecastWrapper):
    """PatchTST (NeuralForecast)."""

    _MODEL_CLASS_KEY = "PatchTST"


class NHITSModel(NeuralForecastWrapper):
    """N-HiTS (NeuralForecast)."""

    _MODEL_CLASS_KEY = "NHITS"


def get_neuralforecast_model(model_name: str, **kwargs: Any) -> NeuralForecastWrapper:
    """Factory for the four supported NeuralForecast-backed models."""
    models: Dict[str, type] = {
        "TFT": TFTModel,
        "NBEATS": NBEATSModel,
        "PatchTST": PatchTSTModel,
        "NHITS": NHITSModel,
    }
    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from: {list(models.keys())}"
        )
    return models[model_name](**kwargs)
