"""
Transformer-based time series forecasting model.
Uses a simple encoder-only Transformer architecture for price prediction.
"""
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from trading.exceptions import ModelPredictionError

logger = logging.getLogger(__name__)


class TransformerForecaster:
    """
    Transformer forecaster using PyTorch's built-in TransformerEncoder.
    Falls back to a Ridge regression baseline if torch is unavailable.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.target_column = self.config.get("target_column", "Close")
        self.horizon = self.config.get("horizon", 30)
        self.seq_len = self.config.get("seq_len", 60)
        self.d_model = self.config.get("d_model", 64)
        self.nhead = self.config.get("nhead", 4)
        self.num_layers = self.config.get("num_layers", 2)
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self._torch_available = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_model()

    def _setup_model(self):
        """Initialize model architecture."""
        try:
            import torch  # noqa: F401
            self._torch_available = True
            self.logger.info("Transformer: torch available, using TransformerEncoder")
        except ImportError:
            self._torch_available = False
            self.logger.warning("Transformer: torch not available, using Ridge fallback")
        self._model_initialized = True

    def _build_torch_model(self, input_size: int):
        import torch.nn as nn

        class _TransformerModel(nn.Module):
            def __init__(self, input_size, d_model, nhead, num_layers, horizon):
                super().__init__()
                self.input_proj = nn.Linear(input_size, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.output = nn.Linear(d_model, horizon)

            def forward(self, x):
                x = self.input_proj(x)
                x = self.encoder(x)
                return self.output(x[:, -1, :])

        return _TransformerModel(
            input_size, self.d_model, self.nhead, self.num_layers, self.horizon
        )

    def _prepare_features(self, df: pd.DataFrame):
        """Extract and normalize features from DataFrame."""
        col = self.target_column if self.target_column in df.columns else "Close"
        if col not in df.columns:
            raise ModelPredictionError(f"Column '{col}' not found in data")
        prices = df[col].values.astype(float)
        returns = np.diff(prices) / (prices[:-1] + 1e-8)
        return returns, prices

    def fit(self, data: pd.DataFrame, **kwargs) -> "TransformerForecaster":
        """Train the transformer model."""
        try:
            returns, prices = self._prepare_features(data)
            self.last_price_ = float(prices[-1])
            self.last_returns_ = (
                returns[-self.seq_len :] if len(returns) >= self.seq_len else returns
            )

            if self._torch_available and len(returns) >= self.seq_len + 10:
                self._fit_torch(returns)
            else:
                from sklearn.linear_model import Ridge
                from sklearn.preprocessing import StandardScaler

                X, y = self._make_sequences(returns)
                if len(X) < 5:
                    raise ModelPredictionError("Insufficient data for Transformer")
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
                self._fallback = Ridge(alpha=1.0)
                self._fallback.fit(X_scaled, y)
            self.is_fitted = True
            return self
        except ModelPredictionError:
            raise
        except Exception as e:
            raise ModelPredictionError(f"TransformerForecaster fit failed: {e}")

    def _make_sequences(self, returns: np.ndarray):
        """Create sliding window sequences."""
        seq_len = min(self.seq_len, len(returns) // 3)
        X, y = [], []
        for i in range(len(returns) - seq_len - 1):
            X.append(returns[i : i + seq_len])
            y.append(returns[i + seq_len])
        return np.array(X), np.array(y)

    def _fit_torch(self, returns: np.ndarray):
        import torch
        import torch.nn as nn

        X, y = self._make_sequences(returns)
        X_t = torch.FloatTensor(X).unsqueeze(-1)
        y_t = torch.FloatTensor(y)
        self.torch_model = self._build_torch_model(1)
        optimizer = torch.optim.Adam(self.torch_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        self.torch_model.train()
        for _ in range(20):
            optimizer.zero_grad()
            pred = self.torch_model(X_t)
            loss = criterion(pred[:, 0], y_t)
            loss.backward()
            optimizer.step()

    def predict(
        self, data: pd.DataFrame, horizon: Optional[int] = None
    ) -> np.ndarray:
        """Generate multi-step ahead price forecasts."""
        if not self.is_fitted:
            self.fit(data)
        horizon = horizon or self.horizon
        try:
            returns, prices = self._prepare_features(data)
            last_price = float(prices[-1])
            recent_returns = (
                returns[-self.seq_len :] if len(returns) >= self.seq_len else returns
            )

            if self._torch_available and hasattr(self, "torch_model"):
                forecasted_returns = self._predict_torch(recent_returns, horizon)
            elif hasattr(self, "_fallback"):
                forecasted_returns = self._predict_ridge(recent_returns, horizon)
            else:
                forecasted_returns = np.full(
                    horizon, float(np.mean(recent_returns)) if len(recent_returns) else 0.0
                )

            forecast_prices = [last_price]
            for r in forecasted_returns[:horizon]:
                forecast_prices.append(forecast_prices[-1] * (1 + float(r)))
            return np.array(forecast_prices[1 : horizon + 1])
        except Exception as e:
            raise ModelPredictionError(f"TransformerForecaster predict failed: {e}")

    def _predict_ridge(self, recent_returns: np.ndarray, horizon: int) -> np.ndarray:
        seq_len = min(self.seq_len, len(recent_returns))
        preds = []
        window = list(recent_returns[-seq_len:])
        n_features = getattr(self.scaler, "n_features_in_", seq_len)
        for _ in range(horizon):
            x = np.array(window[-seq_len:]).reshape(1, -1)
            if x.shape[1] < n_features:
                x = np.pad(x, ((0, 0), (0, n_features - x.shape[1])))
            elif x.shape[1] > n_features:
                x = x[:, -n_features:]
            x_scaled = self.scaler.transform(x)
            r = float(self._fallback.predict(x_scaled)[0])
            r = np.clip(r, -0.05, 0.05)
            preds.append(r)
            window.append(r)
        return np.array(preds)

    def _predict_torch(self, recent_returns: np.ndarray, horizon: int) -> np.ndarray:
        import torch

        preds = []
        window = list(recent_returns[-self.seq_len :])
        self.torch_model.eval()
        with torch.no_grad():
            for _ in range(horizon):
                x = (
                    torch.FloatTensor(window[-self.seq_len :])
                    .unsqueeze(0)
                    .unsqueeze(-1)
                )
                r = float(self.torch_model(x)[0, 0])
                r = np.clip(r, -0.05, 0.05)
                preds.append(r)
                window.append(r)
        return np.array(preds)

    def forecast(self, data: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
        """
        Standard forecast interface matching other models in this project.
        Returns dict with 'forecast' key containing price-space predictions.
        """
        try:
            if not self.is_fitted:
                self.fit(data)
            prices = self.predict(data, horizon=horizon)
            last_price = float(
                data[self.target_column].iloc[-1]
                if self.target_column in data.columns
                else data["Close"].iloc[-1]
            )
            return {
                "forecast": prices.tolist(),
                "model": "Transformer",
                "horizon": horizon,
                "last_price": last_price,
                "already_denormalized": True,
            }
        except ModelPredictionError:
            raise
        except Exception as e:
            raise ModelPredictionError(f"Transformer forecast failed: {e}")
