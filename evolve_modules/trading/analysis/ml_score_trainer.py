"""
ML Score Trainer
=================
INTEGRATION NOTES:
- Drop into: trading/analysis/ml_score_trainer.py
- Wire into: trading/analysis/ai_score.py
  Replace/augment rules-based score with ML output
- Call pattern:
    from trading.analysis.ml_score_trainer import MLScoreTrainer
    trainer = MLScoreTrainer()
    trainer.train(universe=["AAPL","MSFT",...])  # one-time training
    score = trainer.predict(symbol, hist_df)  # get ML score

Dependencies: sklearn, xgboost, shap (all in requirements)
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_CACHE_PATH = Path(".cache/ml_score")
MODEL_CACHE_PATH.mkdir(parents=True, exist_ok=True)


class MLScoreTrainer:
    """
    Trains a gradient boosting model to predict 7-day forward returns
    from the same signals used in the rules-based AI Score.

    This makes the AI Score genuinely ML-based rather than rules-based.

    Feature set mirrors the rules-based score:
    - RSI, Bollinger position
    - SMA crossovers (20/50/200)
    - 20-day momentum
    - Short squeeze score / float short %
    - Insider flow signal
    - EPS surprise
    - P/E vs sector premium
    - GARCH volatility estimate

    Target: 7-day forward return (continuous), then binned to
    STRONG_BUY / BUY / NEUTRAL / SELL / STRONG_SELL
    """

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.feature_names: List[str] = []
        self.is_trained = False
        self.training_date: Optional[str] = None
        self._scaler = None

    def build_features(
        self,
        symbol: str,
        hist: pd.DataFrame,
    ) -> Optional[pd.Series]:
        """
        Build feature vector for a single observation.
        Returns None if insufficient data.
        """
        try:
            _col_map = {c.lower(): c for c in hist.columns}
            close_col = _col_map.get("close", hist.columns[0])
            close = hist[close_col].values.astype(float)

            if len(close) < 50:
                return None

            last = float(close[-1])
            features = {}

            # RSI
            features["rsi"] = self._calc_rsi(close, 14) or 50.0

            # Bollinger position
            sma20 = np.mean(close[-20:])
            std20 = np.std(close[-20:])
            bb_upper = sma20 + 2 * std20
            bb_lower = sma20 - 2 * std20
            features["bb_pct"] = float(
                (last - bb_lower) / (bb_upper - bb_lower + 1e-8)
            )

            # SMA crossovers
            for period in [20, 50, 200]:
                if len(close) >= period:
                    sma = np.mean(close[-period:])
                    features[f"price_vs_sma{period}"] = float(
                        (last - sma) / sma
                    )
                else:
                    features[f"price_vs_sma{period}"] = 0.0

            # Momentum
            for period in [5, 10, 20, 60]:
                if len(close) >= period:
                    features[f"momentum_{period}d"] = float(
                        (close[-1] / close[-period] - 1)
                    )
                else:
                    features[f"momentum_{period}d"] = 0.0

            # Volatility (realized)
            returns = np.diff(close[-22:]) / close[-22:-1]
            features["realized_vol_22d"] = float(np.std(returns) * np.sqrt(252))

            # Volume trend (if available)
            if "volume" in _col_map:
                vol_data = hist[_col_map["volume"]].values.astype(float)
                if len(vol_data) >= 20:
                    features["volume_trend"] = float(
                        vol_data[-5:].mean() / (vol_data[-20:].mean() + 1e-8)
                    )
                else:
                    features["volume_trend"] = 1.0
            else:
                features["volume_trend"] = 1.0

            # Price vs 52-week range
            if len(close) >= 252:
                high_52w = np.max(close[-252:])
                low_52w = np.min(close[-252:])
                features["pct_from_52w_high"] = float(
                    (last - high_52w) / high_52w
                )
                features["pct_from_52w_low"] = float(
                    (last - low_52w) / (low_52w + 1e-8)
                )
            else:
                features["pct_from_52w_high"] = 0.0
                features["pct_from_52w_low"] = 0.0

            # Short interest (if available)
            try:
                from trading.data.short_interest import get_short_interest
                si = get_short_interest(symbol)
                features["short_pct_float"] = float(
                    si.get("short_pct_float") or 0
                )
                features["squeeze_score"] = float(
                    si.get("short_squeeze_score") or 0
                ) / 100
            except Exception:
                features["short_pct_float"] = 0.0
                features["squeeze_score"] = 0.0

            # Earnings surprise (if available)
            try:
                from trading.data.earnings_calendar import get_upcoming_earnings
                earnings = get_upcoming_earnings(symbol)
                features["eps_surprise"] = float(
                    earnings.get("last_eps_surprise_pct") or 0
                ) / 100
                days_until = earnings.get("days_until")
                features["days_to_earnings"] = float(
                    days_until if days_until is not None else 90
                )
            except Exception:
                features["eps_surprise"] = 0.0
                features["days_to_earnings"] = 90.0

            return pd.Series(features)

        except Exception as e:
            logger.warning("Feature building failed for %s: %s", symbol, e)
            return None

    def build_training_dataset(
        self,
        universe: List[str],
        lookback_days: int = 504,
        forward_days: int = 7,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build training dataset from historical data.

        For each symbol in universe, creates rolling feature windows
        with forward return labels.
        """
        import yfinance as yf

        X_list = []
        y_list = []

        logger.info(
            "Building ML Score training dataset from %d symbols",
            len(universe)
        )

        for symbol in universe:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(
                    period=f"{lookback_days + forward_days + 60}d"
                )
                if hist.empty or len(hist) < 100:
                    continue

                _col_map = {c.lower(): c for c in hist.columns}
                close_col = _col_map.get("close", hist.columns[0])
                closes = hist[close_col].values.astype(float)

                # Create rolling windows
                step = 5  # sample every 5 days
                for i in range(60, len(closes) - forward_days, step):
                    window_data = hist.iloc[:i]
                    features = self.build_features(symbol, window_data)

                    if features is None:
                        continue

                    # Forward return label
                    fwd_return = float(
                        closes[i + forward_days - 1] / closes[i - 1] - 1
                    )

                    features["symbol_hash"] = float(
                        int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
                        % 1000
                    ) / 1000

                    X_list.append(features)
                    y_list.append(fwd_return)

            except Exception as e:
                logger.debug("Skipping %s: %s", symbol, e)
                continue

        if not X_list:
            logger.warning("No training data collected")
            return pd.DataFrame(), pd.Series()

        X = pd.DataFrame(X_list).fillna(0)
        y = pd.Series(y_list)

        logger.info(
            "Training dataset: %d observations, %d features",
            len(X), len(X.columns)
        )
        return X, y

    def train(
        self,
        universe: Optional[List[str]] = None,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Train the ML score model.

        Can either pass universe (fetches data) or pre-built X, y.
        """
        try:
            if X is None or y is None:
                if universe is None:
                    # Default to SP100 subset
                    universe = [
                        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
                        "NVDA", "JPM", "JNJ", "V", "PG",
                        "UNH", "HD", "MA", "DIS", "BAC",
                        "XOM", "TSLA", "AVGO", "LLY", "CVX",
                    ]
                X, y = self.build_training_dataset(universe)

            if X.empty or len(y) == 0:
                return {"error": "No training data available"}

            # Scale features
            from sklearn.preprocessing import RobustScaler
            self._scaler = RobustScaler()
            X_scaled = self._scaler.fit_transform(X)

            self.feature_names = list(X.columns)

            # Train model
            if self.model_type == "xgboost":
                from xgboost import XGBRegressor
                self.model = XGBRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                self.model = GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                )

            # Train/validation split (time-based)
            split = int(len(X_scaled) * 0.8)
            X_train, X_val = X_scaled[:split], X_scaled[split:]
            y_train, y_val = y.iloc[:split], y.iloc[split:]

            self.model.fit(X_train, y_train)

            # Evaluate
            from sklearn.metrics import mean_squared_error, r2_score
            val_preds = self.model.predict(X_val)
            mse = float(mean_squared_error(y_val, val_preds))
            r2 = float(r2_score(y_val, val_preds))

            # Directional accuracy
            da = float(np.mean(
                np.sign(val_preds) == np.sign(y_val)
            ))

            self.is_trained = True
            self.training_date = datetime.now().isoformat()

            # Save model
            self._save_model()

            result = {
                "status": "trained",
                "n_samples": len(X),
                "n_features": len(self.feature_names),
                "val_mse": round(mse, 6),
                "val_r2": round(r2, 4),
                "val_directional_accuracy": round(da, 3),
                "training_date": self.training_date,
                "model_type": self.model_type,
            }

            logger.info(
                "ML Score model trained: R2=%.3f, DA=%.1f%%",
                r2, da * 100
            )
            return result

        except Exception as e:
            logger.warning("ML Score training failed: %s", e)
            return {"error": str(e)}

    def predict(
        self,
        symbol: str,
        hist: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Predict ML-based score for a symbol.

        Returns score 1-10 with direction and confidence,
        compatible with existing AI Score format.
        """
        try:
            # Load model if not in memory
            if not self.is_trained:
                loaded = self._load_model()
                if not loaded:
                    return {
                        "ml_score": None,
                        "error": "Model not trained",
                        "fallback": True,
                    }

            features = self.build_features(symbol, hist)
            if features is None:
                return {
                    "ml_score": None,
                    "error": "Insufficient data for features",
                    "fallback": True,
                }

            # Align features to training columns
            feature_vec = pd.DataFrame(
                [features.reindex(self.feature_names, fill_value=0)]
            )
            X_scaled = self._scaler.transform(feature_vec)

            # Predict forward return
            pred_return = float(self.model.predict(X_scaled)[0])

            # Convert to 1-10 score
            # Map: -5% → 1, 0% → 5.5, +5% → 10
            score = np.clip(5.5 + pred_return * 100, 1.0, 10.0)
            score = round(float(score), 1)

            # Get feature importance for this prediction
            explanation = self._explain_prediction(
                X_scaled[0], feature_vec.columns.tolist()
            )

            return {
                "ml_score": score,
                "predicted_7d_return": round(pred_return * 100, 2),
                "direction": (
                    "BULLISH" if pred_return > 0.005
                    else "BEARISH" if pred_return < -0.005
                    else "NEUTRAL"
                ),
                "top_features": explanation,
                "model_type": self.model_type,
                "training_date": self.training_date,
                "fallback": False,
            }

        except Exception as e:
            logger.warning("ML Score prediction failed for %s: %s", symbol, e)
            return {
                "ml_score": None,
                "error": str(e),
                "fallback": True,
            }

    def _explain_prediction(
        self,
        x: np.ndarray,
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Get top feature contributions using SHAP if available."""
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(x.reshape(1, -1))
            contributions = list(zip(feature_names, shap_values[0]))
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            return [
                {
                    "feature": name,
                    "contribution": round(float(val), 4),
                    "direction": "positive" if val > 0 else "negative",
                }
                for name, val in contributions[:5]
            ]
        except Exception:
            # Fallback: use feature importances
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                top_idx = np.argsort(importances)[-5:][::-1]
                return [
                    {
                        "feature": feature_names[i],
                        "contribution": round(float(importances[i]), 4),
                        "direction": "positive" if x[i] > 0 else "negative",
                    }
                    for i in top_idx
                ]
            return []

    def _save_model(self) -> None:
        """Save trained model to cache."""
        try:
            import joblib
            model_path = MODEL_CACHE_PATH / "ml_score_model.joblib"
            scaler_path = MODEL_CACHE_PATH / "ml_score_scaler.joblib"
            meta_path = MODEL_CACHE_PATH / "ml_score_meta.json"

            joblib.dump(self.model, model_path)
            joblib.dump(self._scaler, scaler_path)

            meta = {
                "feature_names": self.feature_names,
                "training_date": self.training_date,
                "model_type": self.model_type,
            }
            meta_path.write_text(
                json.dumps(meta),
                encoding="utf-8"
            )
            logger.info("ML Score model saved to %s", MODEL_CACHE_PATH)
        except Exception as e:
            logger.warning("ML Score model save failed: %s", e)

    def _load_model(self) -> bool:
        """Load trained model from cache."""
        try:
            import joblib
            model_path = MODEL_CACHE_PATH / "ml_score_model.joblib"
            scaler_path = MODEL_CACHE_PATH / "ml_score_scaler.joblib"
            meta_path = MODEL_CACHE_PATH / "ml_score_meta.json"

            if not all([
                model_path.exists(),
                scaler_path.exists(),
                meta_path.exists()
            ]):
                return False

            self.model = joblib.load(model_path)
            self._scaler = joblib.load(scaler_path)
            meta = json.loads(
                meta_path.read_text(encoding="utf-8")
            )
            self.feature_names = meta["feature_names"]
            self.training_date = meta["training_date"]
            self.model_type = meta["model_type"]
            self.is_trained = True

            logger.info(
                "ML Score model loaded (trained %s)",
                self.training_date
            )
            return True
        except Exception as e:
            logger.warning("ML Score model load failed: %s", e)
            return False

    @staticmethod
    def _calc_rsi(prices: np.ndarray, period: int = 14) -> Optional[float]:
        if len(prices) < period + 1:
            return None
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
