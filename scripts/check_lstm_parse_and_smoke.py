from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


def check_parse() -> None:
    path = Path("trading/models/lstm_model.py")
    src = path.read_text(encoding="utf-8", errors="replace")
    ast.parse(src)
    print("Parse OK trading/models/lstm_model.py")


def check_smoke() -> None:
    from trading.models.lstm_model import LSTMForecaster

    cfg = {
        "target_column": "close",
        "feature_columns": ["close", "volume"],
        "sequence_length": 20,
    }
    m = LSTMForecaster(cfg)
    data = pd.DataFrame(
        {
            "close": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(1000, 9999, 100).astype(float),
        }
    )
    m.fit(data)
    r = m.forecast(data, horizon=5)
    print("LSTM OK:", r[:3])


if __name__ == "__main__":
    check_parse()
    check_smoke()

