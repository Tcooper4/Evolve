import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a DataFrame for safe Streamlit
    display. Prevents Arrow serialization errors
    by ensuring consistent column types.

    - Datetime columns → string (ISO format)
    - Object columns with mixed types → string
    - Numeric columns → kept as-is
    - NaN/None → empty string for object cols,
      0.0 for numeric cols
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].dt.strftime("%Y-%m-%d").fillna("")
            elif pd.api.types.is_object_dtype(out[col]):
                out[col] = out[col].fillna("").astype(str)
            elif pd.api.types.is_numeric_dtype(out[col]):
                out[col] = (
                    pd.to_numeric(out[col], errors="coerce").fillna(0.0)
                )
        except Exception as e:
            logger.warning(
                "normalize_for_display: col '%s' failed normalization: %s",
                col,
                e,
            )
            out[col] = out[col].astype(str)
    return out

