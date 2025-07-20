"""
Time Series Dataset

This module provides the TimeSeriesDataset class for handling time series data
in PyTorch models.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Try to import PyTorch
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError as e:
    print("âš ï¸ PyTorch not available. Disabling deep learning datasets.")
    print(f"   Missing: {e}")
    torch = None
    Dataset = None
    TORCH_AVAILABLE = False

# Try to import scikit-learn
try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print("âš ï¸ scikit-learn not available. Disabling data preprocessing.")
    print(f"   Missing: {e}")
    StandardScaler = None
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""


class TimeSeriesDataset(Dataset):
    """Dataset for time series data."""

    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int,
        target_col: str,
        feature_cols: List[str],
        scaler: Optional[StandardScaler] = None,
    ):
        """Initialize dataset.

        Args:
            data: Input data
            sequence_length: Length of input sequences
            target_col: Name of target column
            feature_cols: List of feature column names
            scaler: Optional scaler for features
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Cannot create TimeSeriesDataset.")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not available. Cannot perform data scaling.")
        
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.feature_cols = feature_cols

        # Validate data
        self._validate_data(data)

        # Scale features
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(data[feature_cols])
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(data[feature_cols])

        # Get targets
        self.targets = data[target_col].values

        # Create sequences
        self.sequences = []
        self.sequence_targets = []

        for i in range(len(data) - sequence_length):
            self.sequences.append(self.features[i : i + sequence_length])
            self.sequence_targets.append(self.targets[i + sequence_length])

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data.

        Args:
            data: Input data to validate

        Raises:
            ValidationError: If data is invalid
        """
        # Check for missing values
        if data.isnull().any().any():
            raise ValidationError("Data contains missing values")

        # Check for infinite values
        if np.isinf(data.select_dtypes(include=np.number)).any().any():
            raise ValidationError("Data contains infinite values")

        # Check for required columns
        missing_cols = [
            col
            for col in self.feature_cols + [self.target_col]
            if col not in data.columns
        ]
        if missing_cols:
            raise ValidationError(f"Missing required columns: {missing_cols}")

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dataset item.

        Args:
            idx: Index of item to get

        Returns:
            Tuple of (features, target)
        """
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.sequence_targets[idx]]),
        )

    def get_scaler(self) -> StandardScaler:
        """Get the fitted scaler.

        Returns:
            Fitted StandardScaler
        """
        return self.scaler

    def inverse_transform_features(self, features: np.ndarray) -> np.ndarray:
        """Inverse transform scaled features.

        Args:
            features: Scaled features

        Returns:
            Original scale features
        """
        return self.scaler.inverse_transform(features)
