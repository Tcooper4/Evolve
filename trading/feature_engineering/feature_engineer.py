from trading.data.preprocessing import FeatureEngineering
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from scipy import stats
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime, timedelta

class FeatureEngineer(FeatureEngineering):
    def __init__(self, config: Optional[Dict] = None):
        """Initialize feature engineer with configuration.
        
        Args:
            config: Configuration dictionary for feature engineering
        """
        self.config = config or {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.feature_columns = []
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer all features from the input data.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=data.index)
        
        # Technical indicators
        features = pd.concat([features, self._calculate_technical_indicators(data)], axis=1)
        
        # Statistical features
        features = pd.concat([features, self._calculate_statistical_features(data)], axis=1)
        
        # Market microstructure features
        features = pd.concat([features, self._calculate_microstructure_features(data)], axis=1)
        
        # Time-based features
        features = pd.concat([features, self._calculate_time_features(data)], axis=1)
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        # Scale features
        features = self._scale_features(features)
        
        return features
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages
        for window in [20, 50, 200]:
            features[f'SMA_{window}'] = data['close'].rolling(window=window).mean()
        
        # Volatility
        features['volatility'] = features['returns'].rolling(window=20).std()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        features['MACD'] = exp1 - exp2
        features['Signal_Line'] = features['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        features['BB_Middle'] = features['close'].rolling(window=20).mean()
        features['BB_Upper'] = features['BB_Middle'] + 2 * features['close'].rolling(window=20).std()
        features['BB_Lower'] = features['BB_Middle'] - 2 * features['close'].rolling(window=20).std()
        
        # Volume features
        features['volume_ma'] = data['volume'].rolling(window=20).mean()
        features['volume_std'] = data['volume'].rolling(window=20).std()
        
        # Price momentum
        features['momentum'] = data['close'] / data['close'].shift(10) - 1
        
        return features
    
    def _calculate_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical features."""
        features = pd.DataFrame(index=data.index)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features[f'rolling_mean_{window}'] = data['close'].rolling(window=window).mean()
            features[f'rolling_std_{window}'] = data['close'].rolling(window=window).std()
            features[f'rolling_skew_{window}'] = data['close'].rolling(window=window).skew()
            features[f'rolling_kurt_{window}'] = data['close'].rolling(window=window).kurt()
        
        return features
    
    def _calculate_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features."""
        features = pd.DataFrame(index=data.index)
        
        # Bid-ask spread proxy
        features['spread'] = (data['high'] - data['low']) / data['close']
        
        # Volume profile
        features['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(window=20).mean()
        
        # Price impact
        features['price_impact'] = features['returns'].abs() / data['volume']
        
        # Order flow imbalance
        features['flow_imbalance'] = (data['close'] - data['open']) / (data['high'] - data['low'])
        
        return features
    
    def _calculate_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based features."""
        features = pd.DataFrame(index=data.index)
        
        # Time of day
        features['hour'] = data.index.hour
        features['day_of_week'] = data.index.dayofweek
        features['month'] = data.index.month
        
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        return features
    
    def _scale_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Scale features using StandardScaler."""
        # Fit scaler if not already fitted
        if not hasattr(self.scaler, 'mean_'):
            self.scaler.fit(features)
        
        # Transform features
        scaled_features = pd.DataFrame(
            self.scaler.transform(features),
            index=features.index,
            columns=features.columns
        )
        
        return scaled_features
    
    def reduce_dimensions(self, features: pd.DataFrame) -> pd.DataFrame:
        """Reduce feature dimensions using PCA."""
        # Fit PCA if not already fitted
        if not hasattr(self.pca, 'components_'):
            self.pca.fit(features)
        
        # Transform features
        reduced_features = pd.DataFrame(
            self.pca.transform(features),
            index=features.index,
            columns=[f'pc_{i+1}' for i in range(self.pca.n_components_)]
        )
        
        return reduced_features
    
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis features."""
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['close'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['close'].rolling(window=20).std()
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        
        # Price momentum
        df['momentum'] = df['close'] / df['close'].shift(10) - 1
        
        # Drop NaN values
        df = df.dropna()
        
        # Store feature columns
        self.feature_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    def create_fundamental_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create fundamental analysis features."""
        df = data.copy()
        
        # Add fundamental features here
        # This is a placeholder for actual fundamental data
        
        return df
    
    def create_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment analysis features."""
        df = data.copy()
        
        # Add sentiment features here
        # This is a placeholder for actual sentiment data
        
        return df
    
    def scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale features using StandardScaler."""
        if not self.feature_columns:
            return data
        
        df = data.copy()
        df[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])
        return df
    
    def create_target_variable(self, data: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """Create target variable for prediction."""
        df = data.copy()
        df['target'] = df['close'].shift(-horizon) / df['close'] - 1
        return df
    
    def prepare_training_data(self, data: pd.DataFrame, target_col: str = 'target') -> tuple:
        """Prepare data for training."""
        df = data.copy()
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Split features and target
        X = df[self.feature_columns]
        y = df[target_col]
        
        return X, y
    
    def get_feature_importance(self, model) -> pd.DataFrame:
        """Get feature importance from model."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = model.coef_
        else:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def get_feature_metrics(self) -> Dict[str, int]:
        """Get feature engineering metrics."""
        return {
            'num_features': len(self.feature_columns),
            'feature_types': {
                'technical': len([f for f in self.feature_columns if f.startswith(('SMA', 'RSI', 'MACD', 'BB'))]),
                'fundamental': len([f for f in self.feature_columns if f.startswith(('PE', 'PB', 'ROE'))]),
                'sentiment': len([f for f in self.feature_columns if f.startswith(('sentiment', 'emotion'))])
            }
        } 