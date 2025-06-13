from trading.data.preprocessing import FeatureEngineering
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from scipy import stats
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

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
        
        # Verify indicators
        self._verify_indicators(features)
        
        return features
    
    def _verify_indicators(self, features: pd.DataFrame) -> None:
        """Verify that all indicators are being calculated correctly.
        
        Args:
            features: DataFrame containing all calculated features
        """
        expected_indicators = {
            # Trend Indicators
            'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_20', 'EMA_50', 'EMA_200',
            'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
            'ADX_14',
            'ICHIMOKU_9_26_52', 'ICHIMOKU_9_26_52_26', 'ICHIMOKU_9_26_52_52',
            'PSAR_0.02_0.02_0.2',
            'SUPERT_10_3.0',
            
            # Momentum Indicators
            'RSI_14',
            'STOCH_14_3_3', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
            'CCI_14',
            'WILLR_14',
            'MOM_10',
            'ROC_10',
            'MFI_14',
            'TRIX_18_9',
            'MASSI_9',
            'DPO_20',
            'KST_10_15_20_30_10_10_10_15',
            
            # Volatility Indicators
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
            'ATR_14',
            'NATR_14',
            'TRUERANGE_1',
            
            # Volume Indicators
            'OBV',
            'VWAP',
            'PVT',
            'EFI_13',
            'CFI_14',
            'EBSW_10',
            
            # Custom Indicators
            'TSI_13_25',
            'UO_7_14_28',
            'AO_5_34',
            'BOP',
            'CMO_14',
            'PPO_12_26_9'
        }
        
        # Check for missing indicators
        missing_indicators = expected_indicators - set(features.columns)
        if missing_indicators:
            logger.warning(f"Missing indicators: {missing_indicators}")
        
        # Check for NaN values
        nan_columns = features.columns[features.isna().any()].tolist()
        if nan_columns:
            logger.warning(f"Columns with NaN values: {nan_columns}")
        
        # Check for infinite values
        inf_columns = features.columns[np.isinf(features).any()].tolist()
        if inf_columns:
            logger.warning(f"Columns with infinite values: {inf_columns}")
        
        # Log summary
        logger.info(f"Total indicators calculated: {len(features.columns)}")
        logger.info(f"Expected indicators: {len(expected_indicators)}")
        logger.info(f"Missing indicators: {len(missing_indicators)}")
        logger.info(f"Columns with NaN values: {len(nan_columns)}")
        logger.info(f"Columns with infinite values: {len(inf_columns)}")
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using pandas_ta."""
        # Create a custom strategy
        custom_strategy = ta.Strategy(
            name="custom_strategy",
            description="Custom technical analysis strategy",
            ta=[
                # Trend Indicators
                {"kind": "sma", "length": 20},
                {"kind": "sma", "length": 50},
                {"kind": "sma", "length": 200},
                {"kind": "ema", "length": 20},
                {"kind": "ema", "length": 50},
                {"kind": "ema", "length": 200},
                {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                {"kind": "adx", "length": 14},
                {"kind": "ichimoku", "tenkan": 9, "kijun": 26, "senkou": 52},
                {"kind": "psar", "af0": 0.02, "af": 0.02, "max_af": 0.2},
                {"kind": "supertrend", "length": 10, "multiplier": 3},
                
                # Momentum Indicators
                {"kind": "rsi", "length": 14},
                {"kind": "stoch", "k": 14, "d": 3},
                {"kind": "cci", "length": 14},
                {"kind": "willr", "length": 14},
                {"kind": "mom", "length": 10},
                {"kind": "roc", "length": 10},
                {"kind": "mfi", "length": 14},
                {"kind": "trix", "length": 18, "signal": 9},
                {"kind": "massi", "length": 9},
                {"kind": "dpo", "length": 20},
                {"kind": "kst", "roc1": 10, "roc2": 15, "roc3": 20, "roc4": 30, "sma1": 10, "sma2": 10, "sma3": 10, "sma4": 15},
                
                # Volatility Indicators
                {"kind": "bbands", "length": 20, "std": 2},
                {"kind": "atr", "length": 14},
                {"kind": "natr", "length": 14},
                {"kind": "tr"},
                {"kind": "true_range"},
                
                # Volume Indicators
                {"kind": "obv"},
                {"kind": "vwap"},
                {"kind": "pvt"},
                {"kind": "efi", "length": 13},
                {"kind": "cfi", "length": 14},
                {"kind": "ebsw", "length": 10},
                
                # Custom Indicators
                {"kind": "tsi", "fast": 13, "slow": 25},
                {"kind": "uo", "fast": 7, "medium": 14, "slow": 28},
                {"kind": "ao", "fast": 5, "slow": 34},
                {"kind": "bop"},
                {"kind": "cmo", "length": 14},
                {"kind": "ppo", "fast": 12, "slow": 26, "signal": 9}
            ]
        )

        # Add the strategy to the DataFrame
        data.ta.strategy(custom_strategy)
        
        # Get all the technical indicators
        features = data.ta.indicators()
        
        # Add basic price-based features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['volatility'] = features['returns'].rolling(window=20).std()
        
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