"""
Data pipeline module for orchestrating data loading, validation, preprocessing, and analysis.
"""
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime
import time
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPipeline:
    """Class for managing the data pipeline process."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data pipeline.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.data = None
        self.processed_data = None
        self.pipeline_start_time = None
        self.pipeline_end_time = None
        
        # Initialize validation and preprocessing components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize pipeline components."""
        try:
            from .data_validation import DataValidator
            self.validator = DataValidator()
            logger.info("DataValidator initialized successfully")
        except ImportError:
            logger.warning("DataValidator not available, using basic validation")
            self.validator = None

    def load_data(self, file_path: Union[str, Path], **kwargs) -> bool:
        """
        Load data from a file with traceable logging.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments to pass to pandas read function
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        start_time = time.time()
        logger.info(f"🔄 Starting data loading from: {file_path}")
        
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"❌ File not found: {file_path}")
                return False
                
            # Determine file type and load accordingly
            if file_path.suffix.lower() == '.csv':
                self.data = pd.read_csv(file_path, **kwargs)
                logger.info(f"📄 Loaded CSV file: {file_path}")
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path, **kwargs)
                logger.info(f"📊 Loaded Excel file: {file_path}")
            elif file_path.suffix.lower() == '.parquet':
                self.data = pd.read_parquet(file_path, **kwargs)
                logger.info(f"📦 Loaded Parquet file: {file_path}")
            else:
                logger.error(f"❌ Unsupported file format: {file_path.suffix}")
                return False
            
            load_time = time.time() - start_time
            logger.info(f"✅ Successfully loaded {len(self.data)} rows from {file_path} in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            return False

    def validate_data(self) -> Tuple[bool, str]:
        """
        Validate the loaded data with detailed logging.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        start_time = time.time()
        logger.info("🔍 Starting data validation...")
        
        if self.data is None:
            logger.error("❌ No data loaded for validation")
            return False, "No data loaded"
        
        # Log data shape and basic info
        logger.info(f"📊 Data shape: {self.data.shape}")
        logger.info(f"📊 Data columns: {list(self.data.columns)}")
        logger.info(f"📊 Data types: {self.data.dtypes.to_dict()}")
        
        if self.validator:
            is_valid, error_message = self.validator.validate_dataframe(self.data)
        else:
            # Basic validation
            is_valid, error_message = self._basic_validation()
        
        validation_time = time.time() - start_time
        
        if is_valid:
            logger.info(f"✅ Data validation passed in {validation_time:.2f}s")
        else:
            logger.error(f"❌ Data validation failed: {error_message}")
        
        return is_valid, error_message

    def _basic_validation(self) -> Tuple[bool, str]:
        """Basic data validation when DataValidator is not available."""
        try:
            # Check if dataframe is empty
            if self.data.empty:
                return False, "DataFrame is empty"
            
            # Check for required columns (basic OHLCV)
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}"
            
            # Check for NaN values
            nan_counts = self.data[required_cols].isna().sum()
            if nan_counts.any():
                logger.warning(f"⚠️ Found NaN values: {nan_counts.to_dict()}")
            
            # Check data types
            numeric_cols = self.data[required_cols].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) != len(required_cols):
                return False, "Not all required columns are numeric"
            
            return True, "Basic validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def preprocess_data(self) -> bool:
        """
        Preprocess the loaded data with traceable logging.
        
        Returns:
            bool: True if preprocessing was successful, False otherwise
        """
        start_time = time.time()
        logger.info("⚙️ Starting data preprocessing...")
        
        try:
            if self.data is None:
                logger.error("❌ No data loaded for preprocessing")
                return False
            
            # Log preprocessing configuration
            logger.info(f"⚙️ Preprocessing config: {self.config}")
            
            # Handle missing data
            missing_method = self.config.get('missing_data_method', 'ffill')
            logger.info(f"🔧 Handling missing data with method: {missing_method}")
            
            if self.validator:
                self.data = self.validator.handle_missing_data(self.data, method=missing_method)
            else:
                self.data = self.data.fillna(method=missing_method)
            
            # Remove outliers if configured
            if self.config.get('remove_outliers', False):
                columns = self.config.get('outlier_columns', ['close', 'volume'])
                n_std = self.config.get('outlier_std', 3.0)
                logger.info(f"🔧 Removing outliers from {columns} with {n_std} std")
                
                if self.validator:
                    self.data = self.validator.remove_outliers(self.data, columns, n_std)
                else:
                    self.data = self._remove_outliers_basic(self.data, columns, n_std)
            
            # Preprocess data
            if self.validator:
                self.processed_data = self.validator.preprocess_data(self.data)
            else:
                self.processed_data = self._preprocess_basic(self.data)
            
            # Normalize data if configured
            if self.config.get('normalize', False):
                columns = self.config.get('normalize_columns', ['close', 'volume'])
                method = self.config.get('normalize_method', 'zscore')
                logger.info(f"🔧 Normalizing {columns} with method: {method}")
                
                if self.validator:
                    self.processed_data = self.validator.normalize_data(
                        self.processed_data, columns, method
                    )
                else:
                    self.processed_data = self._normalize_basic(self.processed_data, columns, method)
            
            preprocessing_time = time.time() - start_time
            logger.info(f"✅ Data preprocessing completed in {preprocessing_time:.2f}s")
            logger.info(f"📊 Processed data shape: {self.processed_data.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing data: {str(e)}")
            return False

    def _remove_outliers_basic(self, df: pd.DataFrame, columns: List[str], n_std: float) -> pd.DataFrame:
        """Basic outlier removal."""
        df_cleaned = df.copy()
        for col in columns:
            if col in df_cleaned.columns:
                mean = df_cleaned[col].mean()
                std = df_cleaned[col].std()
                threshold = n_std * std
                df_cleaned.loc[abs(df_cleaned[col] - mean) > threshold, col] = np.nan
        return df_cleaned
    
    def _preprocess_basic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing."""
        df_processed = df.copy()
        
        # Sort by index if it's datetime
        if isinstance(df_processed.index, pd.DatetimeIndex):
            df_processed = df_processed.sort_index()
        
        # Calculate returns
        if 'close' in df_processed.columns:
            df_processed['returns'] = df_processed['close'].pct_change()
            df_processed['log_returns'] = np.log(df_processed['close'] / df_processed['close'].shift(1))
        
        # Calculate volatility
        if 'returns' in df_processed.columns:
            df_processed['volatility'] = df_processed['returns'].rolling(window=20).std()
        
        # Remove the first row which will have NaN values
        df_processed = df_processed.iloc[1:]
        
        return df_processed
    
    def _normalize_basic(self, df: pd.DataFrame, columns: List[str], method: str) -> pd.DataFrame:
        """Basic normalization."""
        df_normalized = df.copy()
        
        for col in columns:
            if col in df_normalized.columns:
                if method == 'zscore':
                    mean = df_normalized[col].mean()
                    std = df_normalized[col].std()
                    df_normalized[col] = (df_normalized[col] - mean) / std
                elif method == 'minmax':
                    min_val = df_normalized[col].min()
                    max_val = df_normalized[col].max()
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        return df_normalized
    
    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """
        Get the processed data.
        
        Returns:
            Processed DataFrame or None if not available
        """
        return self.processed_data
    
    def save_processed_data(self, file_path: Union[str, Path], **kwargs) -> bool:
        """
        Save the processed data to a file with logging.
        
        Args:
            file_path: Path to save the processed data
            **kwargs: Additional arguments to pass to pandas save function
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        start_time = time.time()
        logger.info(f"💾 Starting to save processed data to: {file_path}")
        
        try:
            if self.processed_data is None:
                logger.error("❌ No processed data available")
                return False
                
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save based on file extension
            if file_path.suffix.lower() == '.csv':
                self.processed_data.to_csv(file_path, **kwargs)
                logger.info(f"📄 Saved as CSV: {file_path}")
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                self.processed_data.to_excel(file_path, **kwargs)
                logger.info(f"📊 Saved as Excel: {file_path}")
            elif file_path.suffix.lower() == '.parquet':
                self.processed_data.to_parquet(file_path, **kwargs)
                logger.info(f"📦 Saved as Parquet: {file_path}")
            else:
                logger.error(f"❌ Unsupported file format: {file_path.suffix}")
                return False
                
            save_time = time.time() - start_time
            logger.info(f"✅ Successfully saved processed data to {file_path} in {save_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving processed data: {str(e)}")
            return False
    
    def run_pipeline(self, file_path: Union[str, Path], **kwargs) -> bool:
        """
        Run the complete data pipeline with comprehensive logging.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments to pass to load_data
            
        Returns:
            bool: True if pipeline completed successfully, False otherwise
        """
        self.pipeline_start_time = datetime.now()
        logger.info(f"🚀 Starting data pipeline at {self.pipeline_start_time}")
        logger.info(f"📁 Input file: {file_path}")
        logger.info(f"⚙️ Configuration: {self.config}")
        
        try:
            # Load data
            logger.info("=" * 50)
            logger.info("STEP 1: Loading Data")
            logger.info("=" * 50)
            if not self.load_data(file_path, **kwargs):
                logger.error("❌ Pipeline failed at data loading step")
                return False
            
            # Validate data
            logger.info("=" * 50)
            logger.info("STEP 2: Validating Data")
            logger.info("=" * 50)
            is_valid, error_message = self.validate_data()
            if not is_valid:
                logger.error(f"❌ Pipeline failed at validation step: {error_message}")
                return False
            
            # Preprocess data
            logger.info("=" * 50)
            logger.info("STEP 3: Preprocessing Data")
            logger.info("=" * 50)
            if not self.preprocess_data():
                logger.error("❌ Pipeline failed at preprocessing step")
                return False
            
            self.pipeline_end_time = datetime.now()
            pipeline_duration = self.pipeline_end_time - self.pipeline_start_time
            
            logger.info("=" * 50)
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 50)
            logger.info(f"⏱️ Total duration: {pipeline_duration}")
            logger.info(f"📊 Final data shape: {self.processed_data.shape}")
            logger.info(f"📊 Final data columns: {list(self.processed_data.columns)}")
            logger.info("=" * 50)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed with error: {str(e)}")
            return False
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get pipeline execution statistics.
        
        Returns:
            Dictionary with pipeline statistics
        """
        stats = {
            "pipeline_start_time": self.pipeline_start_time.isoformat() if self.pipeline_start_time else None,
            "pipeline_end_time": self.pipeline_end_time.isoformat() if self.pipeline_end_time else None,
            "pipeline_duration": None,
            "data_shape": self.data.shape if self.data is not None else None,
            "processed_data_shape": self.processed_data.shape if self.processed_data is not None else None,
            "success": self.processed_data is not None
        }
        
        if self.pipeline_start_time and self.pipeline_end_time:
            stats["pipeline_duration"] = str(self.pipeline_end_time - self.pipeline_start_time)
        
        return stats


def run_data_pipeline(file_path: Union[str, Path], config: Optional[Dict] = None, **kwargs) -> Tuple[bool, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Convenience function to run the complete data pipeline.
    
    Args:
        file_path: Path to the data file
        config: Pipeline configuration
        **kwargs: Additional arguments for data loading
        
    Returns:
        Tuple of (success, processed_data, stats)
    """
    pipeline = DataPipeline(config)
    success = pipeline.run_pipeline(file_path, **kwargs)
    processed_data = pipeline.get_processed_data() if success else None
    stats = pipeline.get_pipeline_stats()
    
    return success, processed_data, stats