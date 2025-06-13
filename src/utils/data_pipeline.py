"""
Data pipeline module for orchestrating data loading, validation, preprocessing, and analysis.
"""
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
from .data_validation import DataValidator

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
        self.validator = DataValidator()
        self.data = None
        self.processed_data = None
        
    def load_data(self, file_path: Union[str, Path], **kwargs) -> bool:
        """
        Load data from a file.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments to pass to pandas read function
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
                
            # Determine file type and load accordingly
            if file_path.suffix.lower() == '.csv':
                self.data = pd.read_csv(file_path, **kwargs)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path, **kwargs)
            elif file_path.suffix.lower() == '.parquet':
                self.data = pd.read_parquet(file_path, **kwargs)
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return False
                
            logger.info(f"Successfully loaded data from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def validate_data(self) -> Tuple[bool, str]:
        """
        Validate the loaded data.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.data is None:
            return False, "No data loaded"
            
        return self.validator.validate_dataframe(self.data)
    
    def preprocess_data(self) -> bool:
        """
        Preprocess the loaded data.
        
        Returns:
            bool: True if preprocessing was successful, False otherwise
        """
        try:
            if self.data is None:
                logger.error("No data loaded")
                return False
                
            # Handle missing data
            self.data = self.validator.handle_missing_data(
                self.data,
                method=self.config.get('missing_data_method', 'ffill')
            )
            
            # Remove outliers if configured
            if self.config.get('remove_outliers', False):
                columns = self.config.get('outlier_columns', ['close', 'volume'])
                n_std = self.config.get('outlier_std', 3.0)
                self.data = self.validator.remove_outliers(self.data, columns, n_std)
            
            # Preprocess data
            self.processed_data = self.validator.preprocess_data(self.data)
            
            # Normalize data if configured
            if self.config.get('normalize', False):
                columns = self.config.get('normalize_columns', ['close', 'volume'])
                method = self.config.get('normalize_method', 'zscore')
                self.processed_data = self.validator.normalize_data(
                    self.processed_data,
                    columns,
                    method
                )
            
            logger.info("Successfully preprocessed data")
            return True
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return False
    
    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """
        Get the processed data.
        
        Returns:
            Processed DataFrame or None if not available
        """
        return self.processed_data
    
    def save_processed_data(self, file_path: Union[str, Path], **kwargs) -> bool:
        """
        Save the processed data to a file.
        
        Args:
            file_path: Path to save the processed data
            **kwargs: Additional arguments to pass to pandas save function
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        try:
            if self.processed_data is None:
                logger.error("No processed data available")
                return False
                
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save based on file extension
            if file_path.suffix.lower() == '.csv':
                self.processed_data.to_csv(file_path, **kwargs)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                self.processed_data.to_excel(file_path, **kwargs)
            elif file_path.suffix.lower() == '.parquet':
                self.processed_data.to_parquet(file_path, **kwargs)
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return False
                
            logger.info(f"Successfully saved processed data to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            return False
    
    def run_pipeline(self, file_path: Union[str, Path], **kwargs) -> bool:
        """
        Run the complete data pipeline.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments to pass to load_data
            
        Returns:
            bool: True if pipeline completed successfully, False otherwise
        """
        # Load data
        if not self.load_data(file_path, **kwargs):
            return False
            
        # Validate data
        is_valid, error_message = self.validate_data()
        if not is_valid:
            logger.error(f"Data validation failed: {error_message}")
            return False
            
        # Preprocess data
        if not self.preprocess_data():
            return False
            
        return True 