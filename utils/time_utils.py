"""
Time utilities for datetime parsing and formatting.

This module provides centralized time formatting and datetime parsing logic
for use across the trading platform, particularly for Prophet forecasting.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def parse_datetime(
    date_input: Union[str, datetime, pd.Timestamp], 
    format_hint: Optional[str] = None
) -> datetime:
    """Parse datetime from various input formats.
    
    Args:
        date_input: Date input in various formats
        format_hint: Optional format hint for string parsing
        
    Returns:
        Parsed datetime object
        
    Raises:
        ValueError: If date cannot be parsed
    """
    try:
        if isinstance(date_input, datetime):
            return date_input
        elif isinstance(date_input, pd.Timestamp):
            return date_input.to_pydatetime()
        elif isinstance(date_input, str):
            if format_hint:
                return datetime.strptime(date_input, format_hint)
            else:
                # Try common formats
                formats = [
                    "%Y-%m-%d",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S.%f",
                    "%d/%m/%Y",
                    "%m/%d/%Y",
                    "%Y%m%d",
                ]
                
                for fmt in formats:
                    try:
                        return datetime.strptime(date_input, fmt)
                    except ValueError:
                        continue
                
                # Try pandas parsing as fallback
                return pd.to_datetime(date_input).to_pydatetime()
        else:
            raise ValueError(f"Unsupported date input type: {type(date_input)}")
            
    except Exception as e:
        raise ValueError(f"Failed to parse date '{date_input}': {e}")


def format_datetime_for_prophet(dt: datetime) -> str:
    """Format datetime for Prophet model input.
    
    Args:
        dt: Datetime to format
        
    Returns:
        Formatted datetime string
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def create_date_range(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    freq: str = "D"
) -> pd.DatetimeIndex:
    """Create a date range for forecasting.
    
    Args:
        start_date: Start date
        end_date: End date
        freq: Frequency string (D=daily, H=hourly, etc.)
        
    Returns:
        DatetimeIndex with the specified range
    """
    start = parse_datetime(start_date)
    end = parse_datetime(end_date)
    
    return pd.date_range(start=start, end=end, freq=freq)


def calculate_forecast_horizon(
    data: pd.DataFrame,
    date_column: str,
    target_horizon: Optional[int] = None,
    max_horizon: int = 365
) -> int:
    """Calculate appropriate forecast horizon based on data characteristics.
    
    Args:
        data: Input data DataFrame
        date_column: Name of the date column
        target_horizon: Target horizon (if None, auto-calculate)
        max_horizon: Maximum allowed horizon
        
    Returns:
        Calculated forecast horizon
    """
    try:
        if target_horizon is not None:
            return min(target_horizon, max_horizon)
        
        # Auto-calculate based on data characteristics
        if date_column not in data.columns:
            logger.warning(f"Date column '{date_column}' not found, using default horizon")
            return 30
        
        # Parse dates
        dates = pd.to_datetime(data[date_column])
        
        # Calculate data frequency
        if len(dates) < 2:
            return 30
        
        # Calculate time differences
        time_diffs = dates.diff().dropna()
        median_diff = time_diffs.median()
        
        # Determine frequency
        if median_diff <= timedelta(hours=1):
            freq = "hourly"
            default_horizon = 168  # 1 week
        elif median_diff <= timedelta(days=1):
            freq = "daily"
            default_horizon = 30   # 1 month
        elif median_diff <= timedelta(weeks=1):
            freq = "weekly"
            default_horizon = 12   # 3 months
        elif median_diff <= timedelta(days=30):
            freq = "monthly"
            default_horizon = 12   # 1 year
        else:
            freq = "yearly"
            default_horizon = 5    # 5 years
        
        # Calculate horizon based on data length
        data_length = len(data)
        if data_length < 50:
            horizon = min(default_horizon // 2, max_horizon)
        elif data_length < 200:
            horizon = min(default_horizon, max_horizon)
        else:
            horizon = min(default_horizon * 2, max_horizon)
        
        logger.info(f"Auto-calculated forecast horizon: {horizon} periods for {freq} data")
        return horizon
        
    except Exception as e:
        logger.warning(f"Error calculating forecast horizon: {e}, using default")
        return 30


def validate_date_column(data: pd.DataFrame, date_column: str) -> bool:
    """Validate that date column exists and contains valid dates.
    
    Args:
        data: Input data DataFrame
        date_column: Name of the date column
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if date_column not in data.columns:
            logger.error(f"Date column '{date_column}' not found in data")
            return False
        
        # Try to parse dates
        dates = pd.to_datetime(data[date_column], errors='coerce')
        invalid_count = dates.isnull().sum()
        
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid dates in column '{date_column}'")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating date column: {e}")
        return False


def get_holiday_dates(
    country: str = "US",
    start_year: int = 2020,
    end_year: int = 2030
) -> pd.DataFrame:
    """Get holiday dates for a specific country.
    
    Args:
        country: Country code for holidays
        start_year: Start year for holiday range
        end_year: End year for holiday range
        
    Returns:
        DataFrame with holiday dates and names
    """
    try:
        import holidays
        
        country_holidays = holidays.country_holidays(country, years=range(start_year, end_year + 1))
        
        holiday_data = []
        for date, name in country_holidays.items():
            holiday_data.append({
                'ds': date,
                'holiday': name
            })
        
        return pd.DataFrame(holiday_data)
        
    except ImportError:
        logger.warning("holidays package not available, returning empty DataFrame")
        return pd.DataFrame(columns=['ds', 'holiday'])
    except Exception as e:
        logger.error(f"Error getting holiday dates: {e}")
        return pd.DataFrame(columns=['ds', 'holiday'])


def add_time_features(data: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Add time-based features to DataFrame.
    
    Args:
        data: Input DataFrame
        date_column: Name of the date column
        
    Returns:
        DataFrame with added time features
    """
    try:
        df = data.copy()
        dates = pd.to_datetime(df[date_column])
        
        # Add time features
        df['year'] = dates.dt.year
        df['month'] = dates.dt.month
        df['day'] = dates.dt.day
        df['dayofweek'] = dates.dt.dayofweek
        df['quarter'] = dates.dt.quarter
        df['is_month_end'] = dates.dt.is_month_end.astype(int)
        df['is_month_start'] = dates.dt.is_month_start.astype(int)
        df['is_quarter_end'] = dates.dt.is_quarter_end.astype(int)
        df['is_quarter_start'] = dates.dt.is_quarter_start.astype(int)
        df['is_year_end'] = dates.dt.is_year_end.astype(int)
        df['is_year_start'] = dates.dt.is_year_start.astype(int)
        
        return df
        
    except Exception as e:
        logger.error(f"Error adding time features: {e}")
        return data


def detect_seasonality(data: pd.DataFrame, date_column: str, target_column: str) -> Dict[str, Any]:
    """Detect seasonality patterns in time series data.
    
    Args:
        data: Input DataFrame
        date_column: Name of the date column
        target_column: Name of the target column
        
    Returns:
        Dictionary with seasonality information
    """
    try:
        df = data.copy()
        
        # Calculate autocorrelation for different lags
        target_series = df[target_column]
        
        seasonality_info = {
            'daily': target_series.autocorr(lag=1),
            'weekly': target_series.autocorr(lag=7) if len(target_series) > 7 else None,
            'monthly': target_series.autocorr(lag=30) if len(target_series) > 30 else None,
            'quarterly': target_series.autocorr(lag=90) if len(target_series) > 90 else None,
            'yearly': target_series.autocorr(lag=365) if len(target_series) > 365 else None,
        }
        
        # Determine strongest seasonality
        valid_seasonalities = {k: v for k, v in seasonality_info.items() if v is not None}
        if valid_seasonalities:
            strongest = max(valid_seasonalities.items(), key=lambda x: abs(x[1]))
            seasonality_info['strongest'] = strongest
        
        return seasonality_info
        
    except Exception as e:
        logger.error(f"Error detecting seasonality: {e}")
        return {} 


def normalize_datetime_index(df: pd.DataFrame, enforce_utc: bool = False) -> pd.DataFrame:
    """
    Normalize DataFrame index to remove timezone info or enforce UTC.
    If enforce_utc is True, convert to UTC. Otherwise, remove tz info.
    """
    idx = df.index
    if hasattr(idx, 'tz') and idx.tz is not None:
        if enforce_utc:
            df.index = idx.tz_convert('UTC')
        else:
            df.index = idx.tz_localize(None)
    return df 