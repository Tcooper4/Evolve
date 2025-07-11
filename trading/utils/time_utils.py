"""Time utilities with timezone support and market hours.

This module provides utilities for handling time-related operations with
timezone support, market hours, and trading session management.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, time, timedelta
import pytz
import pandas as pd
import numpy as np
from dateutil import parser
import logging

logger = logging.getLogger(__name__)

class MarketHours:
    """Market hours configuration and validation."""
    
    def __init__(
        self,
        timezone: str = 'UTC',
        regular_start: time = time(9, 30),
        regular_end: time = time(16, 0),
        pre_market_start: Optional[time] = time(4, 0),
        pre_market_end: Optional[time] = time(9, 30),
        post_market_start: Optional[time] = time(16, 0),
        post_market_end: Optional[time] = time(20, 0)
    ):
        """Initialize market hours.
        
        Args:
            timezone: Market timezone
            regular_start: Regular market session start time
            regular_end: Regular market session end time
            pre_market_start: Pre-market session start time
            pre_market_end: Pre-market session end time
            post_market_start: Post-market session start time
            post_market_end: Post-market session end time
        """
        self.timezone = pytz.timezone(timezone)
        self.regular_start = regular_start
        self.regular_end = regular_end
        self.pre_market_start = pre_market_start
        self.pre_market_end = pre_market_end
        self.post_market_start = post_market_start
        self.post_market_end = post_market_end
    
    def is_market_open(self, dt: datetime) -> bool:
        """Check if market is open at given time.
        
        Args:
            dt: Datetime to check
            
        Returns:
            Whether market is open
        """
        dt = self._localize(dt)
        current_time = dt.time()
        
        # Check regular session
        if self.regular_start <= current_time <= self.regular_end:
            return True
        
        # Check pre-market
        if self.pre_market_start and self.pre_market_end:
            if self.pre_market_start <= current_time <= self.pre_market_end:
                return True
        
        # Check post-market
        if self.post_market_start and self.post_market_end:
            if self.post_market_start <= current_time <= self.post_market_end:
                return True
        
        return False
    
    def get_next_market_open(self, dt: datetime) -> datetime:
        """Get next market open time.
        
        Args:
            dt: Current datetime
            
        Returns:
            Next market open datetime
        """
        dt = self._localize(dt)
        current_time = dt.time()
        
        # If before pre-market
        if self.pre_market_start and current_time < self.pre_market_start:
            return self._combine_date_time(dt.date(), self.pre_market_start)
        
        # If between pre-market and regular
        if self.pre_market_end and self.regular_start:
            if self.pre_market_end <= current_time < self.regular_start:
                return self._combine_date_time(dt.date(), self.regular_start)
        
        # If between regular and post-market
        if self.regular_end and self.post_market_start:
            if self.regular_end <= current_time < self.post_market_start:
                return self._combine_date_time(dt.date(), self.post_market_start)
        
        # If after post-market or no post-market
        next_day = dt.date() + timedelta(days=1)
        if self.pre_market_start:
            return self._combine_date_time(next_day, self.pre_market_start)
        return self._combine_date_time(next_day, self.regular_start)
    
    def get_next_market_close(self, dt: datetime) -> datetime:
        """Get next market close time.
        
        Args:
            dt: Current datetime
            
        Returns:
            Next market close datetime
        """
        dt = self._localize(dt)
        current_time = dt.time()
        
        # If in pre-market
        if self.pre_market_start and self.pre_market_end:
            if self.pre_market_start <= current_time < self.pre_market_end:
                return self._combine_date_time(dt.date(), self.pre_market_end)
        
        # If in regular session
        if self.regular_start and self.regular_end:
            if self.regular_start <= current_time < self.regular_end:
                return self._combine_date_time(dt.date(), self.regular_end)
        
        # If in post-market
        if self.post_market_start and self.post_market_end:
            if self.post_market_start <= current_time < self.post_market_end:
                return self._combine_date_time(dt.date(), self.post_market_end)
        
        # If after all sessions
        next_day = dt.date() + timedelta(days=1)
        if self.pre_market_start:
            return self._combine_date_time(next_day, self.pre_market_end)
        return self._combine_date_time(next_day, self.regular_end)
    
    def _localize(self, dt: datetime) -> datetime:
        """Localize datetime to market timezone.
        
        Args:
            dt: Datetime to localize
            
        Returns:
            Localized datetime
        """
        if dt.tzinfo is None:
            return self.timezone.localize(dt)
        return dt.astimezone(self.timezone)
    
    def _combine_date_time(self, date: datetime.date, time: datetime.time) -> datetime:
        """Combine date and time in market timezone.
        
        Args:
            date: Date to combine
            time: Time to combine
            
        Returns:
            Combined datetime
        """
        return self.timezone.localize(datetime.combine(date, time))

class TimeUtils:
    """Utilities for time-related operations."""
    
    def __init__(self, market_hours: Optional[MarketHours] = None):
        """Initialize time utilities.
        
        Args:
            market_hours: Market hours configuration
        """
        self.market_hours = market_hours or MarketHours()
    
    def parse_datetime(
        self,
        dt_str: str,
        timezone: Optional[str] = None
    ) -> datetime:
        """Parse datetime string.
        
        Args:
            dt_str: Datetime string to parse
            timezone: Timezone to use
            
        Returns:
            Parsed datetime
        """
        dt = parser.parse(dt_str)
        if timezone:
            return dt.astimezone(pytz.timezone(timezone))
        return dt
    
    def to_timezone(
        self,
        dt: datetime,
        timezone: str
    ) -> datetime:
        """Convert datetime to timezone.
        
        Args:
            dt: Datetime to convert
            timezone: Target timezone
            
        Returns:
            Converted datetime
        """
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        return dt.astimezone(pytz.timezone(timezone))
    
    def format_datetime(
        self,
        dt: datetime,
        format_str: str = "%Y-%m-%d %H:%M:%S",
        timezone: Optional[str] = None,
        include_timezone: bool = True
    ) -> str:
        """Format datetime with timezone awareness.
        
        Args:
            dt: Datetime to format
            format_str: Format string
            timezone: Target timezone for formatting
            include_timezone: Whether to include timezone info
            
        Returns:
            Formatted datetime string
        """
        if timezone:
            dt = self.to_timezone(dt, timezone)
        
        formatted = dt.strftime(format_str)
        
        if include_timezone and dt.tzinfo:
            tz_name = dt.strftime('%Z')
            tz_offset = dt.strftime('%z')
            formatted += f" {tz_name} ({tz_offset})"
        
        return formatted
    
    def round_to_granularity(
        self,
        dt: datetime,
        granularity: str,
        timezone: Optional[str] = None
    ) -> datetime:
        """Round datetime to specified granularity.
        
        Args:
            dt: Datetime to round
            granularity: Granularity ('minute', 'hour', 'day', 'week', 'month')
            timezone: Timezone to use for rounding
            
        Returns:
            Rounded datetime
        """
        if timezone:
            dt = self.to_timezone(dt, timezone)
        
        if granularity == 'minute':
            return dt.replace(second=0, microsecond=0)
        elif granularity == 'hour':
            return dt.replace(minute=0, second=0, microsecond=0)
        elif granularity == 'day':
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif granularity == 'week':
            # Round to Monday of the week
            days_since_monday = dt.weekday()
            return dt.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
        elif granularity == 'month':
            return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            raise ValueError(f"Unsupported granularity: {granularity}")
    
    def get_logging_timestamp(
        self,
        dt: Optional[datetime] = None,
        timezone: str = 'UTC',
        include_milliseconds: bool = True,
        format_type: str = 'iso'
    ) -> str:
        """Get formatted timestamp for logging.
        
        Args:
            dt: Datetime to format (defaults to current time)
            timezone: Timezone for formatting
            include_milliseconds: Whether to include milliseconds
            format_type: Format type ('iso', 'readable', 'compact')
            
        Returns:
            Formatted timestamp string
        """
        if dt is None:
            dt = datetime.now(pytz.timezone(timezone))
        else:
            dt = self.to_timezone(dt, timezone)
        
        if format_type == 'iso':
            if include_milliseconds:
                return dt.isoformat()
            else:
                return dt.replace(microsecond=0).isoformat()
        elif format_type == 'readable':
            if include_milliseconds:
                return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + f" {dt.strftime('%Z')}"
            else:
                return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        elif format_type == 'compact':
            if include_milliseconds:
                return dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            else:
                return dt.strftime("%Y%m%d_%H%M%S")
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def get_time_granularity_for_logging(
        self,
        time_range: timedelta,
        max_points: int = 1000
    ) -> str:
        """Determine appropriate time granularity for logging based on time range.
        
        Args:
            time_range: Time range to analyze
            max_points: Maximum number of data points desired
            
        Returns:
            Recommended granularity string
        """
        total_seconds = time_range.total_seconds()
        
        if total_seconds <= 3600:  # 1 hour
            return 'minute'
        elif total_seconds <= 86400:  # 1 day
            return 'hour'
        elif total_seconds <= 604800:  # 1 week
            return 'day'
        elif total_seconds <= 2592000:  # 1 month
            return 'week'
        else:
            return 'month'
    
    def get_market_sessions(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """Get market sessions between dates.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of (session_start, session_end) tuples
        """
        sessions = []
        current_date = start_date.date()
        end_date = end_date.date()
        
        while current_date <= end_date:
            # Regular session
            if self.market_hours.regular_start and self.market_hours.regular_end:
                session_start = self.market_hours._combine_date_time(
                    current_date,
                    self.market_hours.regular_start
                )
                session_end = self.market_hours._combine_date_time(
                    current_date,
                    self.market_hours.regular_end
                )
                sessions.append((session_start, session_end))
            
            # Pre-market session
            if (self.market_hours.pre_market_start and
                self.market_hours.pre_market_end):
                session_start = self.market_hours._combine_date_time(
                    current_date,
                    self.market_hours.pre_market_start
                )
                session_end = self.market_hours._combine_date_time(
                    current_date,
                    self.market_hours.pre_market_end
                )
                sessions.append((session_start, session_end))
            
            # Post-market session
            if (self.market_hours.post_market_start and
                self.market_hours.post_market_end):
                session_start = self.market_hours._combine_date_time(
                    current_date,
                    self.market_hours.post_market_start
                )
                session_end = self.market_hours._combine_date_time(
                    current_date,
                    self.market_hours.post_market_end
                )
                sessions.append((session_start, session_end))
            
            current_date += timedelta(days=1)
        
        return sessions
    
    def resample_market_data(
        self,
        df: pd.DataFrame,
        freq: str,
        session: str = 'regular'
    ) -> pd.DataFrame:
        """Resample market data to specified frequency.
        
        Args:
            df: DataFrame with datetime index
            freq: Resampling frequency
            session: Market session to use
            
        Returns:
            Resampled DataFrame
        """
        if session == 'regular':
            # Filter to regular market hours
            mask = df.index.map(lambda x: self.market_hours.is_market_open(x))
            df_filtered = df[mask]
        else:
            df_filtered = df
        
        return df_filtered.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

    def get_timezone_info(self, timezone: str) -> Dict[str, Any]:
        """Get detailed timezone information.
        
        Args:
            timezone: Timezone name
            
        Returns:
            Dictionary with timezone information
        """
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        
        return {
            'name': timezone,
            'utc_offset': now.strftime('%z'),
            'dst_active': now.dst() != timedelta(0),
            'current_time': now.isoformat(),
            'abbreviation': now.strftime('%Z')
        }

    def validate_timezone(self, timezone: str) -> bool:
        """Validate if a timezone string is valid.
        
        Args:
            timezone: Timezone string to validate
            
        Returns:
            Whether timezone is valid
        """
        try:
            pytz.timezone(timezone)
            return True
        except pytz.exceptions.UnknownTimeZoneError:
            return False

    def get_common_timezones(self) -> List[str]:
        """Get list of common timezones for trading.
        
        Returns:
            List of common timezone names
        """
        return [
            'UTC',
            'America/New_York',
            'America/Chicago',
            'America/Denver',
            'America/Los_Angeles',
            'Europe/London',
            'Europe/Paris',
            'Europe/Berlin',
            'Asia/Tokyo',
            'Asia/Shanghai',
            'Asia/Hong_Kong',
            'Australia/Sydney'
        ]

# Global time utils instance
time_utils = TimeUtils()

def get_current_time(timezone: str = 'UTC') -> datetime:
    """Get current time in specified timezone.
    
    Args:
        timezone: Timezone for current time
        
    Returns:
        Current datetime in specified timezone
    """
    return time_utils.to_timezone(datetime.now(), timezone)

def format_timestamp(
    dt: datetime,
    format_str: str = "%Y-%m-%d %H:%M:%S",
    timezone: str = 'UTC'
) -> str:
    """Format timestamp with timezone awareness.
    
    Args:
        dt: Datetime to format
        format_str: Format string
        timezone: Target timezone
        
    Returns:
        Formatted timestamp string
    """
    return time_utils.format_datetime(dt, format_str, timezone)

def round_timestamp(
    dt: datetime,
    granularity: str,
    timezone: str = 'UTC'
) -> datetime:
    """Round timestamp to specified granularity.
    
    Args:
        dt: Datetime to round
        granularity: Granularity ('minute', 'hour', 'day', 'week', 'month')
        timezone: Timezone for rounding
        
    Returns:
        Rounded datetime
    """
    return time_utils.round_to_granularity(dt, granularity, timezone) 