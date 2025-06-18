"""Time utilities with timezone support and market hours.

This module provides utilities for handling time-related operations with
timezone support, market hours, and trading session management.
"""

from typing import Dict, List, Optional, Union, Tuple
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
        
        return sorted(sessions)
    
    def resample_market_data(
        self,
        df: pd.DataFrame,
        freq: str,
        session: str = 'regular'
    ) -> pd.DataFrame:
        """Resample market data to specified frequency.
        
        Args:
            df: DataFrame to resample
            freq: Resampling frequency
            session: Market session to use
            
        Returns:
            Resampled DataFrame
        """
        if session == 'regular':
            start_time = self.market_hours.regular_start
            end_time = self.market_hours.regular_end
        elif session == 'pre':
            start_time = self.market_hours.pre_market_start
            end_time = self.market_hours.pre_market_end
        elif session == 'post':
            start_time = self.market_hours.post_market_start
            end_time = self.market_hours.post_market_end
        else:
            raise ValueError(f"Invalid session: {session}")
        
        if not start_time or not end_time:
            raise ValueError(f"Session {session} not configured")
        
        # Filter data to session
        mask = (df.index.time >= start_time) & (df.index.time <= end_time)
        df = df[mask]
        
        # Resample
        return df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

# Create singleton instance
time_utils = TimeUtils() 