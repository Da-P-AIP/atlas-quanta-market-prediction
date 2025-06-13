"""
Base Data Source Class

Abstract base class for all data sources in Atlas Quanta system.
Provides common interface and utilities for data collection.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import logging
from datetime import datetime, timedelta

class BaseDataSource(ABC):
    """
    Abstract base class for all data sources.
    
    All data source implementations should inherit from this class
    and implement the required abstract methods.
    """
    
    def __init__(self, config: Dict[str, Any], name: str):
        """
        Initialize the data source.
        
        Args:
            config: Configuration dictionary
            name: Name of the data source
        """
        self.config = config
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the data source."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    def get_data(self, 
                 symbols: List[str], 
                 start_date: datetime, 
                 end_date: datetime,
                 **kwargs) -> pd.DataFrame:
        """
        Fetch data for specified symbols and date range.
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date for data
            end_date: End date for data
            **kwargs: Additional parameters specific to data source
            
        Returns:
            DataFrame with fetched data
        """
        pass
    
    @abstractmethod
    def validate_symbols(self, symbols: List[str]) -> Dict[str, bool]:
        """
        Validate if symbols are available from this data source.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            Dictionary mapping symbol to availability status
        """
        pass
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of all available symbols from this data source.
        
        Returns:
            List of available symbols
        """
        # Default implementation - should be overridden by subclasses
        return []
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply common preprocessing to the data.
        
        Args:
            data: Raw data from source
            
        Returns:
            Preprocessed data
        """
        if data.empty:
            return data
            
        # Common preprocessing steps
        try:
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'timestamp' in data.columns:
                    data.set_index('timestamp', inplace=True)
                elif 'date' in data.columns:
                    data.set_index('date', inplace=True)
                    
            # Sort by index
            data = data.sort_index()
            
            # Remove duplicates
            data = data[~data.index.duplicated(keep='last')]
            
            # Forward fill NaN values (conservative approach)
            data = data.fillna(method='ffill')
            
            self.logger.info(f"Preprocessed data: {len(data)} records")
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
            
        return data
    
    def cache_key(self, symbols: List[str], start_date: datetime, end_date: datetime) -> str:
        """
        Generate cache key for data request.
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Cache key string
        """
        symbols_str = "_".join(sorted(symbols))
        return f"{self.name}_{symbols_str}_{start_date.date()}_{end_date.date()}"
    
    def handle_rate_limit(self, retry_count: int = 0, max_retries: int = 3):
        """
        Handle rate limiting with exponential backoff.
        
        Args:
            retry_count: Current retry attempt
            max_retries: Maximum number of retries
        """
        if retry_count >= max_retries:
            raise Exception(f"Max retries ({max_retries}) exceeded for {self.name}")
            
        import time
        wait_time = 2 ** retry_count  # Exponential backoff
        self.logger.warning(f"Rate limited. Waiting {wait_time} seconds before retry {retry_count + 1}")
        time.sleep(wait_time)
    
    def validate_date_range(self, start_date: datetime, end_date: datetime) -> bool:
        """
        Validate if date range is acceptable for this data source.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            True if date range is valid
        """
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
            
        # Check if start_date is not too far in the past
        max_history = self.config.get('max_history_days', 365 * 10)  # 10 years default
        if (datetime.now() - start_date).days > max_history:
            self.logger.warning(f"Requested start date {start_date} exceeds max history of {max_history} days")
            return False
            
        # Check if end_date is not in the future
        if end_date > datetime.now() + timedelta(days=1):
            self.logger.warning(f"Requested end date {end_date} is in the future")
            return False
            
        return True
    
    def get_data_quality_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate data quality metrics for the fetched data.
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        if data.empty:
            return {'completeness': 0.0, 'consistency': 0.0, 'freshness': 0.0}
            
        metrics = {}
        
        # Completeness: percentage of non-null values
        total_cells = data.size
        non_null_cells = data.count().sum()
        metrics['completeness'] = (non_null_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # Consistency: check for obvious anomalies (e.g., negative prices)
        consistency_checks = 0
        total_checks = 0
        
        if 'close' in data.columns:
            total_checks += 1
            if (data['close'] > 0).all():
                consistency_checks += 1
                
        if 'volume' in data.columns:
            total_checks += 1
            if (data['volume'] >= 0).all():
                consistency_checks += 1
                
        metrics['consistency'] = (consistency_checks / total_checks) * 100 if total_checks > 0 else 100
        
        # Freshness: how recent is the latest data
        if not data.empty:
            latest_date = data.index.max()
            days_old = (datetime.now() - latest_date).days
            metrics['freshness'] = max(0, 100 - days_old)  # 100% if today, decreases by 1% per day
        else:
            metrics['freshness'] = 0
            
        return metrics
