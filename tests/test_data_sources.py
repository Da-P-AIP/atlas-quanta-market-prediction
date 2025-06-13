"""
Test cases for data source modules.
"""

import unittest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.data_sources.base_data_source import BaseDataSource

class TestDataSource(BaseDataSource):
    """Test implementation of BaseDataSource for testing."""
    
    def get_data(self, symbols, start_date, end_date, **kwargs):
        # Mock implementation
        dates = pd.date_range(start_date, end_date, freq='D')
        data = pd.DataFrame({
            'close': range(len(dates)),
            'volume': [1000] * len(dates)
        }, index=dates)
        return data
    
    def validate_symbols(self, symbols):
        return {symbol: True for symbol in symbols}

class TestBaseDataSource(unittest.TestCase):
    
    def setUp(self):
        self.config = {
            'max_history_days': 365,
            'test_param': 'test_value'
        }
        self.data_source = TestDataSource(self.config, 'test_source')
    
    def test_initialization(self):
        """Test proper initialization of data source."""
        self.assertEqual(self.data_source.name, 'test_source')
        self.assertEqual(self.data_source.config, self.config)
        self.assertIsNotNone(self.data_source.logger)
    
    def test_validate_date_range_valid(self):
        """Test date range validation with valid dates."""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now() - timedelta(days=1)
        
        result = self.data_source.validate_date_range(start_date, end_date)
        self.assertTrue(result)
    
    def test_validate_date_range_invalid_order(self):
        """Test date range validation with invalid date order."""
        start_date = datetime.now()
        end_date = datetime.now() - timedelta(days=1)
        
        with self.assertRaises(ValueError):
            self.data_source.validate_date_range(start_date, end_date)
    
    def test_preprocess_data(self):
        """Test data preprocessing functionality."""
        # Create test data with some issues
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        data = pd.DataFrame({
            'close': [100, 101, None, 103, 104, 105, 106, 107, 108, 109],
            'volume': [1000, 1100, 1200, None, 1400, 1500, 1600, 1700, 1800, 1900]
        }, index=dates)
        
        processed = self.data_source.preprocess_data(data)
        
        # Check that NaN values are forward filled
        self.assertFalse(processed.isnull().any().any())
        
        # Check that data is sorted
        self.assertTrue(processed.index.is_monotonic_increasing)
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        symbols = ['AAPL', 'MSFT']
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        cache_key = self.data_source.cache_key(symbols, start_date, end_date)
        
        expected = 'test_source_AAPL_MSFT_2023-01-01_2023-01-31'
        self.assertEqual(cache_key, expected)
    
    def test_data_quality_metrics(self):
        """Test data quality metrics calculation."""
        # Create test data
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }, index=dates)
        
        metrics = self.data_source.get_data_quality_metrics(data)
        
        # Should have 100% completeness (no missing values)
        self.assertEqual(metrics['completeness'], 100.0)
        
        # Should have 100% consistency (positive prices and volumes)
        self.assertEqual(metrics['consistency'], 100.0)
        
        # Freshness should be less than 100% unless data is from today
        self.assertGreaterEqual(metrics['freshness'], 0.0)
        self.assertLessEqual(metrics['freshness'], 100.0)

if __name__ == '__main__':
    unittest.main()