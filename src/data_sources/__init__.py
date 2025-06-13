"""
Atlas Quanta Data Sources Module

This module provides unified interfaces for collecting financial data
from various sources including stocks, crypto, forex, and alternative data.
"""

from .base_data_source import BaseDataSource
from .stock_data import StockDataSource
from .crypto_data import CryptoDataSource
from .macro_data import MacroDataSource
from .sentiment_data import SentimentDataSource

__all__ = [
    'BaseDataSource',
    'StockDataSource', 
    'CryptoDataSource',
    'MacroDataSource',
    'SentimentDataSource'
]
