"""
EOD Historical Data Provider for Atlas Quanta

Provides access to end-of-day historical stock data, forex, crypto,
and fundamental data through EOD Historical Data API.
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging

from .base_data_source import BaseDataSource

class EODHistoricalDataProvider(BaseDataSource):
    """
    EOD Historical Data API provider.
    
    Supports:
    - Stock prices (US, Europe, Asia)
    - Forex rates
    - Cryptocurrency prices
    - Fundamental data
    - Economic indicators
    """
    
    def __init__(self, api_key: str, config: Dict = None):
        """
        Initialize EOD Historical Data provider.
        
        Args:
            api_key: EOD Historical Data API key
            config: Additional configuration
        """
        super().__init__(config)
        self.api_key = api_key
        self.base_url = "https://eodhistoricaldata.com/api"
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.requests_per_minute = 1000  # EOD allows 1000 requests/minute
        self.last_request_time = 0
        self.request_interval = 60.0 / self.requests_per_minute
        
        self.logger.info("EOD Historical Data provider initialized")
    
    def _rate_limit(self):
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_interval:
            sleep_time = self.request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with rate limiting."""
        self._rate_limit()
        
        if params is None:
            params = {}
        
        params['api_token'] = self.api_key
        params['fmt'] = 'json'
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"EOD API request failed: {e}")
            raise
    
    def get_historical_data(self, 
                           symbol: str,
                           start_date: Union[str, datetime],
                           end_date: Union[str, datetime] = None,
                           exchange: str = 'US') -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            start_date: Start date
            end_date: End date (default: today)
            exchange: Exchange code (US, TO, L, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if end_date is None:
                end_date = datetime.now()
            
            # Format dates
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            
            # Format symbol with exchange
            if '.' not in symbol and exchange:
                full_symbol = f"{symbol}.{exchange}"
            else:
                full_symbol = symbol
            
            endpoint = f"eod/{full_symbol}"
            params = {
                'from': start_date,
                'to': end_date,
                'period': 'd'  # Daily data
            }
            
            data = self._make_request(endpoint, params)
            
            if not data:
                self.logger.warning(f"No data returned for {full_symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            df['symbol'] = symbol
            
            # Sort by date
            df.sort_index(inplace=True)
            
            self.logger.info(f"Retrieved {len(df)} records for {full_symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_forex_data(self, 
                      currency_pair: str,
                      start_date: Union[str, datetime],
                      end_date: Union[str, datetime] = None) -> pd.DataFrame:
        """
        Get forex exchange rates.
        
        Args:
            currency_pair: Currency pair (e.g., 'EURUSD', 'GBPJPY')
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with forex rates
        """
        try:
            # Add .FOREX suffix
            if not currency_pair.endswith('.FOREX'):
                currency_pair = f"{currency_pair}.FOREX"
            
            return self.get_historical_data(currency_pair, start_date, end_date, exchange='')
            
        except Exception as e:
            self.logger.error(f"Error fetching forex data for {currency_pair}: {e}")
            return pd.DataFrame()
    
    def get_crypto_data(self, 
                       crypto_symbol: str,
                       start_date: Union[str, datetime],
                       end_date: Union[str, datetime] = None) -> pd.DataFrame:
        """
        Get cryptocurrency price data.
        
        Args:
            crypto_symbol: Crypto symbol (e.g., 'BTC-USD', 'ETH-USD')
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with crypto prices
        """
        try:
            # Add .CC suffix for crypto
            if not crypto_symbol.endswith('.CC'):
                crypto_symbol = f"{crypto_symbol}.CC"
            
            return self.get_historical_data(crypto_symbol, start_date, end_date, exchange='')
            
        except Exception as e:
            self.logger.error(f"Error fetching crypto data for {crypto_symbol}: {e}")
            return pd.DataFrame()
    
    def get_fundamental_data(self, symbol: str, exchange: str = 'US') -> Dict:
        """
        Get fundamental data for a symbol.
        
        Args:
            symbol: Stock symbol
            exchange: Exchange code
            
        Returns:
            Dictionary with fundamental data
        """
        try:
            if '.' not in symbol and exchange:
                full_symbol = f"{symbol}.{exchange}"
            else:
                full_symbol = symbol
            
            endpoint = f"fundamentals/{full_symbol}"
            data = self._make_request(endpoint)
            
            self.logger.info(f"Retrieved fundamental data for {full_symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching fundamental data for {symbol}: {e}")
            return {}
    
    def get_economic_events(self, 
                           country: str = 'US',
                           date_from: Union[str, datetime] = None,
                           date_to: Union[str, datetime] = None) -> pd.DataFrame:
        """
        Get economic calendar events.
        
        Args:
            country: Country code (US, GB, EU, etc.)
            date_from: Start date
            date_to: End date
            
        Returns:
            DataFrame with economic events
        """
        try:
            endpoint = "economic-events"
            params = {}
            
            if country:
                params['country'] = country
            if date_from:
                if isinstance(date_from, datetime):
                    date_from = date_from.strftime('%Y-%m-%d')
                params['from'] = date_from
            if date_to:
                if isinstance(date_to, datetime):
                    date_to = date_to.strftime('%Y-%m-%d')
                params['to'] = date_to
            
            data = self._make_request(endpoint, params)
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            self.logger.info(f"Retrieved {len(df)} economic events for {country}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching economic events: {e}")
            return pd.DataFrame()
    
    def get_multiple_symbols(self, 
                            symbols: List[str],
                            start_date: Union[str, datetime],
                            end_date: Union[str, datetime] = None,
                            exchange: str = 'US') -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols efficiently.
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            exchange: Exchange code
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            try:
                df = self.get_historical_data(symbol, start_date, end_date, exchange)
                if not df.empty:
                    results[symbol] = df
                else:
                    self.logger.warning(f"No data for {symbol}")
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
        
        self.logger.info(f"Retrieved data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def get_dividends(self, 
                     symbol: str,
                     start_date: Union[str, datetime],
                     end_date: Union[str, datetime] = None,
                     exchange: str = 'US') -> pd.DataFrame:
        """
        Get dividend data for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            exchange: Exchange code
            
        Returns:
            DataFrame with dividend data
        """
        try:
            if end_date is None:
                end_date = datetime.now()
            
            # Format dates
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            
            if '.' not in symbol and exchange:
                full_symbol = f"{symbol}.{exchange}"
            else:
                full_symbol = symbol
            
            endpoint = f"div/{full_symbol}"
            params = {
                'from': start_date,
                'to': end_date
            }
            
            data = self._make_request(endpoint, params)
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df['symbol'] = symbol
            
            self.logger.info(f"Retrieved {len(df)} dividend records for {full_symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching dividend data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_splits(self, 
                  symbol: str,
                  start_date: Union[str, datetime],
                  end_date: Union[str, datetime] = None,
                  exchange: str = 'US') -> pd.DataFrame:
        """
        Get stock split data.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            exchange: Exchange code
            
        Returns:
            DataFrame with split data
        """
        try:
            if end_date is None:
                end_date = datetime.now()
            
            # Format dates
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            
            if '.' not in symbol and exchange:
                full_symbol = f"{symbol}.{exchange}"
            else:
                full_symbol = symbol
            
            endpoint = f"splits/{full_symbol}"
            params = {
                'from': start_date,
                'to': end_date
            }
            
            data = self._make_request(endpoint, params)
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df['symbol'] = symbol
            
            self.logger.info(f"Retrieved {len(df)} split records for {full_symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching split data for {symbol}: {e}")
            return pd.DataFrame()
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            # Test with a simple request
            endpoint = "eod/AAPL.US"
            params = {
                'from': '2023-01-01',
                'to': '2023-01-02'
            }
            self._make_request(endpoint, params)
            self.logger.info("EOD API connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"EOD API connection test failed: {e}")
            return False
