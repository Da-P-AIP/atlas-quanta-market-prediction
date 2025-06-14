"""
FRED (Federal Reserve Economic Data) Provider for Atlas Quanta

Provides access to over 800,000 economic time series from the
Federal Reserve Bank of St. Louis FRED database.
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging

from .base_data_source import BaseDataSource

class FREDProvider(BaseDataSource):
    """
    FRED API provider for economic data.
    
    Supports:
    - Economic indicators (GDP, inflation, unemployment, etc.)
    - Interest rates and yield curves
    - Money supply and monetary policy data
    - International economic data
    - Financial market indicators
    """
    
    def __init__(self, api_key: str, config: Dict = None):
        """
        Initialize FRED provider.
        
        Args:
            api_key: FRED API key (free from https://fred.stlouisfed.org/docs/api/api_key.html)
            config: Additional configuration
        """
        super().__init__(config)
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting - FRED allows 120 requests per 60 seconds
        self.requests_per_minute = 120
        self.last_request_time = 0
        self.request_interval = 60.0 / self.requests_per_minute
        
        # Common economic indicators
        self.key_indicators = {
            # GDP and Growth
            'GDP': 'GDP',  # Gross Domestic Product
            'GDPC1': 'GDPC1',  # Real GDP
            'GDPPOT': 'GDPPOT',  # Real Potential GDP
            
            # Inflation
            'CPIAUCSL': 'CPIAUCSL',  # Consumer Price Index
            'CPILFESL': 'CPILFESL',  # Core CPI
            'PCEPILFE': 'PCEPILFE',  # Core PCE Price Index
            
            # Employment
            'UNRATE': 'UNRATE',  # Unemployment Rate
            'PAYEMS': 'PAYEMS',  # Nonfarm Payrolls
            'CIVPART': 'CIVPART',  # Labor Force Participation Rate
            
            # Interest Rates
            'FEDFUNDS': 'FEDFUNDS',  # Federal Funds Rate
            'DGS10': 'DGS10',  # 10-Year Treasury
            'DGS2': 'DGS2',  # 2-Year Treasury
            'DGS3MO': 'DGS3MO',  # 3-Month Treasury
            
            # Money Supply
            'M1SL': 'M1SL',  # M1 Money Supply
            'M2SL': 'M2SL',  # M2 Money Supply
            'BOGMBASE': 'BOGMBASE',  # Monetary Base
            
            # Markets
            'VIXCLS': 'VIXCLS',  # VIX
            'DEXUSEU': 'DEXUSEU',  # USD/EUR Exchange Rate
            'GOLDAMGBD228NLBM': 'GOLDAMGBD228NLBM',  # Gold Price
            
            # Credit and Banking
            'TOTRESNS': 'TOTRESNS',  # Total Reserves
            'WALCL': 'WALCL',  # Fed Balance Sheet
            'DRTSCILM': 'DRTSCILM',  # Credit Conditions
        }
        
        self.logger.info("FRED provider initialized")
    
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
        
        params['api_key'] = self.api_key
        params['file_type'] = 'json'
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"FRED API request failed: {e}")
            raise
    
    def get_series(self, 
                  series_id: str,
                  start_date: Union[str, datetime] = None,
                  end_date: Union[str, datetime] = None,
                  frequency: str = None) -> pd.DataFrame:
        """
        Get time series data for a FRED series.
        
        Args:
            series_id: FRED series identifier (e.g., 'GDP', 'UNRATE')
            start_date: Start date (YYYY-MM-DD format or datetime)
            end_date: End date (YYYY-MM-DD format or datetime)
            frequency: Data frequency ('d', 'w', 'm', 'q', 'a')
            
        Returns:
            DataFrame with time series data
        """
        try:
            params = {}
            
            if start_date:
                if isinstance(start_date, datetime):
                    start_date = start_date.strftime('%Y-%m-%d')
                params['observation_start'] = start_date
            
            if end_date:
                if isinstance(end_date, datetime):
                    end_date = end_date.strftime('%Y-%m-%d')
                params['observation_end'] = end_date
            
            if frequency:
                params['frequency'] = frequency
            
            endpoint = f"series/observations"
            params['series_id'] = series_id
            
            data = self._make_request(endpoint, params)
            
            if 'observations' not in data:
                self.logger.warning(f"No observations found for series {series_id}")
                return pd.DataFrame()
            
            observations = data['observations']
            
            # Convert to DataFrame
            df = pd.DataFrame(observations)
            
            if df.empty:
                return df
            
            # Convert date and value columns
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Handle missing values (marked as '.')
            df['value'] = df['value'].replace('.', np.nan)
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Keep only the value column and add series info
            df = df[['value']].copy()
            df.columns = [series_id]
            df['series_id'] = series_id
            
            # Sort by date
            df.sort_index(inplace=True)
            
            self.logger.info(f"Retrieved {len(df)} observations for {series_id}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching FRED series {series_id}: {e}")
            return pd.DataFrame()
    
    def get_multiple_series(self, 
                           series_ids: List[str],
                           start_date: Union[str, datetime] = None,
                           end_date: Union[str, datetime] = None,
                           frequency: str = None) -> pd.DataFrame:
        """
        Get multiple time series and combine them into one DataFrame.
        
        Args:
            series_ids: List of FRED series identifiers
            start_date: Start date
            end_date: End date
            frequency: Data frequency
            
        Returns:
            DataFrame with all series
        """
        all_series = []
        
        for series_id in series_ids:
            try:
                df = self.get_series(series_id, start_date, end_date, frequency)
                if not df.empty:
                    # Keep only the series value column
                    series_data = df[[series_id]].copy()
                    all_series.append(series_data)
                else:
                    self.logger.warning(f"No data for series {series_id}")
            except Exception as e:
                self.logger.error(f"Error fetching series {series_id}: {e}")
        
        if not all_series:
            return pd.DataFrame()
        
        # Combine all series
        combined_df = pd.concat(all_series, axis=1, sort=True)
        
        self.logger.info(f"Combined {len(all_series)} series into single DataFrame")
        return combined_df
    
    def get_key_indicators(self, 
                          start_date: Union[str, datetime] = None,
                          end_date: Union[str, datetime] = None) -> pd.DataFrame:
        """
        Get key economic indicators predefined in the class.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with key economic indicators
        """
        series_ids = list(self.key_indicators.values())
        return self.get_multiple_series(series_ids, start_date, end_date)
    
    def get_yield_curve(self, 
                       date: Union[str, datetime] = None) -> pd.Series:
        """
        Get US Treasury yield curve data.
        
        Args:
            date: Specific date (default: latest available)
            
        Returns:
            Series with yield curve data
        """
        try:
            # Treasury yield series for different maturities
            yield_series = {
                '1M': 'DGS1MO',
                '3M': 'DGS3MO',
                '6M': 'DGS6MO',
                '1Y': 'DGS1',
                '2Y': 'DGS2',
                '3Y': 'DGS3',
                '5Y': 'DGS5',
                '7Y': 'DGS7',
                '10Y': 'DGS10',
                '20Y': 'DGS20',
                '30Y': 'DGS30'
            }
            
            if date:
                if isinstance(date, datetime):
                    date_str = date.strftime('%Y-%m-%d')
                else:
                    date_str = date
                
                # Get data for specific date
                end_date = date_str
                start_date = date_str
            else:
                # Get latest month of data
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Fetch all yield series
            yield_data = {}
            for maturity, series_id in yield_series.items():
                df = self.get_series(series_id, start_date, end_date)
                if not df.empty:
                    # Get the latest value
                    latest_value = df[series_id].dropna().iloc[-1] if not df[series_id].dropna().empty else np.nan
                    yield_data[maturity] = latest_value
            
            yield_curve = pd.Series(yield_data, name='yield')
            yield_curve.index.name = 'maturity'
            
            self.logger.info(f"Retrieved yield curve with {len(yield_curve)} maturities")
            return yield_curve
            
        except Exception as e:
            self.logger.error(f"Error fetching yield curve: {e}")
            return pd.Series()
    
    def search_series(self, search_text: str, limit: int = 20) -> pd.DataFrame:
        """
        Search for FRED series by text.
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results
            
        Returns:
            DataFrame with series information
        """
        try:
            endpoint = "series/search"
            params = {
                'search_text': search_text,
                'limit': limit,
                'order_by': 'popularity'
            }
            
            data = self._make_request(endpoint, params)
            
            if 'seriess' not in data:
                return pd.DataFrame()
            
            series_list = data['seriess']
            df = pd.DataFrame(series_list)
            
            # Keep relevant columns
            relevant_cols = ['id', 'title', 'frequency', 'units', 'popularity', 
                           'last_updated', 'observation_start', 'observation_end']
            available_cols = [col for col in relevant_cols if col in df.columns]
            df = df[available_cols]
            
            self.logger.info(f"Found {len(df)} series matching '{search_text}'")
            return df
            
        except Exception as e:
            self.logger.error(f"Error searching series: {e}")
            return pd.DataFrame()
    
    def get_series_info(self, series_id: str) -> Dict:
        """
        Get detailed information about a series.
        
        Args:
            series_id: FRED series identifier
            
        Returns:
            Dictionary with series metadata
        """
        try:
            endpoint = "series"
            params = {'series_id': series_id}
            
            data = self._make_request(endpoint, params)
            
            if 'seriess' in data and data['seriess']:
                return data['seriess'][0]
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error fetching series info for {series_id}: {e}")
            return {}
    
    def get_recession_indicators(self, 
                               start_date: Union[str, datetime] = None,
                               end_date: Union[str, datetime] = None) -> pd.DataFrame:
        """
        Get recession probability and indicators.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with recession indicators
        """
        recession_series = [
            'USREC',  # US Recession Indicator
            'USRECM',  # US Recession Probability
            'RECPROUSM156N',  # Smoothed Recession Probability
            'T10Y2Y',  # 10Y-2Y Treasury Spread
            'T10Y3M',  # 10Y-3M Treasury Spread
        ]
        
        return self.get_multiple_series(recession_series, start_date, end_date)
    
    def get_monetary_policy_indicators(self, 
                                     start_date: Union[str, datetime] = None,
                                     end_date: Union[str, datetime] = None) -> pd.DataFrame:
        """
        Get Federal Reserve monetary policy indicators.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with monetary policy data
        """
        monetary_series = [
            'FEDFUNDS',  # Federal Funds Rate
            'WALCL',  # Fed Balance Sheet
            'TOTRESNS',  # Total Reserves
            'BOGMBASE',  # Monetary Base
            'M1SL',  # M1 Money Supply
            'M2SL',  # M2 Money Supply
        ]
        
        return self.get_multiple_series(monetary_series, start_date, end_date)
    
    def calculate_real_rates(self, 
                           start_date: Union[str, datetime] = None,
                           end_date: Union[str, datetime] = None) -> pd.DataFrame:
        """
        Calculate real interest rates (nominal - inflation).
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with real rates
        """
        try:
            # Get nominal rates and inflation
            series_to_fetch = [
                'DGS10',  # 10-Year Treasury
                'DGS2',   # 2-Year Treasury
                'FEDFUNDS',  # Fed Funds Rate
                'CPILFESL',  # Core CPI
            ]
            
            df = self.get_multiple_series(series_to_fetch, start_date, end_date)
            
            if df.empty:
                return df
            
            # Calculate year-over-year inflation rate
            inflation_rate = df['CPILFESL'].pct_change(periods=12) * 100
            
            # Calculate real rates
            df['real_10y'] = df['DGS10'] - inflation_rate
            df['real_2y'] = df['DGS2'] - inflation_rate
            df['real_fed_funds'] = df['FEDFUNDS'] - inflation_rate
            
            # Keep only real rate columns
            real_rate_cols = ['real_10y', 'real_2y', 'real_fed_funds']
            result = df[real_rate_cols].copy()
            
            self.logger.info("Calculated real interest rates")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating real rates: {e}")
            return pd.DataFrame()
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            # Test with a simple series request
            self.get_series('GDP', start_date='2023-01-01', end_date='2023-01-01')
            self.logger.info("FRED API connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"FRED API connection test failed: {e}")
            return False
