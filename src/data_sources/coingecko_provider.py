"""
CoinGecko API Provider for Atlas Quanta

Provides access to cryptocurrency market data, including prices,
volume, market cap, and on-chain indicators from CoinGecko API.
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging

from .base_data_source import BaseDataSource

class CoinGeckoProvider(BaseDataSource):
    """
    CoinGecko API provider for cryptocurrency data.
    
    Supports:
    - Real-time and historical price data
    - Market cap and volume data
    - Market indicators (Fear & Greed Index, etc.)
    - Global cryptocurrency market statistics
    - Exchange data
    - DeFi protocols data
    """
    
    def __init__(self, api_key: str = None, config: Dict = None):
        """
        Initialize CoinGecko provider.
        
        Args:
            api_key: CoinGecko Pro API key (optional for free tier)
            config: Additional configuration
        """
        super().__init__(config)
        self.api_key = api_key
        
        # Use Pro API if key is provided, otherwise free API
        if api_key:
            self.base_url = "https://pro-api.coingecko.com/api/v3"
            self.requests_per_minute = 500  # Pro tier limit
        else:
            self.base_url = "https://api.coingecko.com/api/v3"
            self.requests_per_minute = 10   # Free tier limit
        
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.last_request_time = 0
        self.request_interval = 60.0 / self.requests_per_minute
        
        # Common cryptocurrency IDs
        self.major_cryptos = {
            'bitcoin': 'bitcoin',
            'ethereum': 'ethereum',
            'binancecoin': 'binancecoin',
            'cardano': 'cardano',
            'solana': 'solana',
            'polkadot': 'polkadot',
            'dogecoin': 'dogecoin',
            'avalanche-2': 'avalanche-2',
            'polygon': 'matic-network',
            'chainlink': 'chainlink'
        }
        
        self.logger.info("CoinGecko provider initialized")
    
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
        
        # Add API key for Pro users
        if self.api_key:
            params['x_cg_pro_api_key'] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"CoinGecko API request failed: {e}")
            raise
    
    def get_price_data(self, 
                      crypto_ids: Union[str, List[str]],
                      vs_currencies: Union[str, List[str]] = 'usd',
                      include_market_cap: bool = True,
                      include_24hr_vol: bool = True,
                      include_24hr_change: bool = True) -> pd.DataFrame:
        """
        Get current price data for cryptocurrencies.
        
        Args:
            crypto_ids: Crypto ID or list of IDs (e.g., 'bitcoin', 'ethereum')
            vs_currencies: Currency or currencies to price against
            include_market_cap: Include market cap data
            include_24hr_vol: Include 24h volume data
            include_24hr_change: Include 24h price change data
            
        Returns:
            DataFrame with current price data
        """
        try:
            if isinstance(crypto_ids, str):
                crypto_ids = [crypto_ids]
            if isinstance(vs_currencies, str):
                vs_currencies = [vs_currencies]
            
            params = {
                'ids': ','.join(crypto_ids),
                'vs_currencies': ','.join(vs_currencies),
                'include_market_cap': str(include_market_cap).lower(),
                'include_24hr_vol': str(include_24hr_vol).lower(),
                'include_24hr_change': str(include_24hr_change).lower()
            }
            
            data = self._make_request('simple/price', params)
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            rows = []
            for crypto_id, price_data in data.items():
                for currency, price in price_data.items():
                    if not currency.endswith('_market_cap') and not currency.endswith('_24h_vol') and not currency.endswith('_24h_change'):
                        row = {
                            'crypto_id': crypto_id,
                            'currency': currency,
                            'price': price,
                            'timestamp': datetime.now()
                        }
                        
                        # Add optional data if available
                        if f"{currency}_market_cap" in price_data:
                            row['market_cap'] = price_data[f"{currency}_market_cap"]
                        if f"{currency}_24h_vol" in price_data:
                            row['volume_24h'] = price_data[f"{currency}_24h_vol"]
                        if f"{currency}_24h_change" in price_data:
                            row['change_24h'] = price_data[f"{currency}_24h_change"]
                        
                        rows.append(row)
            
            df = pd.DataFrame(rows)
            if not df.empty:
                df.set_index(['crypto_id', 'currency'], inplace=True)
            
            self.logger.info(f"Retrieved price data for {len(crypto_ids)} cryptocurrencies")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching price data: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, 
                           crypto_id: str,
                           vs_currency: str = 'usd',
                           days: int = 30,
                           interval: str = 'daily') -> pd.DataFrame:
        """
        Get historical price data for a cryptocurrency.
        
        Args:
            crypto_id: CoinGecko cryptocurrency ID
            vs_currency: Currency to price against
            days: Number of days of history (1-max)
            interval: Data interval ('daily' for >90 days, 'hourly' for <=90 days)
            
        Returns:
            DataFrame with historical OHLC data
        """
        try:
            endpoint = f"coins/{crypto_id}/ohlc"
            params = {
                'vs_currency': vs_currency,
                'days': days
            }
            
            data = self._make_request(endpoint, params)
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add crypto info
            df['crypto_id'] = crypto_id
            df['vs_currency'] = vs_currency
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            self.logger.info(f"Retrieved {len(df)} historical records for {crypto_id}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {crypto_id}: {e}")
            return pd.DataFrame()
    
    def get_market_data(self, 
                       crypto_id: str,
                       days: int = 30) -> pd.DataFrame:
        """
        Get comprehensive market data including price, volume, and market cap.
        
        Args:
            crypto_id: CoinGecko cryptocurrency ID
            days: Number of days of history
            
        Returns:
            DataFrame with market data
        """
        try:
            endpoint = f"coins/{crypto_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily' if days > 90 else 'hourly'
            }
            
            data = self._make_request(endpoint, params)
            
            if not data:
                return pd.DataFrame()
            
            # Extract price, volume, and market cap data
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
            
            # Convert timestamps
            for df_temp in [prices, volumes, market_caps]:
                df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'], unit='ms')
                df_temp.set_index('timestamp', inplace=True)
            
            # Combine all data
            df = pd.concat([prices, volumes, market_caps], axis=1)
            df['crypto_id'] = crypto_id
            
            # Calculate additional metrics
            df['returns'] = df['price'].pct_change()
            df['volatility'] = df['returns'].rolling(window=7).std()
            df['volume_ma'] = df['volume'].rolling(window=7).mean()
            
            self.logger.info(f"Retrieved market data for {crypto_id}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching market data for {crypto_id}: {e}")
            return pd.DataFrame()
    
    def get_fear_greed_index(self, limit: int = 30) -> pd.DataFrame:
        """
        Get cryptocurrency Fear & Greed Index data.
        
        Args:
            limit: Number of historical data points to retrieve
            
        Returns:
            DataFrame with Fear & Greed Index data
        """
        try:
            # Note: CoinGecko doesn't have native F&G index, we'll use Alternative.me API
            url = "https://api.alternative.me/fng/"
            params = {'limit': limit, 'format': 'json'}
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            # Convert value to numeric
            df['value'] = pd.to_numeric(df['value'])
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            self.logger.info(f"Retrieved {len(df)} Fear & Greed Index records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching Fear & Greed Index: {e}")
            return pd.DataFrame()
    
    def get_global_data(self) -> Dict:
        """
        Get global cryptocurrency market statistics.
        
        Returns:
            Dictionary with global market data
        """
        try:
            data = self._make_request('global')
            
            if 'data' in data:
                global_data = data['data']
                
                # Extract key metrics
                result = {
                    'total_market_cap_usd': global_data.get('total_market_cap', {}).get('usd'),
                    'total_volume_24h_usd': global_data.get('total_volume', {}).get('usd'),
                    'market_cap_percentage_btc': global_data.get('market_cap_percentage', {}).get('btc'),
                    'market_cap_percentage_eth': global_data.get('market_cap_percentage', {}).get('eth'),
                    'active_cryptocurrencies': global_data.get('active_cryptocurrencies'),
                    'markets': global_data.get('markets'),
                    'market_cap_change_24h': global_data.get('market_cap_change_percentage_24h_usd'),
                    'updated_at': datetime.now()
                }
                
                self.logger.info("Retrieved global cryptocurrency market data")
                return result
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error fetching global data: {e}")
            return {}
    
    def get_trending_coins(self) -> pd.DataFrame:
        """
        Get trending cryptocurrencies.
        
        Returns:
            DataFrame with trending coins
        """
        try:
            data = self._make_request('search/trending')
            
            if 'coins' not in data:
                return pd.DataFrame()
            
            trending_coins = []
            for coin_data in data['coins']:
                coin = coin_data['item']
                trending_coins.append({
                    'id': coin.get('id'),
                    'symbol': coin.get('symbol'),
                    'name': coin.get('name'),
                    'market_cap_rank': coin.get('market_cap_rank'),
                    'score': coin.get('score'),
                    'large_image': coin.get('large'),
                    'retrieved_at': datetime.now()
                })
            
            df = pd.DataFrame(trending_coins)
            
            self.logger.info(f"Retrieved {len(df)} trending coins")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching trending coins: {e}")
            return pd.DataFrame()
    
    def get_top_coins(self, 
                     vs_currency: str = 'usd',
                     order: str = 'market_cap_desc',
                     per_page: int = 100,
                     page: int = 1) -> pd.DataFrame:
        """
        Get top cryptocurrencies by market cap.
        
        Args:
            vs_currency: Currency to price against
            order: Sorting order
            per_page: Results per page (max 250)
            page: Page number
            
        Returns:
            DataFrame with top coins data
        """
        try:
            params = {
                'vs_currency': vs_currency,
                'order': order,
                'per_page': min(per_page, 250),
                'page': page,
                'sparkline': 'false',
                'price_change_percentage': '1h,24h,7d,14d,30d'
            }
            
            data = self._make_request('coins/markets', params)
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # Add timestamp
            df['retrieved_at'] = datetime.now()
            
            # Set index
            if 'id' in df.columns:
                df.set_index('id', inplace=True)
            
            self.logger.info(f"Retrieved data for {len(df)} top coins")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching top coins: {e}")
            return pd.DataFrame()
    
    def get_coin_info(self, crypto_id: str) -> Dict:
        """
        Get detailed information about a specific cryptocurrency.
        
        Args:
            crypto_id: CoinGecko cryptocurrency ID
            
        Returns:
            Dictionary with detailed coin information
        """
        try:
            endpoint = f"coins/{crypto_id}"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'true',
                'developer_data': 'true'
            }
            
            data = self._make_request(endpoint, params)
            
            self.logger.info(f"Retrieved detailed info for {crypto_id}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching coin info for {crypto_id}: {e}")
            return {}
    
    def get_exchange_data(self, exchange_id: str = None) -> pd.DataFrame:
        """
        Get cryptocurrency exchange data.
        
        Args:
            exchange_id: Specific exchange ID (if None, returns all exchanges)
            
        Returns:
            DataFrame with exchange data
        """
        try:
            if exchange_id:
                endpoint = f"exchanges/{exchange_id}"
                data = self._make_request(endpoint)
                # Convert single exchange to list format
                data = [data] if data else []
            else:
                endpoint = "exchanges"
                params = {'per_page': 100}
                data = self._make_request(endpoint, params)
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['retrieved_at'] = datetime.now()
            
            if 'id' in df.columns:
                df.set_index('id', inplace=True)
            
            self.logger.info(f"Retrieved data for {len(df)} exchanges")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching exchange data: {e}")
            return pd.DataFrame()
    
    def get_multiple_cryptos_data(self, 
                                 crypto_ids: List[str],
                                 days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Get market data for multiple cryptocurrencies.
        
        Args:
            crypto_ids: List of cryptocurrency IDs
            days: Number of days of history
            
        Returns:
            Dictionary mapping crypto IDs to DataFrames
        """
        results = {}
        
        for crypto_id in crypto_ids:
            try:
                df = self.get_market_data(crypto_id, days)
                if not df.empty:
                    results[crypto_id] = df
                else:
                    self.logger.warning(f"No data for {crypto_id}")
            except Exception as e:
                self.logger.error(f"Error fetching data for {crypto_id}: {e}")
        
        self.logger.info(f"Retrieved data for {len(results)}/{len(crypto_ids)} cryptocurrencies")
        return results
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            # Test with a simple ping request
            self._make_request('ping')
            self.logger.info("CoinGecko API connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"CoinGecko API connection test failed: {e}")
            return False
