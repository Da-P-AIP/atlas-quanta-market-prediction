"""
Main Sentiment Analyzer for Atlas Quanta

Combines multiple sentiment sources and creates unified sentiment indicators
for use in market prediction models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

class SentimentAnalyzer:
    """
    Main sentiment analysis engine that combines multiple sentiment sources.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.sentiment_sources = {}
        self.weights = config.get('sentiment_weights', {
            'social': 0.3,
            'news': 0.4, 
            'fear_greed': 0.2,
            'vix': 0.1
        })
        
    def add_sentiment_source(self, name: str, source):
        """Add a sentiment data source."""
        self.sentiment_sources[name] = source
        self.logger.info(f"Added sentiment source: {name}")
    
    def get_composite_sentiment(self, 
                               symbols: List[str],
                               start_date: datetime,
                               end_date: datetime) -> pd.DataFrame:
        """
        Calculate composite sentiment score from multiple sources.
        
        Args:
            symbols: List of symbols to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            DataFrame with composite sentiment scores
        """
        
        sentiment_data = {}
        
        # Collect data from all sources
        for source_name, source in self.sentiment_sources.items():
            try:
                data = source.get_sentiment(symbols, start_date, end_date)
                if not data.empty:
                    sentiment_data[source_name] = data
                    self.logger.info(f"Collected {len(data)} records from {source_name}")
            except Exception as e:
                self.logger.error(f"Error collecting from {source_name}: {e}")
        
        if not sentiment_data:
            self.logger.warning("No sentiment data collected")
            return pd.DataFrame()
        
        # Combine sentiments with weights
        composite_df = self._combine_sentiments(sentiment_data)
        
        # Apply smoothing and normalization
        composite_df = self._smooth_sentiment(composite_df)
        
        return composite_df
    
    def _combine_sentiments(self, sentiment_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple sentiment sources with weighted average.
        """
        
        # Create unified time index
        all_dates = set()
        for data in sentiment_data.values():
            all_dates.update(data.index)
        
        date_index = pd.DatetimeIndex(sorted(all_dates))
        
        # Initialize result DataFrame
        result = pd.DataFrame(index=date_index)
        
        # For each symbol, combine sentiments
        all_symbols = set()
        for data in sentiment_data.values():
            all_symbols.update(data.columns)
        
        for symbol in all_symbols:
            weighted_scores = []
            total_weight = 0
            
            for source_name, data in sentiment_data.items():
                if symbol in data.columns:
                    weight = self.weights.get(source_name, 0.1)
                    # Align to common index
                    aligned_data = data[symbol].reindex(date_index)
                    weighted_scores.append(aligned_data * weight)
                    total_weight += weight
            
            if weighted_scores and total_weight > 0:
                # Combine weighted scores
                combined = sum(weighted_scores) / total_weight
                result[f'{symbol}_sentiment'] = combined
                
                # Calculate sentiment momentum
                result[f'{symbol}_sentiment_momentum'] = combined.diff(5)  # 5-day momentum
                
                # Calculate sentiment extremes (contrarian indicators)
                rolling_std = combined.rolling(30).std()
                rolling_mean = combined.rolling(30).mean()
                result[f'{symbol}_sentiment_zscore'] = (combined - rolling_mean) / rolling_std
        
        return result
    
    def _smooth_sentiment(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply smoothing to reduce noise in sentiment data.
        """
        
        smoothing_window = self.config.get('smoothing_window', 5)
        
        # Apply exponential weighted moving average
        smoothed = data.ewm(span=smoothing_window).mean()
        
        return smoothed
    
    def get_sentiment_extremes(self, data: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
        """
        Identify sentiment extremes that may indicate market turning points.
        
        Args:
            data: Sentiment data
            threshold: Z-score threshold for extremes
            
        Returns:
            DataFrame with extreme sentiment signals
        """
        
        extremes = pd.DataFrame(index=data.index)
        
        for col in data.columns:
            if '_sentiment_zscore' in col:
                symbol = col.replace('_sentiment_zscore', '')
                
                # Extreme pessimism (potential bottom)
                extremes[f'{symbol}_extreme_pessimism'] = data[col] < -threshold
                
                # Extreme optimism (potential top)
                extremes[f'{symbol}_extreme_optimism'] = data[col] > threshold
                
                # Sentiment divergence (momentum vs level)
                sentiment_col = f'{symbol}_sentiment'
                momentum_col = f'{symbol}_sentiment_momentum'
                
                if sentiment_col in data.columns and momentum_col in data.columns:
                    # Bullish divergence: sentiment improving while at low levels
                    extremes[f'{symbol}_bullish_divergence'] = (
                        (data[sentiment_col] < data[sentiment_col].rolling(30).quantile(0.25)) &
                        (data[momentum_col] > 0)
                    )
                    
                    # Bearish divergence: sentiment deteriorating while at high levels
                    extremes[f'{symbol}_bearish_divergence'] = (
                        (data[sentiment_col] > data[sentiment_col].rolling(30).quantile(0.75)) &
                        (data[momentum_col] < 0)
                    )
        
        return extremes
    
    def calculate_market_sentiment_regime(self, data: pd.DataFrame) -> pd.Series:
        """
        Classify overall market sentiment into regimes.
        
        Returns:
            Series with sentiment regime classification
        """
        
        # Calculate aggregate sentiment across all symbols
        sentiment_cols = [col for col in data.columns if '_sentiment' in col and '_momentum' not in col and '_zscore' not in col]
        
        if not sentiment_cols:
            return pd.Series(index=data.index, dtype=str)
        
        aggregate_sentiment = data[sentiment_cols].mean(axis=1)
        
        # Define regime thresholds
        high_threshold = aggregate_sentiment.rolling(252).quantile(0.8)  # Top 20%
        low_threshold = aggregate_sentiment.rolling(252).quantile(0.2)   # Bottom 20%
        
        # Classify regimes
        regimes = pd.Series(index=data.index, dtype=str)
        regimes[aggregate_sentiment > high_threshold] = 'euphoric'
        regimes[aggregate_sentiment < low_threshold] = 'despair' 
        regimes[(aggregate_sentiment >= low_threshold) & (aggregate_sentiment <= high_threshold)] = 'neutral'
        
        return regimes