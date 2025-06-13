"""
Fear & Greed Index Data Collector

Collects and processes CNN Fear & Greed Index data for market sentiment analysis.
"""

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

class FearGreedIndexCollector:
    """
    Collector for CNN Fear & Greed Index data.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        
    def get_sentiment(self, 
                     symbols: List[str],  # Not used for F&G but kept for interface compatibility
                     start_date: datetime,
                     end_date: datetime) -> pd.DataFrame:
        """
        Fetch Fear & Greed Index data.
        
        Returns:
            DataFrame with F&G index values
        """
        
        try:
            # Fetch current and historical data
            response = requests.get(self.base_url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Process the data
            df = self._process_fear_greed_data(data, start_date, end_date)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching Fear & Greed Index: {e}")
            return pd.DataFrame()
    
    def _process_fear_greed_data(self, data: Dict, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Process raw Fear & Greed Index data.
        """
        
        records = []
        
        # Current score
        if 'fear_and_greed' in data:
            current_data = data['fear_and_greed']
            if 'score' in current_data and 'timestamp' in current_data:
                timestamp = datetime.fromtimestamp(current_data['timestamp'] / 1000)
                score = float(current_data['score'])
                records.append({
                    'timestamp': timestamp,
                    'fear_greed_score': score,
                    'fear_greed_rating': self._score_to_rating(score)
                })
        
        # Historical data
        if 'fear_and_greed_historical' in data and 'data' in data['fear_and_greed_historical']:
            for item in data['fear_and_greed_historical']['data']:
                if 'x' in item and 'y' in item:
                    timestamp = datetime.fromtimestamp(item['x'] / 1000)
                    score = float(item['y'])
                    
                    if start_date <= timestamp <= end_date:
                        records.append({
                            'timestamp': timestamp,
                            'fear_greed_score': score,
                            'fear_greed_rating': self._score_to_rating(score)
                        })
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Add derived indicators
        df['fear_greed_momentum'] = df['fear_greed_score'].diff(5)  # 5-day momentum
        df['fear_greed_ma20'] = df['fear_greed_score'].rolling(20).mean()
        df['fear_greed_deviation'] = df['fear_greed_score'] - df['fear_greed_ma20']
        
        # Normalize to [-1, 1] scale for consistency with other sentiment sources
        df['fear_greed_normalized'] = (df['fear_greed_score'] - 50) / 50
        
        # Create contrarian signals
        df['fear_greed_contrarian'] = -df['fear_greed_normalized']  # Contrarian interpretation
        
        self.logger.info(f"Processed {len(df)} Fear & Greed Index records")
        
        return df
    
    def _score_to_rating(self, score: float) -> str:
        """
        Convert numeric score to rating category.
        """
        if score >= 75:
            return "Extreme Greed"
        elif score >= 55:
            return "Greed"
        elif score >= 45:
            return "Neutral"
        elif score >= 25:
            return "Fear"
        else:
            return "Extreme Fear"
    
    def get_extreme_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate extreme sentiment signals from Fear & Greed data.
        """
        
        signals = pd.DataFrame(index=data.index)
        
        if 'fear_greed_score' in data.columns:
            # Extreme fear (potential buying opportunity)
            signals['extreme_fear_signal'] = data['fear_greed_score'] <= 20
            
            # Extreme greed (potential selling opportunity)
            signals['extreme_greed_signal'] = data['fear_greed_score'] >= 80
            
            # Rapid sentiment changes
            if 'fear_greed_momentum' in data.columns:
                signals['rapid_fear_increase'] = data['fear_greed_momentum'] <= -15  # Rapid decline in score
                signals['rapid_greed_increase'] = data['fear_greed_momentum'] >= 15   # Rapid increase in score
        
        return signals