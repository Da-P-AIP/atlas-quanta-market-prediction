"""
Atlas Quanta Main Prediction System

Main class that orchestrates all components of the Atlas Quanta market prediction system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import logging
import yaml
from pathlib import Path

from .clustering.investor_clustering import InvestorClustering
from .indicators.vpin import VPINCalculator
from .models.tvp_var import TVPVARModel
from .models.dma import DMAModel
from .sentiment.sentiment_analyzer import SentimentAnalyzer
from .risk_management.var_calculator import VaRCalculator
from .risk_management.position_sizing import PositionSizer

class AtlasQuanta:
    """
    Main Atlas Quanta prediction system that combines all components.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the Atlas Quanta system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.investor_clustering = InvestorClustering(self.config.get('clustering', {}))
        self.vpin_calculator = VPINCalculator(self.config.get('vpin', {}))
        self.tvp_var_model = TVPVARModel(self.config.get('tvp_var', {}))
        self.dma_model = DMAModel(self.config.get('dma', {}))
        self.sentiment_analyzer = SentimentAnalyzer(self.config.get('sentiment', {}))
        self.var_calculator = VaRCalculator(self.config.get('risk', {}))
        self.position_sizer = PositionSizer(self.config.get('position_sizing', {}))
        
        # Data storage
        self.market_data = {}
        self.features = {}
        self.predictions = {}
        
        self.logger.info("Atlas Quanta system initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            raise ValueError(f"Error loading config from {config_path}: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the system."""
        logger = logging.getLogger('AtlasQuanta')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(self.config.get('log_level', 'INFO'))
        return logger
    
    def update_data(self, symbols: List[str] = None, days_back: int = 252) -> None:
        """
        Update market data for specified symbols.
        
        Args:
            symbols: List of symbols to update. If None, uses default symbols from config
            days_back: Number of days of historical data to fetch
        """
        if symbols is None:
            symbols = self.config.get('default_symbols', ['SPY', 'QQQ', 'IWM'])
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        self.logger.info(f"Updating data for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
        
        # TODO: Implement actual data fetching from configured sources
        # For now, this is a placeholder
        for symbol in symbols:
            self.market_data[symbol] = self._fetch_symbol_data(symbol, start_date, end_date)
        
        self.logger.info("Market data update completed")
    
    def _fetch_symbol_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data for a single symbol (placeholder implementation)."""
        # This is a placeholder - actual implementation would use data sources
        dates = pd.date_range(start_date, end_date, freq='D')
        n_days = len(dates)
        
        # Generate synthetic data for demonstration
        np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
        price_base = 100
        returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
        prices = price_base * np.cumprod(1 + returns)
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
            'close': prices,
            'volume': np.random.lognormal(15, 0.5, n_days),
            'symbol': symbol
        }, index=dates)
        
        return data
    
    def generate_features(self, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate all features for prediction models.
        
        Args:
            symbols: List of symbols to generate features for
            
        Returns:
            Dictionary mapping symbol to feature DataFrame
        """
        if symbols is None:
            symbols = list(self.market_data.keys())
        
        self.logger.info(f"Generating features for {len(symbols)} symbols")
        
        for symbol in symbols:
            if symbol not in self.market_data:
                self.logger.warning(f"No market data available for {symbol}")
                continue
            
            data = self.market_data[symbol]
            features = pd.DataFrame(index=data.index)
            
            # Calculate VPIN indicator
            try:
                vpin_values = self.vpin_calculator.calculate_vpin(
                    data['close'], data['volume']
                )
                features['vpin'] = vpin_values
                self.logger.debug(f"VPIN calculated for {symbol}")
            except Exception as e:
                self.logger.error(f"Error calculating VPIN for {symbol}: {e}")
            
            # Calculate investor clustering features
            try:
                # This would use actual investor flow data in production
                cluster_features = self._generate_clustering_features(data)
                features = pd.concat([features, cluster_features], axis=1)
                self.logger.debug(f"Clustering features calculated for {symbol}")
            except Exception as e:
                self.logger.error(f"Error calculating clustering features for {symbol}: {e}")
            
            # Add technical indicators
            features['returns'] = data['close'].pct_change()
            features['volatility'] = features['returns'].rolling(20).std()
            features['rsi'] = self._calculate_rsi(data['close'])
            features['volume_sma'] = data['volume'].rolling(20).mean()
            
            # Seasonal features
            features['month'] = features.index.month
            features['day_of_week'] = features.index.dayofweek
            features['quarter'] = features.index.quarter
            
            self.features[symbol] = features
        
        self.logger.info("Feature generation completed")
        return self.features
    
    def _generate_clustering_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate investor clustering features (placeholder)."""
        # Placeholder implementation - would use actual investor flow data
        features = pd.DataFrame(index=data.index)
        
        # Simulate cluster deviation scores
        features['cluster_permanent_deviation'] = np.random.normal(0, 1, len(data))
        features['cluster_tactical_deviation'] = np.random.normal(0, 1, len(data))
        features['cluster_panic_deviation'] = np.random.normal(0, 1.5, len(data))
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def predict(self, 
                symbols: List[str],
                horizon_days: int = 100,
                include_sentiment: bool = True,
                include_risk_metrics: bool = True) -> Dict[str, Dict]:
        """
        Generate predictions for specified symbols.
        
        Args:
            symbols: List of symbols to predict
            horizon_days: Prediction horizon in days
            include_sentiment: Whether to include sentiment analysis
            include_risk_metrics: Whether to calculate risk metrics
            
        Returns:
            Dictionary with predictions for each symbol
        """
        self.logger.info(f"Generating {horizon_days}-day predictions for {len(symbols)} symbols")
        
        predictions = {}
        
        for symbol in symbols:
            if symbol not in self.features:
                self.logger.warning(f"No features available for {symbol}")
                continue
            
            symbol_predictions = {}
            features = self.features[symbol].dropna()
            
            if len(features) < 50:  # Minimum data requirement
                self.logger.warning(f"Insufficient data for {symbol} ({len(features)} observations)")
                continue
            
            # TVP-VAR prediction
            try:
                tvp_pred = self.tvp_var_model.predict(
                    features[['returns', 'volatility', 'volume_sma']].tail(100),
                    horizon=horizon_days
                )
                symbol_predictions['tvp_var'] = tvp_pred
                self.logger.debug(f"TVP-VAR prediction completed for {symbol}")
            except Exception as e:
                self.logger.error(f"TVP-VAR prediction error for {symbol}: {e}")
            
            # DMA prediction
            try:
                dma_pred = self.dma_model.predict(
                    features[['returns', 'rsi', 'volatility']].tail(100),
                    horizon=horizon_days
                )
                symbol_predictions['dma'] = dma_pred
                self.logger.debug(f"DMA prediction completed for {symbol}")
            except Exception as e:
                self.logger.error(f"DMA prediction error for {symbol}: {e}")
            
            # Sentiment analysis
            if include_sentiment:
                try:
                    sentiment_data = self.sentiment_analyzer.get_composite_sentiment(
                        [symbol],
                        datetime.now() - timedelta(days=30),
                        datetime.now()
                    )
                    symbol_predictions['sentiment'] = {
                        'current_sentiment': sentiment_data.iloc[-1] if not sentiment_data.empty else None,
                        'sentiment_trend': 'neutral'  # Placeholder
                    }
                except Exception as e:
                    self.logger.error(f"Sentiment analysis error for {symbol}: {e}")
            
            # Risk metrics
            if include_risk_metrics:
                try:
                    returns = features['returns'].dropna()
                    var_95 = self.var_calculator.calculate_historical_var(returns, 0.95)
                    var_99 = self.var_calculator.calculate_historical_var(returns, 0.99)
                    
                    symbol_predictions['risk_metrics'] = {
                        'var_95': var_95.iloc[-1] if not var_95.empty else None,
                        'var_99': var_99.iloc[-1] if not var_99.empty else None,
                        'current_volatility': features['volatility'].iloc[-1]
                    }
                except Exception as e:
                    self.logger.error(f"Risk metrics error for {symbol}: {e}")
            
            # Generate composite prediction
            symbol_predictions['composite'] = self._generate_composite_prediction(
                symbol_predictions, horizon_days
            )
            
            predictions[symbol] = symbol_predictions
        
        self.predictions = predictions
        self.logger.info("Prediction generation completed")
        return predictions
    
    def _generate_composite_prediction(self, 
                                      symbol_predictions: Dict, 
                                      horizon_days: int) -> Dict:
        """
        Generate composite prediction from individual model predictions.
        """
        # Placeholder implementation - would use sophisticated ensemble methods
        composite = {
            'direction': 'neutral',  # up, down, neutral
            'confidence': 0.5,
            'price_target': None,
            'probability_up': 0.5,
            'expected_return': 0.0
        }
        
        # Simple averaging of available predictions
        predictions_available = []
        
        if 'tvp_var' in symbol_predictions:
            predictions_available.append(0.1)  # Placeholder return
        
        if 'dma' in symbol_predictions:
            predictions_available.append(-0.05)  # Placeholder return
        
        if predictions_available:
            avg_return = np.mean(predictions_available)
            composite['expected_return'] = avg_return
            composite['direction'] = 'up' if avg_return > 0.02 else 'down' if avg_return < -0.02 else 'neutral'
            composite['confidence'] = min(0.9, 0.5 + abs(avg_return) * 10)
            composite['probability_up'] = 0.5 + avg_return * 5  # Simple transformation
        
        return composite
    
    def get_prediction_summary(self) -> pd.DataFrame:
        """
        Get summary of all current predictions.
        
        Returns:
            DataFrame with prediction summary
        """
        if not self.predictions:
            return pd.DataFrame()
        
        summary_data = []
        
        for symbol, pred_data in self.predictions.items():
            if 'composite' in pred_data:
                comp = pred_data['composite']
                summary_data.append({
                    'symbol': symbol,
                    'direction': comp.get('direction', 'neutral'),
                    'confidence': comp.get('confidence', 0.5),
                    'expected_return': comp.get('expected_return', 0.0),
                    'probability_up': comp.get('probability_up', 0.5),
                    'var_95': pred_data.get('risk_metrics', {}).get('var_95'),
                    'current_sentiment': pred_data.get('sentiment', {}).get('current_sentiment')
                })
        
        return pd.DataFrame(summary_data)
    
    def display_dashboard(self, predictions: Dict = None):
        """
        Display prediction dashboard (placeholder for future GUI/web interface).
        """
        if predictions is None:
            predictions = self.predictions
        
        print("\n" + "="*80)
        print("                    ATLAS QUANTA PREDICTION DASHBOARD")
        print("="*80)
        
        summary = self.get_prediction_summary()
        
        if summary.empty:
            print("No predictions available. Run predict() first.")
            return
        
        print(f"\nPrediction Summary ({len(summary)} symbols):")
        print("-" * 60)
        
        for _, row in summary.iterrows():
            print(f"Symbol: {row['symbol']:>8} | Direction: {row['direction']:>8} | "
                  f"Confidence: {row['confidence']:>5.1%} | Expected Return: {row['expected_return']:>+6.2%}")
        
        print("\n" + "="*80)
        print("Disclaimer: This is for educational purposes only. Not financial advice.")
        print("="*80 + "\n")