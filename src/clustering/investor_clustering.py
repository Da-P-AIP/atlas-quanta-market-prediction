"""
Investor Behavior Clustering Analysis

Implements the core investor clustering methodology described in Atlas Quanta:
- Permanent holding type (long-term oriented investors)
- Fixed value trading type (price target achievers)
- Panic type (news/rumor sensitive with panic selling)
- Standard deviation deviation scoring for bottom/top detection
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class InvestorCluster:
    """Data class representing an investor cluster"""
    cluster_id: int
    name: str
    description: str
    characteristics: Dict[str, float]
    current_deviation: float
    historical_std: float
    

class InvestorClusteringAnalyzer:
    """
    Investor Behavior Clustering and Deviation Analysis
    
    This class implements the core methodology for:
    1. Clustering investors based on trading behavior patterns
    2. Calculating deviation scores for each cluster
    3. Generating market extreme signals (bottom/top detection)
    """
    
    def __init__(self, window_size: int = 252, n_clusters: int = 3):
        """
        Initialize the clustering analyzer
        
        Args:
            window_size: Rolling window for deviation calculation (trading days)
            n_clusters: Number of investor clusters (default: 3)
        """
        self.window_size = window_size
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.clusters: List[InvestorCluster] = []
        self.fitted = False
        
        # Cluster naming convention based on Atlas Quanta methodology
        self.cluster_names = {
            0: "Permanent Holding Type",
            1: "Fixed Value Trading Type", 
            2: "Panic Type"
        }
        
        self.cluster_descriptions = {
            0: "Long-term oriented investors with stable holding patterns",
            1: "Price target achievers who sell at predetermined levels",
            2: "News/rumor sensitive investors prone to panic selling"
        }
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature matrix for clustering analysis
        
        Expected columns in data:
        - volume: Trading volume
        - turnover_rate: Stock turnover rate
        - holding_period: Average holding period
        - price_volatility: Price volatility measure
        - trade_frequency: Trading frequency
        - position_change: Position change rate
        
        Args:
            data: DataFrame with investor behavior data
            
        Returns:
            Feature matrix for clustering
        """
        logger.info("Preparing features for investor clustering")
        
        features = pd.DataFrame()
        
        # Core behavioral indicators
        features['turnover_rate'] = data['turnover_rate']
        features['holding_period'] = data['holding_period']
        features['trade_frequency'] = data['trade_frequency']
        features['volatility_response'] = data['price_volatility'] / data['volume']
        features['position_stability'] = 1 / (1 + np.abs(data['position_change']))
        
        # Rolling statistics (momentum indicators)
        features['turnover_ma'] = features['turnover_rate'].rolling(20).mean()
        features['holding_trend'] = features['holding_period'].pct_change(10)
        features['frequency_accel'] = features['trade_frequency'].diff(5)
        
        # News sensitivity proxy
        if 'news_volume' in data.columns:
            features['news_sensitivity'] = data['position_change'].abs() / (data['news_volume'] + 1)
        else:
            features['news_sensitivity'] = features['volatility_response']
            
        return features.fillna(method='ffill').fillna(0)
    
    def fit_clusters(self, data: pd.DataFrame) -> 'InvestorClusteringAnalyzer':
        """
        Fit clustering model to investor behavior data
        
        Args:
            data: DataFrame with investor behavior features
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting investor clusters with {len(data)} observations")
        
        # Prepare features
        features = self.prepare_features(data)
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit clustering model
        cluster_labels = self.kmeans.fit_predict(features_scaled)
        
        # Create cluster objects with characteristics
        self.clusters = []
        for i in range(self.n_clusters):
            cluster_mask = cluster_labels == i
            cluster_data = features[cluster_mask]
            
            # Calculate cluster characteristics
            characteristics = {
                'avg_turnover': cluster_data['turnover_rate'].mean(),
                'avg_holding_period': cluster_data['holding_period'].mean(),
                'avg_trade_frequency': cluster_data['trade_frequency'].mean(),
                'volatility_sensitivity': cluster_data['volatility_response'].mean(),
                'position_stability': cluster_data['position_stability'].mean(),
                'news_sensitivity': cluster_data['news_sensitivity'].mean()
            }
            
            cluster = InvestorCluster(
                cluster_id=i,
                name=self.cluster_names.get(i, f"Cluster {i}"),
                description=self.cluster_descriptions.get(i, f"Investor cluster {i}"),
                characteristics=characteristics,
                current_deviation=0.0,
                historical_std=1.0
            )
            
            self.clusters.append(cluster)
        
        self.fitted = True
        logger.info("Investor clustering completed successfully")
        return self
    
    def calculate_deviation_scores(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate current deviation scores for each investor cluster
        
        This is the core of the Atlas Quanta bottom/top detection methodology.
        When deviation scores exceed ±2σ, it signals potential market extremes.
        
        Args:
            current_data: Current period investor behavior data
            
        Returns:
            Dictionary of cluster deviation scores
        """
        if not self.fitted:
            raise ValueError("Must fit clusters before calculating deviations")
            
        logger.info("Calculating investor cluster deviation scores")
        
        # Prepare current features
        current_features = self.prepare_features(current_data)
        current_scaled = self.scaler.transform(current_features)
        
        # Predict cluster assignments
        cluster_assignments = self.kmeans.predict(current_scaled)
        
        deviation_scores = {}
        
        for i, cluster in enumerate(self.clusters):
            # Get current period data for this cluster
            cluster_mask = cluster_assignments == i
            
            if cluster_mask.sum() == 0:
                deviation_scores[cluster.name] = 0.0
                continue
                
            current_cluster_data = current_features[cluster_mask]
            
            # Calculate key behavioral indicators
            current_turnover = current_cluster_data['turnover_rate'].mean()
            current_holding = current_cluster_data['holding_period'].mean()
            current_frequency = current_cluster_data['trade_frequency'].mean()
            
            # Historical benchmarks (from fitted characteristics)
            hist_turnover = cluster.characteristics['avg_turnover']
            hist_holding = cluster.characteristics['avg_holding_period']
            hist_frequency = cluster.characteristics['avg_trade_frequency']
            
            # Calculate deviation components
            turnover_dev = (current_turnover - hist_turnover) / (hist_turnover + 1e-6)
            holding_dev = (hist_holding - current_holding) / (hist_holding + 1e-6)  # Inverse for holding
            frequency_dev = (current_frequency - hist_frequency) / (hist_frequency + 1e-6)
            
            # Composite deviation score (weighted average)
            composite_deviation = (
                0.4 * turnover_dev +
                0.3 * holding_dev +
                0.3 * frequency_dev
            )
            
            # Convert to standard deviation units (approximate)
            # In practice, this would use historical rolling standard deviation
            standardized_deviation = composite_deviation / 0.2  # Rough approximation
            
            deviation_scores[cluster.name] = standardized_deviation
            
            # Update cluster with current deviation
            cluster.current_deviation = standardized_deviation
        
        return deviation_scores
    
    def detect_market_extremes(self, deviation_scores: Dict[str, float], 
                              threshold: float = 2.0) -> Dict[str, str]:
        """
        Detect potential market bottom/top signals based on cluster deviations
        
        Atlas Quanta methodology:
        - Bottom signal: Multiple clusters showing extreme selling (deviation > +2σ)
        - Top signal: Speculative clusters showing extreme buying while conservative clusters sell
        
        Args:
            deviation_scores: Current deviation scores for each cluster
            threshold: Standard deviation threshold for extreme signal
            
        Returns:
            Dictionary of market signals
        """
        signals = {}
        
        # Extract deviations by cluster type
        permanent_dev = deviation_scores.get("Permanent Holding Type", 0)
        trading_dev = deviation_scores.get("Fixed Value Trading Type", 0) 
        panic_dev = deviation_scores.get("Panic Type", 0)
        
        # Bottom detection logic
        if (permanent_dev > threshold and panic_dev > threshold):
            signals['market_signal'] = 'BOTTOM_CANDIDATE'
            signals['confidence'] = min((permanent_dev + panic_dev) / (2 * threshold), 3.0)
            signals['reasoning'] = "Even stable investors are selling heavily - potential exhaustion"
            
        # Top detection logic  
        elif (panic_dev < -threshold and trading_dev < -threshold and permanent_dev > 0):
            signals['market_signal'] = 'TOP_CANDIDATE'
            signals['confidence'] = min(abs(panic_dev + trading_dev) / (2 * threshold), 3.0)
            signals['reasoning'] = "Speculative buying while smart money exits"
            
        # Neutral/trending market
        else:
            signals['market_signal'] = 'NEUTRAL'
            signals['confidence'] = 0.5
            signals['reasoning'] = "No extreme cluster behavior detected"
        
        # Add individual cluster signals
        for cluster_name, deviation in deviation_scores.items():
            if abs(deviation) > threshold:
                direction = "EXTREME_SELL" if deviation > 0 else "EXTREME_BUY"
                signals[f"{cluster_name}_signal"] = direction
        
        return signals
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Get summary of all investor clusters and their characteristics
        
        Returns:
            DataFrame with cluster summary information
        """
        if not self.fitted:
            return pd.DataFrame()
            
        summary_data = []
        for cluster in self.clusters:
            row = {
                'cluster_id': cluster.cluster_id,
                'name': cluster.name,
                'description': cluster.description,
                'current_deviation': cluster.current_deviation,
                **cluster.characteristics
            }
            summary_data.append(row)
            
        return pd.DataFrame(summary_data)
