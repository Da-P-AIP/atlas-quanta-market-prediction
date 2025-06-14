"""
Investor Clustering Module for Atlas Quanta

Implements the core investor behavior clustering analysis based on deviation scoring.
This module classifies investors into different behavioral types and calculates
deviation scores for market timing signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
from datetime import datetime, timedelta

class InvestorClustering:
    """
    Investor behavior clustering and deviation analysis.
    
    This class implements the core logic for:
    1. Clustering investors based on behavioral patterns
    2. Calculating deviation scores for each cluster
    3. Generating bottom/top signals from extreme deviations
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the investor clustering system.
        
        Args:
            config: Configuration dictionary with clustering parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.n_clusters = self.config.get('n_clusters', 3)
        self.lookback_window = self.config.get('lookback_window', 252)  # Trading days
        self.deviation_threshold = self.config.get('deviation_threshold', 2.0)  # Sigma
        
        # Cluster names based on Atlas Quanta specification
        self.cluster_names = {
            0: 'permanent_hold',      # 恒久保有型
            1: 'tactical_trading',    # 一定値売買型
            2: 'panic_driven'         # 狼狽型
        }
        
        # Models and scalers
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.pca = PCA(n_components=0.95)  # Retain 95% variance
        
        # Historical data for deviation calculation
        self.historical_features = {}
        self.cluster_statistics = {}
        
        self.logger.info(f"Investor clustering initialized with {self.n_clusters} clusters")
    
    def fit_clusters(self, investor_data: pd.DataFrame) -> Dict[str, any]:
        """
        Fit clustering model on historical investor behavior data.
        
        Args:
            investor_data: DataFrame with columns:
                - turnover_rate: Trading frequency measure
                - hold_period: Average holding period in days
                - volatility_sensitivity: Reaction to market volatility
                - news_sensitivity: Reaction to news events
                - volume_ratio: Relative trading volume
                
        Returns:
            Dictionary with clustering results and statistics
        """
        self.logger.info(f"Fitting clusters on {len(investor_data)} observations")
        
        # Feature engineering
        features = self._engineer_features(investor_data)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA for dimensionality reduction
        features_pca = self.pca.fit_transform(features_scaled)
        
        # Fit clustering
        cluster_labels = self.kmeans.fit_predict(features_pca)
        
        # Store results
        results = {
            'cluster_labels': cluster_labels,
            'cluster_centers': self.kmeans.cluster_centers_,
            'features_original': features,
            'features_scaled': features_scaled,
            'features_pca': features_pca,
            'explained_variance_ratio': self.pca.explained_variance_ratio_
        }
        
        # Calculate cluster statistics
        self.cluster_statistics = self._calculate_cluster_statistics(
            investor_data, cluster_labels, features
        )
        
        # Store historical features for deviation calculation
        self.historical_features = features.copy()
        
        self.logger.info("Cluster fitting completed successfully")
        return results
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for investor clustering.
        
        Args:
            data: Raw investor behavior data
            
        Returns:
            Engineered features DataFrame
        """
        features = pd.DataFrame(index=data.index)
        
        # Core behavioral features
        features['turnover_rate'] = data.get('turnover_rate', 0)
        features['hold_period_log'] = np.log1p(data.get('hold_period', 30))
        features['volatility_beta'] = data.get('volatility_sensitivity', 1.0)
        features['news_reaction_speed'] = data.get('news_sensitivity', 0.5)
        features['relative_volume'] = data.get('volume_ratio', 1.0)
        
        # Derived features
        features['trading_intensity'] = (
            features['turnover_rate'] / (features['hold_period_log'] + 1)
        )
        features['stability_score'] = (
            1 / (1 + features['volatility_beta'] * features['news_reaction_speed'])
        )
        features['momentum_tendency'] = (
            features['turnover_rate'] * features['relative_volume']
        )
        
        # Risk behavior features
        features['panic_probability'] = (
            features['news_reaction_speed'] * features['volatility_beta']
        )
        features['contrarian_score'] = 1 / (1 + features['momentum_tendency'])
        
        return features.fillna(features.median())
    
    def _calculate_cluster_statistics(self, 
                                    original_data: pd.DataFrame,
                                    cluster_labels: np.ndarray,
                                    features: pd.DataFrame) -> Dict[int, Dict]:
        """
        Calculate statistical properties for each cluster.
        
        Args:
            original_data: Original investor data
            cluster_labels: Cluster assignment for each observation
            features: Engineered features
            
        Returns:
            Dictionary with statistics for each cluster
        """
        stats = {}
        
        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            cluster_data = features[mask]
            
            if len(cluster_data) == 0:
                continue
            
            stats[cluster_id] = {
                'name': self.cluster_names.get(cluster_id, f'cluster_{cluster_id}'),
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(features) * 100,
                'mean_features': cluster_data.mean().to_dict(),
                'std_features': cluster_data.std().to_dict(),
                'characteristics': self._characterize_cluster(cluster_data)
            }
            
        return stats
    
    def _characterize_cluster(self, cluster_data: pd.DataFrame) -> Dict[str, str]:
        """
        Generate human-readable characteristics for a cluster.
        
        Args:
            cluster_data: Feature data for the cluster
            
        Returns:
            Dictionary with cluster characteristics
        """
        characteristics = {}
        
        # Trading frequency
        avg_turnover = cluster_data['turnover_rate'].mean()
        if avg_turnover > 2.0:
            characteristics['trading_style'] = 'High frequency trader'
        elif avg_turnover > 0.5:
            characteristics['trading_style'] = 'Active trader'
        else:
            characteristics['trading_style'] = 'Long-term holder'
        
        # Risk tolerance
        avg_volatility_beta = cluster_data['volatility_beta'].mean()
        if avg_volatility_beta > 1.5:
            characteristics['risk_tolerance'] = 'High risk, volatile reactions'
        elif avg_volatility_beta > 0.8:
            characteristics['risk_tolerance'] = 'Moderate risk sensitivity'
        else:
            characteristics['risk_tolerance'] = 'Low risk, stable behavior'
        
        # News sensitivity
        avg_news_sensitivity = cluster_data['news_reaction_speed'].mean()
        if avg_news_sensitivity > 0.8:
            characteristics['news_behavior'] = 'Highly reactive to news'
        elif avg_news_sensitivity > 0.3:
            characteristics['news_behavior'] = 'Moderately news-sensitive'
        else:
            characteristics['news_behavior'] = 'News-independent decisions'
        
        return characteristics
    
    def calculate_deviation_scores(self, 
                                 current_data: pd.DataFrame,
                                 reference_period_days: int = 60) -> Dict[str, float]:
        """
        Calculate deviation scores for current investor behavior.
        
        Args:
            current_data: Current investor behavior data
            reference_period_days: Period for calculating historical norms
            
        Returns:
            Dictionary with deviation scores for each cluster
        """
        if self.historical_features.empty:
            raise ValueError("No historical data available. Run fit_clusters first.")
        
        self.logger.info("Calculating deviation scores for current market conditions")
        
        # Engineer features for current data
        current_features = self._engineer_features(current_data)
        
        # Predict cluster assignments for current data
        current_scaled = self.scaler.transform(current_features)
        current_pca = self.pca.transform(current_scaled)
        current_clusters = self.kmeans.predict(current_pca)
        
        # Calculate deviations for each cluster
        deviation_scores = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_name = self.cluster_names.get(cluster_id, f'cluster_{cluster_id}')
            
            # Get current cluster data
            current_mask = current_clusters == cluster_id
            if not np.any(current_mask):
                deviation_scores[f'{cluster_name}_deviation'] = 0.0
                continue
            
            current_cluster_data = current_features[current_mask]
            
            # Get historical reference for this cluster
            if cluster_id in self.cluster_statistics:
                historical_means = pd.Series(
                    self.cluster_statistics[cluster_id]['mean_features']
                )
                historical_stds = pd.Series(
                    self.cluster_statistics[cluster_id]['std_features']
                )
                
                # Calculate z-scores for key behavioral indicators
                current_means = current_cluster_data.mean()
                
                # Focus on key deviation indicators
                key_features = ['turnover_rate', 'volatility_beta', 'panic_probability']
                z_scores = []
                
                for feature in key_features:
                    if feature in current_means.index and historical_stds[feature] > 0:
                        z_score = (
                            (current_means[feature] - historical_means[feature]) /
                            historical_stds[feature]
                        )
                        z_scores.append(z_score)
                
                # Aggregate deviation score
                if z_scores:
                    deviation_score = np.mean(np.abs(z_scores))
                else:
                    deviation_score = 0.0
                
                deviation_scores[f'{cluster_name}_deviation'] = deviation_score
            else:
                deviation_scores[f'{cluster_name}_deviation'] = 0.0
        
        # Generate market timing signals
        signals = self._generate_timing_signals(deviation_scores)
        deviation_scores.update(signals)
        
        return deviation_scores
    
    def _generate_timing_signals(self, deviation_scores: Dict[str, float]) -> Dict[str, any]:
        """
        Generate bottom/top signals from deviation scores.
        
        Args:
            deviation_scores: Calculated deviation scores
            
        Returns:
            Dictionary with timing signals
        """
        signals = {}
        
        # Extract individual cluster deviations
        permanent_dev = deviation_scores.get('permanent_hold_deviation', 0)
        tactical_dev = deviation_scores.get('tactical_trading_deviation', 0)
        panic_dev = deviation_scores.get('panic_driven_deviation', 0)
        
        # Bottom signal logic (extreme selling)
        # When even stable investors are selling heavily
        bottom_signal_strength = 0
        if permanent_dev > self.deviation_threshold and panic_dev > self.deviation_threshold:
            bottom_signal_strength = min(5, (permanent_dev + panic_dev) / 2)
        
        # Top signal logic (extreme buying/complacency)
        # When tactical traders are very active but fundamentals don't justify
        top_signal_strength = 0
        if tactical_dev > self.deviation_threshold and permanent_dev < 0.5:
            top_signal_strength = min(5, tactical_dev - permanent_dev)
        
        signals.update({
            'bottom_signal_strength': bottom_signal_strength,
            'top_signal_strength': top_signal_strength,
            'overall_market_stress': np.mean([permanent_dev, tactical_dev, panic_dev]),
            'contrarian_opportunity': bottom_signal_strength - top_signal_strength
        })
        
        # Signal interpretation
        if bottom_signal_strength > 3:
            signals['market_signal'] = 'STRONG_BOTTOM'
        elif bottom_signal_strength > 2:
            signals['market_signal'] = 'POTENTIAL_BOTTOM'
        elif top_signal_strength > 3:
            signals['market_signal'] = 'STRONG_TOP'
        elif top_signal_strength > 2:
            signals['market_signal'] = 'POTENTIAL_TOP'
        else:
            signals['market_signal'] = 'NEUTRAL'
        
        return signals
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Get summary of cluster characteristics.
        
        Returns:
            DataFrame with cluster summary information
        """
        if not self.cluster_statistics:
            return pd.DataFrame()
        
        summary_data = []
        for cluster_id, stats in self.cluster_statistics.items():
            summary_data.append({
                'cluster_id': cluster_id,
                'name': stats['name'],
                'size': stats['size'],
                'percentage': stats['percentage'],
                'avg_turnover': stats['mean_features'].get('turnover_rate', 0),
                'avg_hold_period': np.exp(stats['mean_features'].get('hold_period_log', 0)) - 1,
                'volatility_sensitivity': stats['mean_features'].get('volatility_beta', 1),
                'trading_style': stats['characteristics'].get('trading_style', 'Unknown'),
                'risk_tolerance': stats['characteristics'].get('risk_tolerance', 'Unknown')
            })
        
        return pd.DataFrame(summary_data)

    def generate_synthetic_investor_data(self, 
                                       n_samples: int = 1000,
                                       market_regime: str = 'normal') -> pd.DataFrame:
        """
        Generate synthetic investor behavior data for testing.
        
        Args:
            n_samples: Number of samples to generate
            market_regime: Market condition ('normal', 'stress', 'euphoria')
            
        Returns:
            Synthetic investor behavior DataFrame
        """
        np.random.seed(42)  # For reproducible results
        
        # Base parameters by market regime
        regime_params = {
            'normal': {'vol_mult': 1.0, 'news_mult': 1.0, 'panic_mult': 1.0},
            'stress': {'vol_mult': 2.0, 'news_mult': 1.5, 'panic_mult': 2.5},
            'euphoria': {'vol_mult': 0.7, 'news_mult': 0.8, 'panic_mult': 0.5}
        }
        
        params = regime_params.get(market_regime, regime_params['normal'])
        
        # Generate base features with regime adjustments
        data = {
            'turnover_rate': np.random.lognormal(
                mean=-1, sigma=1, size=n_samples
            ) * params['vol_mult'],
            'hold_period': np.random.exponential(
                scale=45, size=n_samples
            ) / params['vol_mult'],
            'volatility_sensitivity': np.random.gamma(
                shape=2, scale=0.5, size=n_samples
            ) * params['vol_mult'],
            'news_sensitivity': np.random.beta(
                a=2, b=3, size=n_samples
            ) * params['news_mult'],
            'volume_ratio': np.random.lognormal(
                mean=0, sigma=0.3, size=n_samples
            ) * params['panic_mult']
        }
        
        df = pd.DataFrame(data)
        
        # Add timestamp
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=n_samples),
            periods=n_samples,
            freq='D'
        )
        df.index = dates
        
        return df
