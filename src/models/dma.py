"""
DMA (Dynamic Model Averaging) for Atlas Quanta

This module implements Dynamic Model Averaging for combining multiple prediction models
with time-varying weights based on their recent performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from datetime import datetime
from scipy.stats import norm, multivariate_normal
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

class DMAModel:
    """
    Dynamic Model Averaging for adaptive ensemble forecasting.
    
    Combines multiple base models with time-varying weights that adapt
    based on recent predictive performance and model uncertainty.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize DMA model.
        
        Args:
            config: Configuration dictionary with DMA parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # DMA configuration
        self.forgetting_factor = self.config.get('forgetting_factor', 0.99)
        self.include_predictions = self.config.get('include_predictions', True)
        self.prediction_window = self.config.get('prediction_window', 20)
        self.model_types = self.config.get('model_types', ['linear', 'ridge', 'random_forest'])
        
        # Performance tracking
        self.performance_window = self.config.get('performance_window', 50)
        self.weight_smoothing = self.config.get('weight_smoothing', 0.1)
        
        # Storage
        self.base_models = {}
        self.model_weights = {}
        self.model_performance = {}
        self.prediction_history = {}
        self.is_fitted = False
        
        # Initialize base models
        self._initialize_base_models()
        
        self.logger.info(f"DMA model initialized with {len(self.base_models)} base models")
    
    def _initialize_base_models(self) -> None:
        """Initialize the base models for ensemble."""
        model_configs = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'random_forest': RandomForestRegressor(
                n_estimators=50, 
                max_depth=5, 
                random_state=42
            ),
            'svr': SVR(kernel='rbf', gamma='scale')
        }
        
        for model_type in self.model_types:
            if model_type in model_configs:
                self.base_models[model_type] = {
                    'model': model_configs[model_type],
                    'fitted': False,
                    'last_performance': 0.0
                }
                self.model_weights[model_type] = 1.0 / len(self.model_types)
                self.model_performance[model_type] = []
                self.prediction_history[model_type] = []
    
    def fit(self, 
            data: pd.DataFrame,
            target_col: str,
            feature_cols: List[str] = None) -> Dict[str, any]:
        """
        Fit the DMA model and all base models.
        
        Args:
            data: Training data DataFrame
            target_col: Name of target variable column
            feature_cols: List of feature column names
            
        Returns:
            Dictionary with fitting results
        """
        self.logger.info(f"Fitting DMA model on {len(data)} observations")
        
        if feature_cols is None:
            feature_cols = [col for col in data.columns if col != target_col]
        
        # Prepare data
        X = data[feature_cols].dropna()
        y = data[target_col].loc[X.index]
        
        if len(X) < 20:
            raise ValueError("Insufficient data for DMA fitting")
        
        self.target_col = target_col
        self.feature_cols = feature_cols
        
        # Fit base models
        fit_results = {}
        
        for model_name, model_info in self.base_models.items():
            try:
                model = model_info['model']
                model.fit(X, y)
                model_info['fitted'] = True
                
                # Calculate in-sample performance
                y_pred = model.predict(X)
                mse = np.mean((y - y_pred) ** 2)
                mae = np.mean(np.abs(y - y_pred))
                
                fit_results[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'fitted': True
                }
                
                self.logger.debug(f"Model {model_name} fitted successfully")
                
            except Exception as e:
                self.logger.error(f"Error fitting model {model_name}: {e}")
                model_info['fitted'] = False
                fit_results[model_name] = {'fitted': False, 'error': str(e)}
        
        # Initialize equal weights for fitted models
        fitted_models = [name for name, info in self.base_models.items() if info['fitted']]
        if fitted_models:
            equal_weight = 1.0 / len(fitted_models)
            for model_name in self.base_models:
                self.model_weights[model_name] = equal_weight if model_name in fitted_models else 0.0
        
        self.is_fitted = True
        self.logger.info(f"DMA fitting completed. {len(fitted_models)} models ready.")
        
        return {
            'fitted_models': fitted_models,
            'model_results': fit_results,
            'initial_weights': self.model_weights.copy()
        }
    
    def predict(self, 
                data: pd.DataFrame,
                horizon: int = 1,
                update_weights: bool = True) -> Dict[str, any]:
        """
        Generate predictions using dynamic model averaging.
        
        Args:
            data: Input data for prediction
            horizon: Number of periods to forecast
            update_weights: Whether to update model weights based on recent performance
            
        Returns:
            Dictionary with predictions and weight evolution
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.logger.info(f"Generating DMA predictions for {horizon} periods")
        
        # Prepare features
        X = data[self.feature_cols].dropna()
        if len(X) == 0:
            raise ValueError("No valid feature data for prediction")
        
        # Generate predictions from each base model
        model_predictions = {}
        prediction_uncertainties = {}
        
        for model_name, model_info in self.base_models.items():
            if not model_info['fitted']:
                continue
                
            try:
                model = model_info['model']
                
                if horizon == 1:
                    # Single-step prediction
                    pred = model.predict(X.tail(1))
                    model_predictions[model_name] = pred[0]
                    
                    # Estimate uncertainty (simplified)
                    recent_errors = self._get_recent_errors(model_name)
                    uncertainty = np.std(recent_errors) if recent_errors else 0.1
                    prediction_uncertainties[model_name] = uncertainty
                    
                else:
                    # Multi-step prediction (recursive)
                    predictions = self._recursive_predict(model, X.tail(10), horizon)
                    model_predictions[model_name] = predictions
                    
                    # Multi-step uncertainty (increases with horizon)
                    base_uncertainty = 0.1
                    uncertainty = base_uncertainty * np.sqrt(horizon)
                    prediction_uncertainties[model_name] = uncertainty
                    
            except Exception as e:
                self.logger.error(f"Prediction error for model {model_name}: {e}")
                continue
        
        if not model_predictions:
            raise ValueError("No models available for prediction")
        
        # Update weights if requested and we have recent performance data
        if update_weights:
            self._update_model_weights(prediction_uncertainties)
        
        # Combine predictions using current weights
        if horizon == 1:
            weighted_prediction = self._combine_predictions(
                model_predictions, 
                self.model_weights
            )
            
            # Calculate ensemble uncertainty
            ensemble_uncertainty = self._calculate_ensemble_uncertainty(
                model_predictions,
                prediction_uncertainties,
                self.model_weights
            )
            
            return {
                'prediction': weighted_prediction,
                'uncertainty': ensemble_uncertainty,
                'model_predictions': model_predictions,
                'model_weights': self.model_weights.copy(),
                'model_uncertainties': prediction_uncertainties
            }
        else:
            # Multi-step predictions
            weighted_predictions = []
            
            for h in range(horizon):
                step_predictions = {name: preds[h] if isinstance(preds, (list, np.ndarray)) else preds 
                                  for name, preds in model_predictions.items()}
                
                weighted_pred = self._combine_predictions(step_predictions, self.model_weights)
                weighted_predictions.append(weighted_pred)
            
            return {
                'predictions': weighted_predictions,
                'model_predictions': model_predictions,
                'model_weights': self.model_weights.copy(),
                'horizon': horizon
            }
    
    def _recursive_predict(self, model, recent_data: pd.DataFrame, horizon: int) -> List[float]:
        """
        Generate multi-step predictions recursively.
        
        Args:
            model: Fitted base model
            recent_data: Recent feature data
            horizon: Number of steps to predict
            
        Returns:
            List of predictions
        """
        predictions = []
        current_features = recent_data.iloc[-1:].copy()
        
        for h in range(horizon):
            # Predict next step
            pred = model.predict(current_features)[0]
            predictions.append(pred)
            
            # Update features for next prediction
            # This is a simplified approach - in practice, you'd have domain-specific logic
            if 'lag_1' in current_features.columns:
                current_features['lag_1'] = pred
            
            # Add some noise to prevent overly confident long-term predictions
            if h > 0:
                noise_std = 0.01 * np.sqrt(h + 1)
                pred += np.random.normal(0, noise_std)
        
        return predictions
    
    def _get_recent_errors(self, model_name: str, window: int = 10) -> List[float]:
        """
        Get recent prediction errors for a model.
        
        Args:
            model_name: Name of the model
            window: Number of recent errors to return
            
        Returns:
            List of recent prediction errors
        """
        if model_name in self.model_performance:
            return self.model_performance[model_name][-window:]
        return []
    
    def _update_model_weights(self, uncertainties: Dict[str, float]) -> None:
        """
        Update model weights based on recent performance and uncertainties.
        
        Args:
            uncertainties: Dictionary of model uncertainties
        """
        # Calculate inverse uncertainty weights (lower uncertainty = higher weight)
        inverse_uncertainties = {}
        total_inverse_uncertainty = 0
        
        for model_name in self.base_models:
            if model_name in uncertainties and uncertainties[model_name] > 0:
                inv_unc = 1.0 / (uncertainties[model_name] + 1e-8)
                inverse_uncertainties[model_name] = inv_unc
                total_inverse_uncertainty += inv_unc
        
        if total_inverse_uncertainty > 0:
            # Update weights using exponential smoothing
            for model_name in self.base_models:
                if model_name in inverse_uncertainties:
                    new_weight = inverse_uncertainties[model_name] / total_inverse_uncertainty
                    # Smooth weight updates
                    self.model_weights[model_name] = (
                        (1 - self.weight_smoothing) * self.model_weights[model_name] +
                        self.weight_smoothing * new_weight
                    )
                else:
                    # Decay weights for models without predictions
                    self.model_weights[model_name] *= 0.9
        
        # Renormalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for model_name in self.model_weights:
                self.model_weights[model_name] /= total_weight
    
    def _combine_predictions(self, 
                           predictions: Dict[str, float], 
                           weights: Dict[str, float]) -> float:
        """
        Combine model predictions using weights.
        
        Args:
            predictions: Dictionary of model predictions
            weights: Dictionary of model weights
            
        Returns:
            Weighted average prediction
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_name, prediction in predictions.items():
            if model_name in weights and weights[model_name] > 0:
                weight = weights[model_name]
                weighted_sum += weight * prediction
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            # Fallback to simple average
            return np.mean(list(predictions.values()))
    
    def _calculate_ensemble_uncertainty(self,
                                      predictions: Dict[str, float],
                                      uncertainties: Dict[str, float],
                                      weights: Dict[str, float]) -> float:
        """
        Calculate ensemble prediction uncertainty.
        
        Args:
            predictions: Model predictions
            uncertainties: Model uncertainties
            weights: Model weights
            
        Returns:
            Ensemble uncertainty estimate
        """
        # Weighted average of individual uncertainties
        weighted_uncertainty = 0.0
        total_weight = 0.0
        
        for model_name in predictions:
            if model_name in weights and model_name in uncertainties:
                weight = weights[model_name]
                uncertainty = uncertainties[model_name]
                weighted_uncertainty += weight * uncertainty ** 2
                total_weight += weight
        
        if total_weight > 0:
            avg_uncertainty = np.sqrt(weighted_uncertainty / total_weight)
        else:
            avg_uncertainty = 0.1
        
        # Add disagreement between models
        if len(predictions) > 1:
            pred_values = list(predictions.values())
            disagreement = np.std(pred_values)
            total_uncertainty = np.sqrt(avg_uncertainty ** 2 + disagreement ** 2)
        else:
            total_uncertainty = avg_uncertainty
        
        return total_uncertainty
    
    def get_model_rankings(self) -> pd.DataFrame:
        """
        Get current model rankings based on performance and weights.
        
        Returns:
            DataFrame with model rankings
        """
        ranking_data = []
        
        for model_name, model_info in self.base_models.items():
            if not model_info['fitted']:
                continue
            
            recent_errors = self._get_recent_errors(model_name)
            
            ranking_data.append({
                'model': model_name,
                'weight': self.model_weights[model_name],
                'recent_mse': np.mean([e**2 for e in recent_errors]) if recent_errors else np.inf,
                'recent_mae': np.mean([abs(e) for e in recent_errors]) if recent_errors else np.inf,
                'prediction_count': len(recent_errors),
                'rank_score': self.model_weights[model_name] / (model_info['last_performance'] + 1e-8)
            })
        
        df = pd.DataFrame(ranking_data)
        if not df.empty:
            df = df.sort_values('rank_score', ascending=False).reset_index(drop=True)
            df['rank'] = df.index + 1
        
        return df
