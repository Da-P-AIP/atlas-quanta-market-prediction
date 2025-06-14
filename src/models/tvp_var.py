"""
TVP-VAR (Time-Varying Parameter Vector Autoregression) Model for Atlas Quanta

This module implements time-varying parameter VAR models for dynamic forecasting
with adaptive parameter adjustment based on changing market conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
from scipy import linalg
from scipy.stats import invwishart, multivariate_normal
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TVPVARModel:
    """
    Time-Varying Parameter Vector Autoregression Model.
    
    Implements Bayesian TVP-VAR with stochastic volatility for dynamic
    parameter estimation in financial time series forecasting.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize TVP-VAR model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.lags = self.config.get('lags', 2)
        self.n_draws = self.config.get('n_draws', 1000)
        self.n_burn = self.config.get('n_burn', 200)
        self.decay_factor = self.config.get('decay_factor', 0.99)
        
        # Hyperparameters for priors
        self.kappa_beta = self.config.get('kappa_beta', 0.01)  # State equation variance
        self.kappa_alpha = self.config.get('kappa_alpha', 0.01)  # Stochastic volatility
        
        # Model components
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Storage for model results
        self.parameters = {}
        self.volatility = {}
        self.forecasts = {}
        
        self.logger.info(f"TVP-VAR model initialized with {self.lags} lags")
    
    def fit(self, data: pd.DataFrame, target_cols: List[str] = None) -> Dict[str, any]:
        """
        Fit the TVP-VAR model on historical data.
        
        Args:
            data: DataFrame with time series data
            target_cols: List of column names to use as endogenous variables
            
        Returns:
            Dictionary with fitting results and diagnostics
        """
        self.logger.info(f"Fitting TVP-VAR model on {len(data)} observations")
        
        if target_cols is None:
            target_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Prepare data
        y = data[target_cols].dropna()
        if len(y) < self.lags + 10:
            raise ValueError(f"Insufficient data: need at least {self.lags + 10} observations")
        
        # Scale data
        y_scaled = pd.DataFrame(
            self.scaler.fit_transform(y),
            index=y.index,
            columns=y.columns
        )
        
        # Create lagged variables
        Y, X = self._create_var_matrices(y_scaled)
        
        # Estimate model using Kalman Filter with time-varying parameters
        results = self._estimate_tvp_var(Y, X)
        
        # Store results
        self.n_vars = Y.shape[1]
        self.n_obs = Y.shape[0]
        self.target_columns = target_cols
        self.parameters = results['parameters']
        self.volatility = results['volatility']
        self.log_likelihood = results['log_likelihood']
        self.is_fitted = True
        
        # Calculate model diagnostics
        diagnostics = self._calculate_diagnostics(Y, X, results)
        
        self.logger.info("TVP-VAR model fitting completed successfully")
        return {
            'parameters': results['parameters'],
            'volatility': results['volatility'],
            'diagnostics': diagnostics,
            'log_likelihood': results['log_likelihood']
        }
    
    def _create_var_matrices(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create VAR matrices Y and X from time series data.
        
        Args:
            data: Time series DataFrame
            
        Returns:
            Tuple of (Y, X) matrices for VAR estimation
        """
        n_obs, n_vars = data.shape
        
        # Create Y matrix (dependent variables)
        Y = data.iloc[self.lags:].values
        
        # Create X matrix (lagged variables + constant)
        X_list = []
        
        # Add constant
        X_list.append(np.ones((n_obs - self.lags, 1)))
        
        # Add lagged variables
        for lag in range(1, self.lags + 1):
            lagged_data = data.iloc[self.lags - lag:-lag].values
            X_list.append(lagged_data)
        
        X = np.hstack(X_list)
        
        return Y, X
    
    def _estimate_tvp_var(self, Y: np.ndarray, X: np.ndarray) -> Dict[str, any]:
        """
        Estimate TVP-VAR using Kalman Filter with time-varying parameters.
        
        Args:
            Y: Dependent variable matrix
            X: Independent variable matrix (including lags)
            
        Returns:
            Dictionary with parameter estimates and volatility
        """
        T, n = Y.shape
        k = X.shape[1]  # Number of regressors per equation
        
        # Initialize parameter storage
        beta_mean = np.zeros((T, n * k))  # Vectorized parameters
        beta_var = np.zeros((T, n * k, n * k))
        h_mean = np.zeros((T, n))  # Log volatilities
        sigma = np.zeros((T, n, n))  # Covariance matrices
        
        # Prior specifications
        beta_0_mean = np.zeros(n * k)
        beta_0_var = np.eye(n * k) * 10  # Diffuse prior
        h_0_mean = np.zeros(n)
        h_0_var = np.eye(n) * 10
        
        # Hyperparameters
        Q_beta = np.eye(n * k) * self.kappa_beta  # State evolution covariance
        Q_h = np.eye(n) * self.kappa_alpha  # Volatility evolution covariance
        
        # Initialize
        beta_mean[0] = beta_0_mean
        beta_var[0] = beta_0_var
        h_mean[0] = h_0_mean
        
        log_likelihood = 0
        
        # Kalman Filter loop
        for t in range(T):
            # Current observation
            y_t = Y[t, :]
            x_t = X[t, :]
            
            # Create design matrix for vectorized system
            X_t = np.kron(np.eye(n), x_t.reshape(1, -1))
            
            if t == 0:
                # Initial prediction
                beta_pred = beta_0_mean
                P_pred = beta_0_var
                h_pred = h_0_mean
            else:
                # Prediction step
                beta_pred = beta_mean[t-1]  # Random walk assumption
                P_pred = beta_var[t-1] + Q_beta
                h_pred = h_mean[t-1]  # Random walk for log volatilities
            
            # Current volatility
            sigma_t = np.diag(np.exp(h_pred))
            sigma[t] = sigma_t
            
            # Prediction error
            y_pred = X_t @ beta_pred
            v_t = y_t - y_pred
            
            # Prediction error covariance
            F_t = X_t @ P_pred @ X_t.T + sigma_t
            
            # Update step (if F_t is invertible)
            try:
                F_inv = linalg.inv(F_t)
                K_t = P_pred @ X_t.T @ F_inv
                
                # Updated estimates
                beta_mean[t] = beta_pred + K_t @ v_t
                beta_var[t] = P_pred - K_t @ X_t @ P_pred
                
                # Log likelihood contribution
                log_likelihood += -0.5 * (
                    n * np.log(2 * np.pi) + 
                    np.log(linalg.det(F_t)) + 
                    v_t.T @ F_inv @ v_t
                )
                
            except linalg.LinAlgError:
                # If F_t is singular, use prediction
                beta_mean[t] = beta_pred
                beta_var[t] = P_pred
                self.logger.warning(f"Singular covariance matrix at time {t}")
            
            # Update volatility (simplified approach)
            # In practice, this would use particle filter or MCMC
            if t > 0:
                # Simple exponential smoothing of squared residuals
                alpha_vol = 0.1
                log_squared_resid = np.log(np.maximum(v_t**2, 1e-8))
                h_mean[t] = (1 - alpha_vol) * h_mean[t-1] + alpha_vol * log_squared_resid
            else:
                h_mean[t] = h_pred
        
        # Reshape parameters back to matrix form
        parameters = {}
        for t in range(T):
            beta_matrix = beta_mean[t].reshape((k, n))
            parameters[t] = {
                'coefficients': beta_matrix,
                'covariance': sigma[t],
                'log_volatility': h_mean[t]
            }
        
        return {
            'parameters': parameters,
            'volatility': h_mean,
            'log_likelihood': log_likelihood,
            'beta_mean': beta_mean,
            'beta_var': beta_var
        }
    
    def _calculate_diagnostics(self, Y: np.ndarray, X: np.ndarray, results: Dict) -> Dict[str, float]:
        """
        Calculate model diagnostics and goodness-of-fit measures.
        
        Args:
            Y: Dependent variable matrix
            X: Independent variable matrix
            results: Model estimation results
            
        Returns:
            Dictionary with diagnostic statistics
        """
        T, n = Y.shape
        k = X.shape[1]
        
        # Calculate fitted values and residuals
        fitted_values = np.zeros_like(Y)
        residuals = np.zeros_like(Y)
        
        for t in range(T):
            x_t = X[t, :]
            beta_t = results['parameters'][t]['coefficients']
            fitted_values[t, :] = x_t @ beta_t
            residuals[t, :] = Y[t, :] - fitted_values[t, :]
        
        # R-squared (time-varying)
        ss_res = np.sum(residuals**2, axis=0)
        ss_tot = np.sum((Y - np.mean(Y, axis=0))**2, axis=0)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Information criteria
        n_params = n * k * T  # Approximate (parameters vary over time)
        aic = -2 * results['log_likelihood'] + 2 * n_params
        bic = -2 * results['log_likelihood'] + np.log(T) * n_params
        
        # Durbin-Watson statistic for residual autocorrelation
        dw_stats = []
        for i in range(n):
            resid_diff = np.diff(residuals[:, i])
            dw = np.sum(resid_diff**2) / np.sum(residuals[1:, i]**2)
            dw_stats.append(dw)
        
        # Parameter stability (variance of parameter changes)
        param_stability = []
        for i in range(n):
            for j in range(k):
                param_series = [results['parameters'][t]['coefficients'][j, i] for t in range(T)]
                param_stability.append(np.var(param_series))
        
        return {
            'r_squared': r_squared.tolist(),
            'mean_r_squared': np.mean(r_squared),
            'aic': aic,
            'bic': bic,
            'log_likelihood': results['log_likelihood'],
            'durbin_watson': dw_stats,
            'parameter_stability': np.mean(param_stability),
            'residual_std': np.std(residuals, axis=0).tolist()
        }
    
    def predict(self, 
                data: pd.DataFrame,
                horizon: int = 10,
                confidence_level: float = 0.95) -> Dict[str, any]:
        """
        Generate predictions using the fitted TVP-VAR model.
        
        Args:
            data: Recent data for prediction
            horizon: Number of periods to forecast
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.logger.info(f"Generating {horizon}-period predictions")
        
        # Prepare data
        recent_data = data[self.target_columns].tail(50)  # Use recent 50 observations
        if len(recent_data) < self.lags:
            raise ValueError(f"Need at least {self.lags} observations for prediction")
        
        # Scale data using fitted scaler
        recent_scaled = pd.DataFrame(
            self.scaler.transform(recent_data),
            index=recent_data.index,
            columns=recent_data.columns
        )
        
        # Get latest parameters (from last fitted period)
        latest_params = self.parameters[len(self.parameters) - 1]
        beta = latest_params['coefficients']
        sigma = latest_params['covariance']
        
        # Initialize prediction
        predictions = []
        prediction_vars = []
        
        # Use last observations as initial conditions
        last_obs = recent_scaled.tail(self.lags).values
        current_state = last_obs.flatten()
        
        for h in range(horizon):
            # Create prediction input
            if h == 0:
                # Use actual lagged values
                x_pred = np.concatenate([[1], current_state])  # Add constant
            else:
                # Use predicted values for longer horizons
                x_pred = np.concatenate([[1], 
                    np.concatenate([pred.flatten() for pred in predictions[-self.lags:]])])
            
            # Ensure correct dimensions
            x_pred = x_pred[:beta.shape[0]]
            
            # Point prediction
            y_pred = x_pred @ beta
            predictions.append(y_pred)
            
            # Prediction variance (approximation)
            pred_var = np.diag(sigma)
            prediction_vars.append(pred_var)
            
            # Update state for next prediction
            if h < self.lags - 1:
                current_state = np.concatenate([
                    y_pred.flatten(),
                    current_state[:-len(y_pred)]
                ])
        
        # Convert to arrays
        predictions = np.array(predictions)
        prediction_vars = np.array(prediction_vars)
        
        # Transform back to original scale
        predictions_original = self.scaler.inverse_transform(predictions)
        
        # Calculate confidence intervals (approximate)
        alpha = 1 - confidence_level
        z_score = 1.96  # Approximate for 95% confidence
        
        prediction_std = np.sqrt(prediction_vars)
        prediction_std_original = prediction_std * self.scaler.scale_
        
        lower_bound = predictions_original - z_score * prediction_std_original
        upper_bound = predictions_original + z_score * prediction_std_original
        
        # Create prediction DataFrame
        future_dates = pd.date_range(
            start=recent_data.index[-1] + pd.Timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        predictions_df = pd.DataFrame(
            predictions_original,
            index=future_dates,
            columns=self.target_columns
        )
        
        lower_df = pd.DataFrame(
            lower_bound,
            index=future_dates,
            columns=[f"{col}_lower" for col in self.target_columns]
        )
        
        upper_df = pd.DataFrame(
            upper_bound,
            index=future_dates,
            columns=[f"{col}_upper" for col in self.target_columns]
        )
        
        return {
            'predictions': predictions_df,
            'lower_bound': lower_df,
            'upper_bound': upper_df,
            'confidence_level': confidence_level,
            'prediction_variance': prediction_vars,
            'model_params': latest_params
        }
    
    def get_parameter_evolution(self) -> pd.DataFrame:
        """
        Get the evolution of model parameters over time.
        
        Returns:
            DataFrame with parameter evolution
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        param_data = []
        
        for t, params in self.parameters.items():
            beta = params['coefficients']
            log_vol = params['log_volatility']
            
            # Flatten coefficient matrix and add to record
            for i, col in enumerate(self.target_columns):
                for j in range(beta.shape[0]):
                    param_data.append({
                        'time_index': t,
                        'variable': col,
                        'lag_or_const': j,
                        'coefficient': beta[j, i],
                        'log_volatility': log_vol[i]
                    })
        
        return pd.DataFrame(param_data)
    
    def calculate_impulse_responses(self, 
                                  horizon: int = 20,
                                  shock_size: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Calculate impulse response functions for the latest time period.
        
        Args:
            horizon: Number of periods for impulse responses
            shock_size: Size of the shock (in standard deviations)
            
        Returns:
            Dictionary with impulse response matrices
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Use latest parameters
        latest_params = self.parameters[len(self.parameters) - 1]
        beta = latest_params['coefficients']
        sigma = latest_params['covariance']
        
        n_vars = len(self.target_columns)
        
        # Extract coefficient matrices for each lag
        coef_matrices = []
        for lag in range(self.lags):
            start_idx = 1 + lag * n_vars  # Skip constant
            end_idx = start_idx + n_vars
            if end_idx <= beta.shape[0]:
                coef_matrices.append(beta[start_idx:end_idx, :])
        
        # Create companion form
        companion_size = self.lags * n_vars
        companion = np.zeros((companion_size, companion_size))
        
        # Fill in coefficient matrices
        for i, coef_matrix in enumerate(coef_matrices):
            companion[:n_vars, i*n_vars:(i+1)*n_vars] = coef_matrix.T
        
        # Identity matrix for lags
        if self.lags > 1:
            companion[n_vars:, :-n_vars] = np.eye((self.lags-1) * n_vars)
        
        # Calculate impulse responses
        impulse_responses = {}
        
        for shock_var_idx in range(n_vars):
            # Create shock vector
            shock = np.zeros(companion_size)
            shock[shock_var_idx] = shock_size * np.sqrt(sigma[shock_var_idx, shock_var_idx])
            
            # Propagate shock
            responses = np.zeros((horizon, n_vars))
            current_state = shock.copy()
            
            for h in range(horizon):
                # Extract response for original variables
                responses[h, :] = current_state[:n_vars]
                
                # Update state
                current_state = companion @ current_state
            
            shock_var_name = self.target_columns[shock_var_idx]
            impulse_responses[shock_var_name] = responses
        
        return impulse_responses

    def forecast_error_variance_decomposition(self, horizon: int = 20) -> pd.DataFrame:
        """
        Calculate forecast error variance decomposition.
        
        Args:
            horizon: Forecast horizon for decomposition
            
        Returns:
            DataFrame with variance decomposition results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Calculate impulse responses
        impulse_responses = self.calculate_impulse_responses(horizon)
        
        n_vars = len(self.target_columns)
        fevd_results = []
        
        for target_var_idx, target_var in enumerate(self.target_columns):
            for h in range(1, horizon + 1):
                # Calculate cumulative squared impulse responses
                total_variance = 0
                variance_contributions = {}
                
                for shock_var in self.target_columns:
                    cumulative_response = np.sum(
                        impulse_responses[shock_var][:h, target_var_idx] ** 2
                    )
                    variance_contributions[shock_var] = cumulative_response
                    total_variance += cumulative_response
                
                # Convert to percentages
                for shock_var in self.target_columns:
                    if total_variance > 0:
                        contribution_pct = (
                            variance_contributions[shock_var] / total_variance * 100
                        )
                    else:
                        contribution_pct = 0
                    
                    fevd_results.append({
                        'target_variable': target_var,
                        'shock_variable': shock_var,
                        'horizon': h,
                        'contribution_percent': contribution_pct
                    })
        
        return pd.DataFrame(fevd_results)
