"""
Value at Risk (VaR) Calculator

Provides multiple methodologies for calculating VaR including:
- Historical simulation
- Parametric (normal distribution)
- Monte Carlo simulation
- Conditional VaR (Expected Shortfall)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging

class VaRCalculator:
    """
    Comprehensive Value at Risk calculator with multiple methodologies.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.confidence_levels = config.get('confidence_levels', [0.95, 0.99])
        self.window_size = config.get('var_window_size', 252)  # 1 year
        
    def calculate_historical_var(self, 
                                 returns: pd.Series,
                                 confidence_level: float = 0.95,
                                 window_size: Optional[int] = None) -> pd.Series:
        """
        Calculate VaR using historical simulation method.
        
        Args:
            returns: Time series of returns
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            window_size: Rolling window size
            
        Returns:
            Series with rolling VaR values
        """
        
        if window_size is None:
            window_size = self.window_size
        
        quantile_level = 1 - confidence_level
        
        # Calculate rolling quantile
        var_series = returns.rolling(window=window_size).quantile(quantile_level)
        
        return var_series
    
    def calculate_parametric_var(self,
                                returns: pd.Series,
                                confidence_level: float = 0.95,
                                window_size: Optional[int] = None) -> pd.Series:
        """
        Calculate VaR using parametric method (assumes normal distribution).
        """
        
        if window_size is None:
            window_size = self.window_size
        
        # Calculate rolling mean and std
        rolling_mean = returns.rolling(window=window_size).mean()
        rolling_std = returns.rolling(window=window_size).std()
        
        # Calculate z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # VaR = mean + z_score * std (z_score is negative)
        var_series = rolling_mean + z_score * rolling_std
        
        return var_series
    
    def calculate_conditional_var(self,
                                 returns: pd.Series,
                                 confidence_level: float = 0.95,
                                 window_size: Optional[int] = None) -> pd.Series:
        """
        Calculate Conditional VaR (Expected Shortfall).
        
        CVaR is the expected value of losses that exceed the VaR threshold.
        """
        
        if window_size is None:
            window_size = self.window_size
        
        # First calculate VaR
        var_series = self.calculate_historical_var(returns, confidence_level, window_size)
        
        # Calculate CVaR
        cvar_series = pd.Series(index=returns.index, dtype=float)
        
        for i in range(window_size, len(returns)):
            window_returns = returns.iloc[i-window_size:i]
            var_threshold = var_series.iloc[i]
            
            # Find returns worse than VaR
            tail_returns = window_returns[window_returns <= var_threshold]
            
            if len(tail_returns) > 0:
                cvar_series.iloc[i] = tail_returns.mean()
            else:
                cvar_series.iloc[i] = var_threshold
        
        return cvar_series
    
    def calculate_portfolio_var(self,
                               portfolio_returns: pd.DataFrame,
                               weights: np.ndarray,
                               confidence_level: float = 0.95,
                               method: str = 'historical') -> pd.Series:
        """
        Calculate portfolio VaR given individual asset returns and weights.
        
        Args:
            portfolio_returns: DataFrame with asset returns
            weights: Array of portfolio weights
            confidence_level: Confidence level
            method: 'historical', 'parametric', or 'monte_carlo'
            
        Returns:
            Series with portfolio VaR
        """
        
        # Calculate portfolio returns
        portfolio_return_series = (portfolio_returns * weights).sum(axis=1)
        
        if method == 'historical':
            return self.calculate_historical_var(portfolio_return_series, confidence_level)
        elif method == 'parametric':
            return self.calculate_parametric_var(portfolio_return_series, confidence_level)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def calculate_component_var(self,
                               portfolio_returns: pd.DataFrame,
                               weights: np.ndarray,
                               confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Calculate component VaR for each asset in the portfolio.
        
        Component VaR shows how much each asset contributes to the total portfolio VaR.
        """
        
        # Calculate covariance matrix
        cov_matrix = portfolio_returns.cov()
        
        # Calculate portfolio variance
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Calculate marginal VaR
        marginal_var = np.dot(cov_matrix, weights) / portfolio_vol
        
        # Calculate component VaR
        component_var = weights * marginal_var
        
        # Convert to percentage contribution
        component_var_pct = component_var / portfolio_vol
        
        result = pd.DataFrame({
            'asset': portfolio_returns.columns,
            'weight': weights,
            'marginal_var': marginal_var,
            'component_var': component_var,
            'component_var_pct': component_var_pct
        })
        
        return result
    
    def backtest_var_model(self,
                          returns: pd.Series,
                          var_forecasts: pd.Series,
                          confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Backtest VaR model performance.
        
        Args:
            returns: Actual returns
            var_forecasts: VaR forecasts
            confidence_level: Confidence level used for VaR
            
        Returns:
            Dictionary with backtesting metrics
        """
        
        # Align series
        common_index = returns.index.intersection(var_forecasts.index)
        actual_returns = returns.loc[common_index]
        forecasts = var_forecasts.loc[common_index]
        
        # Count violations (actual return worse than VaR)
        violations = actual_returns < forecasts
        violation_count = violations.sum()
        total_observations = len(actual_returns)
        
        # Calculate violation rate
        violation_rate = violation_count / total_observations
        expected_violation_rate = 1 - confidence_level
        
        # Kupiec test for unconditional coverage
        # H0: violation rate = expected rate
        if violation_count > 0:
            lr_uc = 2 * (
                violation_count * np.log(violation_rate / expected_violation_rate) +
                (total_observations - violation_count) * np.log((1 - violation_rate) / (1 - expected_violation_rate))
            )
        else:
            lr_uc = 2 * total_observations * np.log(1 - expected_violation_rate)
        
        # Calculate average loss when violation occurs
        violation_losses = actual_returns[violations] - forecasts[violations]
        avg_violation_loss = violation_losses.mean() if len(violation_losses) > 0 else 0
        
        results = {
            'total_observations': total_observations,
            'violation_count': violation_count,
            'violation_rate': violation_rate,
            'expected_violation_rate': expected_violation_rate,
            'kupiec_lr_statistic': lr_uc,
            'kupiec_p_value': 1 - stats.chi2.cdf(lr_uc, df=1),
            'avg_violation_loss': avg_violation_loss,
            'max_violation_loss': violation_losses.min() if len(violation_losses) > 0 else 0
        }
        
        return results