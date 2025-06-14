"""
Position Sizing Module for Atlas Quanta

Implements various position sizing strategies including Kelly Criterion,
CVaR optimization, and dynamic risk-based position sizing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

class PositionSizer:
    """
    Advanced position sizing calculator using multiple methodologies.
    
    Supports:
    - Kelly Criterion
    - CVaR (Conditional Value at Risk) optimization
    - Fixed fractional
    - Volatility-based sizing
    - Maximum drawdown constraints
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the position sizer.
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.max_position_size = self.config.get('max_position_size', 0.1)  # 10% max
        self.min_position_size = self.config.get('min_position_size', 0.001)  # 0.1% min
        self.target_volatility = self.config.get('target_volatility', 0.15)  # 15% annual vol
        self.max_leverage = self.config.get('max_leverage', 1.0)  # No leverage by default
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)  # 2% risk-free rate
        self.confidence_level = self.config.get('confidence_level', 0.95)  # 95% confidence
        
        self.logger.info("PositionSizer initialized successfully")
    
    def kelly_criterion(self, 
                       expected_return: float, 
                       win_rate: float, 
                       avg_win: float, 
                       avg_loss: float) -> float:
        """
        Calculate Kelly Criterion position size.
        
        Args:
            expected_return: Expected return of the strategy
            win_rate: Probability of winning trades
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)
            
        Returns:
            Optimal position size as fraction of capital
        """
        try:
            if win_rate <= 0 or win_rate >= 1:
                self.logger.warning(f"Invalid win rate: {win_rate}")
                return self.min_position_size
            
            if avg_loss <= 0:
                self.logger.warning(f"Invalid average loss: {avg_loss}")
                return self.min_position_size
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply constraints
            kelly_fraction = max(0, kelly_fraction)  # No negative positions
            kelly_fraction = min(kelly_fraction, self.max_position_size)
            kelly_fraction = max(kelly_fraction, self.min_position_size)
            
            self.logger.debug(f"Kelly fraction calculated: {kelly_fraction:.4f}")
            return kelly_fraction
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly criterion: {e}")
            return self.min_position_size
    
    def volatility_targeting(self, 
                           returns: pd.Series, 
                           target_vol: float = None) -> float:
        """
        Calculate position size based on volatility targeting.
        
        Args:
            returns: Historical returns series
            target_vol: Target portfolio volatility (annualized)
            
        Returns:
            Position size to achieve target volatility
        """
        if target_vol is None:
            target_vol = self.target_volatility
        
        try:
            # Calculate realized volatility (annualized)
            realized_vol = returns.std() * np.sqrt(252)
            
            if realized_vol <= 0:
                self.logger.warning("Zero or negative volatility detected")
                return self.min_position_size
            
            # Position size = target_vol / realized_vol
            position_size = target_vol / realized_vol
            
            # Apply constraints
            position_size = min(position_size, self.max_position_size)
            position_size = max(position_size, self.min_position_size)
            
            self.logger.debug(f"Volatility-targeted position size: {position_size:.4f}")
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error in volatility targeting: {e}")
            return self.min_position_size
    
    def cvar_optimization(self, 
                         returns: pd.Series, 
                         confidence_level: float = None) -> float:
        """
        Calculate position size using CVaR (Conditional Value at Risk) optimization.
        
        Args:
            returns: Historical returns series
            confidence_level: Confidence level for CVaR calculation
            
        Returns:
            Optimal position size based on CVaR
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        try:
            # Calculate VaR and CVaR
            var_percentile = (1 - confidence_level) * 100
            var = np.percentile(returns, var_percentile)
            
            # CVaR is the expected value of returns below VaR
            cvar_returns = returns[returns <= var]
            if len(cvar_returns) == 0:
                cvar = var
            else:
                cvar = cvar_returns.mean()
            
            if cvar >= 0:  # If CVaR is positive, use minimum position
                return self.min_position_size
            
            # Position size inversely related to CVaR risk
            # Higher negative CVaR -> smaller position
            max_acceptable_cvar = -0.05  # -5% max daily loss
            position_size = min(abs(max_acceptable_cvar / cvar), self.max_position_size)
            position_size = max(position_size, self.min_position_size)
            
            self.logger.debug(f"CVaR-optimized position size: {position_size:.4f}")
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error in CVaR optimization: {e}")
            return self.min_position_size
    
    def fixed_fractional(self, 
                        fraction: float = 0.02) -> float:
        """
        Simple fixed fractional position sizing.
        
        Args:
            fraction: Fixed fraction of capital to risk
            
        Returns:
            Fixed position size
        """
        fraction = max(fraction, self.min_position_size)
        fraction = min(fraction, self.max_position_size)
        
        self.logger.debug(f"Fixed fractional position size: {fraction:.4f}")
        return fraction
    
    def maximum_drawdown_constraint(self, 
                                  returns: pd.Series, 
                                  max_dd_threshold: float = 0.2) -> float:
        """
        Calculate position size with maximum drawdown constraint.
        
        Args:
            returns: Historical returns series
            max_dd_threshold: Maximum acceptable drawdown (e.g., 0.2 for 20%)
            
        Returns:
            Position size constrained by maximum drawdown
        """
        try:
            # Calculate maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            if max_drawdown <= 0:
                return self.max_position_size
            
            # Scale position size to keep expected max drawdown below threshold
            position_scale = max_dd_threshold / max_drawdown
            position_size = min(position_scale, self.max_position_size)
            position_size = max(position_size, self.min_position_size)
            
            self.logger.debug(f"Max DD constrained position size: {position_size:.4f}")
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error in max drawdown constraint: {e}")
            return self.min_position_size
    
    def optimal_portfolio_weights(self, 
                                 expected_returns: np.ndarray, 
                                 covariance_matrix: np.ndarray, 
                                 risk_aversion: float = 3.0) -> np.ndarray:
        """
        Calculate optimal portfolio weights using mean-variance optimization.
        
        Args:
            expected_returns: Array of expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            risk_aversion: Risk aversion parameter (higher = more conservative)
            
        Returns:
            Optimal portfolio weights
        """
        try:
            n_assets = len(expected_returns)
            
            # Objective function: maximize utility = return - (risk_aversion/2) * variance
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
                utility = portfolio_return - (risk_aversion / 2) * portfolio_variance
                return -utility  # Negative because we minimize
            
            # Constraints: weights sum to 1, all weights >= 0
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
            ]
            
            # Bounds: each weight between 0 and max_position_size
            bounds = [(0, self.max_position_size) for _ in range(n_assets)]
            
            # Initial guess: equal weights
            x0 = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )
            
            if result.success:
                weights = result.x
                # Ensure minimum position constraints
                weights = np.maximum(weights, self.min_position_size)
                # Renormalize
                weights = weights / np.sum(weights)
                
                self.logger.debug(f"Optimal weights calculated: {weights}")
                return weights
            else:
                self.logger.warning("Optimization failed, using equal weights")
                return np.array([1/n_assets] * n_assets)
                
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            n_assets = len(expected_returns)
            return np.array([1/n_assets] * n_assets)
    
    def calculate_position_size(self, 
                              symbol: str,
                              strategy_data: Dict,
                              method: str = 'kelly',
                              portfolio_context: Dict = None) -> Dict:
        """
        Calculate position size using specified method.
        
        Args:
            symbol: Trading symbol
            strategy_data: Dictionary containing strategy performance data
            method: Position sizing method ('kelly', 'vol_target', 'cvar', 'fixed')
            portfolio_context: Additional portfolio context data
            
        Returns:
            Dictionary with position sizing results
        """
        try:
            results = {
                'symbol': symbol,
                'method': method,
                'position_size': self.min_position_size,
                'confidence': 0.5,
                'risk_metrics': {}
            }
            
            if 'returns' not in strategy_data:
                self.logger.warning(f"No returns data for {symbol}")
                return results
            
            returns = strategy_data['returns']
            
            if method == 'kelly':
                win_rate = strategy_data.get('win_rate', 0.5)
                avg_win = strategy_data.get('avg_win', 0.02)
                avg_loss = strategy_data.get('avg_loss', 0.02)
                expected_return = strategy_data.get('expected_return', 0.0)
                
                position_size = self.kelly_criterion(expected_return, win_rate, avg_win, avg_loss)
                
            elif method == 'vol_target':
                position_size = self.volatility_targeting(returns)
                
            elif method == 'cvar':
                position_size = self.cvar_optimization(returns)
                
            elif method == 'fixed':
                fraction = strategy_data.get('fixed_fraction', 0.02)
                position_size = self.fixed_fractional(fraction)
                
            elif method == 'max_dd':
                max_dd_threshold = strategy_data.get('max_dd_threshold', 0.2)
                position_size = self.maximum_drawdown_constraint(returns, max_dd_threshold)
                
            else:
                self.logger.warning(f"Unknown method {method}, using fixed fractional")
                position_size = self.fixed_fractional()
            
            # Apply leverage constraint
            position_size = min(position_size, self.max_leverage)
            
            # Calculate risk metrics
            if len(returns) > 0:
                results['risk_metrics'] = {
                    'volatility': returns.std() * np.sqrt(252),
                    'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                    'max_drawdown': self._calculate_max_drawdown(returns),
                    'var_95': np.percentile(returns, 5),
                    'expected_return': returns.mean() * 252
                }
            
            results['position_size'] = position_size
            results['confidence'] = min(1.0, len(returns) / 252)  # More data = higher confidence
            
            self.logger.info(f"Position size for {symbol}: {position_size:.4f} using {method}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return {
                'symbol': symbol,
                'method': method,
                'position_size': self.min_position_size,
                'confidence': 0.0,
                'risk_metrics': {}
            }
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if returns.std() == 0:
            return 0.0
        excess_returns = returns.mean() - self.risk_free_rate / 252
        return excess_returns / returns.std() * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def batch_position_sizing(self, 
                             strategies_data: Dict[str, Dict],
                             method: str = 'kelly') -> pd.DataFrame:
        """
        Calculate position sizes for multiple strategies/symbols.
        
        Args:
            strategies_data: Dictionary mapping symbols to strategy data
            method: Position sizing method to use
            
        Returns:
            DataFrame with position sizing results
        """
        results = []
        
        for symbol, data in strategies_data.items():
            result = self.calculate_position_size(symbol, data, method)
            results.append(result)
        
        df = pd.DataFrame(results)
        
        # Normalize position sizes to ensure they don't exceed 100% total
        if not df.empty and 'position_size' in df.columns:
            total_position = df['position_size'].sum()
            if total_position > 1.0:
                self.logger.warning(f"Total position exceeds 100%, normalizing from {total_position:.2f}")
                df['position_size'] = df['position_size'] / total_position
                df['normalized'] = True
            else:
                df['normalized'] = False
        
        return df
    
    def risk_parity_weights(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate risk parity (equal risk contribution) weights.
        
        Args:
            covariance_matrix: Asset covariance matrix
            
        Returns:
            Risk parity weights
        """
        try:
            n_assets = covariance_matrix.shape[0]
            
            # Objective: minimize sum of squared differences in risk contributions
            def objective(weights):
                portfolio_var = np.dot(weights.T, np.dot(covariance_matrix, weights))
                marginal_contrib = np.dot(covariance_matrix, weights)
                contrib = weights * marginal_contrib / portfolio_var
                target_contrib = 1.0 / n_assets
                return np.sum((contrib - target_contrib) ** 2)
            
            # Constraints
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]
            
            # Initial guess
            x0 = np.array([1/n_assets] * n_assets)
            
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                return result.x
            else:
                self.logger.warning("Risk parity optimization failed, using equal weights")
                return np.array([1/n_assets] * n_assets)
                
        except Exception as e:
            self.logger.error(f"Error in risk parity calculation: {e}")
            n_assets = covariance_matrix.shape[0]
            return np.array([1/n_assets] * n_assets)
