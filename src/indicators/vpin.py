"""
VPIN (Volume-Synchronized Probability of Informed Trading) Implementation

Core Atlas Quanta indicator focusing on volume analysis rather than price movements.
VPIN detects informed trading activity and provides early warning signals for
volatility spikes and flash crashes (15-30 minutes advance warning).

Reference: Easley, D., López de Prado, M. M., & O'Hara, M. (2012)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from dataclasses import dataclass
from scipy.stats import norm
import warnings
from loguru import logger


@dataclass
class VPINResult:
    """VPIN calculation result"""
    timestamp: pd.Timestamp
    vpin_value: float
    volume_bucket: int
    buy_volume: float
    sell_volume: float
    total_volume: float
    imbalance: float
    volatility_estimate: float
    alert_level: str


class VPINCalculator:
    """
    Volume-Synchronized Probability of Informed Trading Calculator
    
    This implementation follows the Atlas Quanta methodology:
    1. Volume-based bucketing (not time-based)
    2. Order flow imbalance calculation using tick rule
    3. Rolling VPIN estimation with dynamic windows
    4. Early warning system for volatility events
    """
    
    def __init__(self, 
                 bucket_size: int = 50,
                 sample_window: int = 50,
                 alert_threshold: float = 0.8):
        """
        Initialize VPIN calculator
        
        Args:
            bucket_size: Volume per bucket (e.g., 50 = aggregate every 50 volume units)
            sample_window: Number of buckets for VPIN calculation
            alert_threshold: VPIN threshold for volatility alerts (0-1)
        """
        self.bucket_size = bucket_size
        self.sample_window = sample_window
        self.alert_threshold = alert_threshold
        
        # Storage for calculations
        self.volume_buckets: List[dict] = []
        self.vpin_history: List[VPINResult] = []
        
        # Current bucket accumulator
        self.current_bucket = {
            'volume': 0.0,
            'buy_volume': 0.0,
            'sell_volume': 0.0,
            'trades': 0,
            'start_time': None,
            'end_time': None,
            'price_start': None,
            'price_end': None
        }
        
        logger.info(f"VPIN Calculator initialized: bucket_size={bucket_size}, "
                   f"window={sample_window}, threshold={alert_threshold}")
    
    def classify_trade_direction(self, 
                               price: float, 
                               volume: float,
                               bid: Optional[float] = None,
                               ask: Optional[float] = None,
                               last_price: Optional[float] = None) -> str:
        """
        Classify trade as buy or sell using tick rule or bid-ask spread
        
        Args:
            price: Trade price
            volume: Trade volume
            bid: Current bid price (optional)
            ask: Current ask price (optional) 
            last_price: Previous trade price for tick rule
            
        Returns:
            'buy', 'sell', or 'neutral'
        """
        # Method 1: Bid-Ask classification (most accurate)
        if bid is not None and ask is not None:
            mid_price = (bid + ask) / 2
            if price >= mid_price:
                return 'buy'
            else:
                return 'sell'
        
        # Method 2: Tick rule (price change direction)
        if last_price is not None:
            if price > last_price:
                return 'buy'
            elif price < last_price:
                return 'sell'
            else:
                return 'neutral'
        
        # Default: assume neutral if no classification possible
        return 'neutral'
    
    def update_bucket(self, 
                     timestamp: pd.Timestamp,
                     price: float,
                     volume: float,
                     bid: Optional[float] = None,
                     ask: Optional[float] = None) -> Optional[VPINResult]:
        """
        Update current volume bucket with new trade data
        
        Args:
            timestamp: Trade timestamp
            price: Trade price
            volume: Trade volume
            bid: Bid price (optional)
            ask: Ask price (optional)
            
        Returns:
            VPINResult if bucket is complete and VPIN calculated, None otherwise
        """
        # Classify trade direction
        last_price = None
        if self.current_bucket['price_end'] is not None:
            last_price = self.current_bucket['price_end']
            
        direction = self.classify_trade_direction(price, volume, bid, ask, last_price)
        
        # Initialize bucket if empty
        if self.current_bucket['start_time'] is None:
            self.current_bucket['start_time'] = timestamp
            self.current_bucket['price_start'] = price
        
        # Update bucket data
        self.current_bucket['volume'] += volume
        self.current_bucket['trades'] += 1
        self.current_bucket['end_time'] = timestamp
        self.current_bucket['price_end'] = price
        
        # Accumulate buy/sell volumes
        if direction == 'buy':
            self.current_bucket['buy_volume'] += volume
        elif direction == 'sell':
            self.current_bucket['sell_volume'] += volume
        else:
            # Neutral trades split 50/50
            self.current_bucket['buy_volume'] += volume / 2
            self.current_bucket['sell_volume'] += volume / 2
        
        # Check if bucket is complete
        if self.current_bucket['volume'] >= self.bucket_size:
            return self._complete_bucket()
        
        return None
    
    def _complete_bucket(self) -> Optional[VPINResult]:
        """
        Complete current bucket and calculate VPIN if enough buckets available
        
        Returns:
            VPINResult if VPIN calculated, None otherwise
        """
        # Store completed bucket
        self.volume_buckets.append(self.current_bucket.copy())
        
        # Reset current bucket
        self.current_bucket = {
            'volume': 0.0,
            'buy_volume': 0.0, 
            'sell_volume': 0.0,
            'trades': 0,
            'start_time': None,
            'end_time': None,
            'price_start': None,
            'price_end': None
        }
        
        # Calculate VPIN if we have enough buckets
        if len(self.volume_buckets) >= self.sample_window:
            return self._calculate_vpin()
        
        return None
    
    def _calculate_vpin(self) -> VPINResult:
        """
        Calculate VPIN for the current window of volume buckets
        
        Returns:
            VPINResult with VPIN value and related metrics
        """
        # Get last N buckets for calculation
        recent_buckets = self.volume_buckets[-self.sample_window:]
        
        # Calculate volume imbalances for each bucket
        imbalances = []
        total_volume = 0
        total_buy_volume = 0
        total_sell_volume = 0
        
        for bucket in recent_buckets:
            volume_imbalance = abs(bucket['buy_volume'] - bucket['sell_volume'])
            imbalances.append(volume_imbalance)
            
            total_volume += bucket['volume']
            total_buy_volume += bucket['buy_volume']
            total_sell_volume += bucket['sell_volume']
        
        # VPIN = Average(|Buy Volume - Sell Volume|) / Average(Total Volume)
        avg_imbalance = np.mean(imbalances)
        avg_volume = total_volume / len(recent_buckets)
        
        vpin = avg_imbalance / avg_volume if avg_volume > 0 else 0
        
        # Estimate volatility based on price changes in buckets
        price_changes = []
        for bucket in recent_buckets:
            if bucket['price_start'] and bucket['price_end']:
                pct_change = abs(bucket['price_end'] - bucket['price_start']) / bucket['price_start']
                price_changes.append(pct_change)
        
        volatility_estimate = np.std(price_changes) if price_changes else 0
        
        # Determine alert level
        alert_level = self._get_alert_level(vpin)
        
        # Create result
        result = VPINResult(
            timestamp=recent_buckets[-1]['end_time'],
            vpin_value=vpin,
            volume_bucket=len(self.volume_buckets),
            buy_volume=total_buy_volume,
            sell_volume=total_sell_volume,
            total_volume=total_volume,
            imbalance=total_buy_volume - total_sell_volume,
            volatility_estimate=volatility_estimate,
            alert_level=alert_level
        )
        
        # Store in history
        self.vpin_history.append(result)
        
        # Log alerts
        if alert_level in ['HIGH', 'CRITICAL']:
            logger.warning(f"VPIN Alert: {alert_level} - VPIN={vpin:.3f} at {result.timestamp}")
        
        return result
    
    def _get_alert_level(self, vpin: float) -> str:
        """
        Determine alert level based on VPIN value
        
        Args:
            vpin: Current VPIN value
            
        Returns:
            Alert level: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'
        """
        if vpin < self.alert_threshold * 0.5:
            return 'LOW'
        elif vpin < self.alert_threshold * 0.75:
            return 'MEDIUM'
        elif vpin < self.alert_threshold:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def get_current_vpin(self) -> Optional[float]:
        """
        Get the most recent VPIN value
        
        Returns:
            Current VPIN value or None if not enough data
        """
        if self.vpin_history:
            return self.vpin_history[-1].vpin_value
        return None
    
    def get_vpin_history(self, periods: Optional[int] = None) -> pd.DataFrame:
        """
        Get VPIN calculation history as DataFrame
        
        Args:
            periods: Number of recent periods to return (None for all)
            
        Returns:
            DataFrame with VPIN history
        """
        if not self.vpin_history:
            return pd.DataFrame()
        
        history = self.vpin_history[-periods:] if periods else self.vpin_history
        
        data = []
        for result in history:
            data.append({
                'timestamp': result.timestamp,
                'vpin': result.vpin_value,
                'volume_bucket': result.volume_bucket,
                'buy_volume': result.buy_volume,
                'sell_volume': result.sell_volume,
                'total_volume': result.total_volume,
                'imbalance': result.imbalance,
                'volatility_estimate': result.volatility_estimate,
                'alert_level': result.alert_level
            })
        
        return pd.DataFrame(data).set_index('timestamp')
    
    def detect_flash_crash_risk(self, lookback_periods: int = 10) -> dict:
        """
        Detect potential flash crash risk based on VPIN pattern analysis
        
        Atlas Quanta methodology: Monitor for sustained high VPIN levels
        which historically precede volatility events by 15-30 minutes
        
        Args:
            lookback_periods: Number of recent periods to analyze
            
        Returns:
            Dictionary with risk assessment
        """
        if len(self.vpin_history) < lookback_periods:
            return {'risk_level': 'INSUFFICIENT_DATA', 'confidence': 0.0}
        
        recent_vpins = [r.vpin_value for r in self.vpin_history[-lookback_periods:]]
        
        # Calculate risk indicators
        avg_vpin = np.mean(recent_vpins)
        vpin_trend = np.polyfit(range(len(recent_vpins)), recent_vpins, 1)[0]
        high_vpin_ratio = sum(1 for v in recent_vpins if v > self.alert_threshold) / len(recent_vpins)
        
        # Risk scoring
        risk_score = 0
        risk_factors = []
        
        # High average VPIN
        if avg_vpin > self.alert_threshold:
            risk_score += 3
            risk_factors.append(f"High average VPIN ({avg_vpin:.3f})")
        
        # Rising VPIN trend
        if vpin_trend > 0.01:
            risk_score += 2
            risk_factors.append(f"Rising VPIN trend ({vpin_trend:.4f})")
        
        # High proportion of alerts
        if high_vpin_ratio > 0.5:
            risk_score += 2
            risk_factors.append(f"Frequent alerts ({high_vpin_ratio:.1%})")
        
        # Volatility clustering
        recent_volatility = [r.volatility_estimate for r in self.vpin_history[-lookback_periods:]]
        if np.mean(recent_volatility) > np.std(recent_volatility):
            risk_score += 1
            risk_factors.append("Volatility clustering detected")
        
        # Determine risk level
        if risk_score >= 6:
            risk_level = 'CRITICAL'
            confidence = 0.9
        elif risk_score >= 4:
            risk_level = 'HIGH'
            confidence = 0.7
        elif risk_score >= 2:
            risk_level = 'MEDIUM'
            confidence = 0.5
        else:
            risk_level = 'LOW'
            confidence = 0.3
        
        return {
            'risk_level': risk_level,
            'confidence': confidence,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'current_vpin': recent_vpins[-1],
            'avg_vpin': avg_vpin,
            'vpin_trend': vpin_trend
        }
