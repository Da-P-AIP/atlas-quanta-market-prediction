"""Atlas Quanta: Multi-dimensional Market Prediction System"""

__version__ = "1.0.0"
__author__ = "Atlas Quanta Team"
__email__ = "team@atlas-quanta.com"

from .clustering.investor_clustering import InvestorClusteringAnalyzer
from .indicators.vpin import VPINCalculator
from .models.tvp_var import TVPVARModel
from .models.dma import DynamicModelAveraging
from .core.predictor import MarketPredictor

__all__ = [
    "InvestorClusteringAnalyzer",
    "VPINCalculator", 
    "TVPVARModel",
    "DynamicModelAveraging",
    "MarketPredictor"
]