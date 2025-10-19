"""Global Forecasting Library.

A library for training machine learning models on multiple time series data
with support for zero-shot forecasting.
"""

from .forecaster import GlobalForecaster
from .types import FeatureDefinition, ForecastResult

__version__ = "0.1.0"
__all__ = ["GlobalForecaster", "FeatureDefinition", "ForecastResult"]
