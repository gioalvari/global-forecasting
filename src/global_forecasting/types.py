"""Type definitions for the global forecasting library."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureDefinition:
    """Definition of a feature to be generated."""

    name: str
    feature_type: str
    parameters: Dict[str, Any]


@dataclass
class ForecastResult:
    """Result of a forecasting operation."""

    predictions: np.ndarray
    confidence_intervals: Optional[np.ndarray]
    feature_importance: Dict[str, float]
    metrics: Dict[str, float]

    def to_dataframe(
        self, dates: pd.DatetimeIndex, series_ids: pd.Index
    ) -> pd.DataFrame:
        """Convert forecast results to a pandas DataFrame."""
        df = pd.DataFrame(
            self.predictions,
            index=pd.MultiIndex.from_product(
                [series_ids, dates], names=["unique_id", "ds"]
            ),
            columns=["forecast"],
        )

        if self.confidence_intervals is not None:
            df["lower_bound"] = self.confidence_intervals[:, 0]
            df["upper_bound"] = self.confidence_intervals[:, 1]

        return df
