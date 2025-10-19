"""Main forecaster class implementing the global forecasting model."""

from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from dateutil import relativedelta

from .utils import get_period
from logging import getLogger
from .metrics import rmse, mae, mape, smape, mase

logger = getLogger(__name__)

class GlobalForecaster:
    """Forecaster for multiple time series using a single global model."""

    def __init__(
        self,
        model: BaseEstimator,
        freq: Optional[str] = None,
        lags: List[int] = None,
        static_features: Optional[List[str]] = None,
        dynamic_features: Optional[List[str]] = None,
        debug: bool = False,
    ) -> None:
        """Initialize the global forecaster.

        Args:
            model: Sklearn-compatible model for forecasting
            freq: Frequency string for time series
            lags: List of lag values to use as features
            static_features: List of static feature column names
            dynamic_features: List of dynamic feature column names
        """
        self.model = model
        self.lags = sorted(lags) if lags is not None else []
        self.static_features = static_features or []
        self.dynamic_features = dynamic_features or []

        # Will be set during fit
        self.fitted_model: Optional[BaseEstimator] = None
        self.feature_names_: List[str] = []
        self._freq: Optional[str] = freq
        self.debug = debug
        if self.debug:
            logger.setLevel("DEBUG")
            logger.debug("GlobalForecaster initialized with parameters:")
            logger.debug(f"  Model: {model}")
            logger.debug(f"  Frequency: {freq}")
            logger.debug(f"  Lags: {lags}")
            logger.debug(f"  Static Features: {static_features}")
            logger.debug(f"  Dynamic Features: {dynamic_features}")

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "y",
        ds_col: str = "ds",
        unique_id_col: str = "unique_id",
    ) -> "GlobalForecaster":
        """Fit the forecaster on multiple time series.

        Args:
            df: DataFrame with time series data
            target_col: Name of target column (default: 'y')
            ds_col: Name of datetime column (default: 'ds')
            unique_id_col: Name of series ID column (default: 'unique_id')

        Returns:
            self: Trained forecaster
        """

        df = df.copy()
        # Validate input
        required_cols = [ds_col, target_col, unique_id_col]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")

        # infer frequency if not provided
        if self._freq is None:
            self._freq = pd.infer_freq(df[ds_col])

        if len(self.lags + self.static_features + self.dynamic_features) == 0:
            raise ValueError(
                "At least one of lags, static_features, or dynamic_features must be provided"
            )

        if len(self.lags) > 0:
            df = self._generate_lags(
                df, ds_col=ds_col, unique_id_col=unique_id_col, target_col=target_col
            )
            self._store_history_y_df(
                df, ds_col=ds_col, unique_id_col=unique_id_col, target_col=target_col
            )

        feature_cols = [f"{target_col}_lag_{lag}" for lag in self.lags]
        feature_cols.extend(self.static_features)
        feature_cols.extend(self.dynamic_features)
        self.feature_names_ = feature_cols

        df = df.dropna(subset=[target_col])
        X = df[feature_cols].values
        y = df[target_col].values

        # Fit the model
        self.fitted_model = self.model.fit(X, y)

        if self.debug:
            logger.debug("Model fitted successfully.")
            logger.debug(f"Features used for training: {self.feature_names_}")

        return self

    def _store_history_y_df(
        self, df: pd.DataFrame, unique_id_col: str, ds_col: str, target_col: str
    ) -> None:
        """Save the history of the target variable."""

        max_lag = max(self.lags)
        max_date = df[ds_col].max()
        cutoff_date = max_date - get_period(self._freq, n=max_lag)
        self._y_history = df[df[ds_col] >= cutoff_date][
            [unique_id_col, ds_col, target_col]
        ]

    def _generate_lags(
        self, df: pd.DataFrame, ds_col: str, unique_id_col: str, target_col: str
    ) -> pd.DataFrame:
        """Generate lag features of the target variable, respecting the time frequency and series ID.

        Args:
            df: DataFrame with time series data.
            ds_col: Name of the date column.
            unique_id_col: Name of the unique ID column.
            target_col: Name of the target column.

        Returns:
            DataFrame with lag features.

        Raises:
            ValueError: If frequency is not set.
        """

        if self._freq is None:
            raise ValueError("Frequency must be set to generate lags")

        df = df.sort_values([unique_id_col, ds_col]).copy()

        for lag in self.lags:
            df[f"{target_col}_lag_{lag}"] = df.groupby(unique_id_col)[target_col].shift(lag)

        return df

    def _update_with_predictions(
        self,
        df: pd.DataFrame,
        predictions: pd.DataFrame,
        unique_id_col: str,
        ds_col: str,
    ) -> pd.DataFrame:
        """Update the main DataFrame with the latest predictions for lag generation.

        Args:
            df: Main DataFrame with time series data.
            predictions: DataFrame with the latest predictions containing
                columns [unique_id_col, ds_col, target_col].
            unique_id_col: Name of the unique ID column.
            ds_col: Name of the date column.
            target_col: Name of the target column.

        Returns:
            Updated DataFrame with new predictions included.
        """

        # Set multi-index for both DataFrames
        df = df.set_index([ds_col, unique_id_col])
        predictions = predictions.set_index([ds_col, unique_id_col])

        # Update values
        df.update(predictions)

        return df.reset_index()

    def _create_X_df_from_history(
        self, horizon: int, unique_id_col: str, ds_col: str, target_col: str
    ) -> pd.DataFrame:
        """
        Create a DataFrame for prediction from the stored history of y.

        Args:
            horizon: Number of future steps to generate
            unique_id_col: Name of the unique identifier column
            ds_col: Name of the date/timestamp column
            target_col: Name of the target column

        Returns:
            DataFrame with n rows per unique_id, where n = horizon,
            with incremented ds_col and target_col set to NaN.
            Includes only unique_id at the maximum date.
        """
        max_date_absolute = self._y_history[ds_col].max()
        unique_ids = self._y_history[unique_id_col].unique()
        n_ids = len(unique_ids)

        unique_ids_repeated = np.repeat(unique_ids, horizon)

        periods = np.tile(np.arange(1, horizon + 1), n_ids)

        future_dates = np.array(
            [max_date_absolute + get_period(self._freq, n=p) for p in periods]
        )

        X_df = pd.DataFrame(
            {
                unique_id_col: unique_ids_repeated,
                ds_col: future_dates,
                target_col: np.nan,
            }
        )

        X_df = X_df.sort_values([unique_id_col, ds_col]).reset_index(drop=True)

        return X_df

    def predict(
        self,
        horizon: int,
        X_df: Optional[pd.DataFrame] = None,
        ds_col: str = "ds",
        unique_id_col: str = "unique_id",
        target_col: str = "y",
    ) -> pd.DataFrame:
        """Generate forecasts for provided series step by step.

        Args:
            horizon: Number of steps to forecast
            X_df: DataFrame with features for prediction. If provided, should contain
                static and/or dynamic features as needed.
            ds_col: Name of the date column. Default is 'ds'.
            unique_id_col: Name of the unique ID column. Default is 'unique_id'.
            target_col: Name of the target column. Default is 'y'.

        Returns:
            DataFrame with predictions in long format containing:
                - unique_id: series identifier
                - ds: forecast timestamp
                - y: predicted value
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before making predictions")

        # Create future dataframe
        future_df = self._create_X_df_from_history(
            horizon, unique_id_col, ds_col, target_col
        )

        if X_df is not None:
            future_df = pd.merge(future_df, X_df, on=[ds_col, unique_id_col], how="left")

        history_df = self._y_history

        for i in range(horizon):
            current_date = future_df[ds_col].min() + get_period(self._freq, n=i)
            
            merged_df = pd.concat([history_df, future_df], ignore_index=True)

            lags_df = self._generate_lags(merged_df, ds_col, unique_id_col, target_col)

            predict_df = lags_df[lags_df[ds_col] == current_date]

            if predict_df.empty:
                continue

            for col in self.feature_names_:
                if col not in predict_df.columns:
                    predict_df[col] = np.nan

            predictions = self.fitted_model.predict(predict_df[self.feature_names_])

            future_df.loc[future_df[ds_col] == current_date, target_col] = predictions
            history_df = pd.concat([history_df, future_df[future_df[ds_col] == current_date]])

        return future_df.rename(columns={target_col: "preds"})[        
            [unique_id_col, ds_col, "preds"]
        ]

    def _update_with_predictions(
        self,
        df: pd.DataFrame,
        predictions: pd.DataFrame,
        unique_id_col: str,
        ds_col: str,
    ) -> pd.DataFrame:
        """Update the main DataFrame with the latest predictions for lag generation.

        Args:
            df: Main DataFrame with time series data.
            predictions: DataFrame with the latest predictions containing
                columns [unique_id_col, ds_col, target_col].
            unique_id_col: Name of the unique ID column.
            ds_col: Name of the date column.

        Returns:
            Updated DataFrame with new predictions included.
        """

        # Set multi-index for both DataFrames
        df = df.set_index([ds_col, unique_id_col])
        predictions = predictions.set_index([ds_col, unique_id_col])

        # Update values
        df.update(predictions)

        return df.reset_index()

    def _get_n_splits(
        self,
        df: pd.DataFrame,
        window_size: int,
        horizon: int,
        validation_strategy: str,
        step_size: int,
        ds_col: str = "ds",
    ) -> int:
        """
        Calculate the number of splits for backtesting.
        
        Args:
            df: DataFrame with time series data
            window_size: Size of the rolling window (required for 'rolling')
            horizon: Forecast horizon for validation
            validation_strategy: One of ['rolling', 'expanding']
            step_size: Number of steps between each validation fold
            ds_col: Name of the date column
        """
        if validation_strategy == "rolling":
            return (len(df[ds_col].unique()) - window_size - horizon) // step_size + 1
        else:  # expanding
            return len(df[ds_col].unique()) - horizon

    def backtest(
        self,
        df: pd.DataFrame,
        validation_strategy: str = "rolling",
        window_size: Optional[int] = None,
        horizon: int = 1,
        step_size: int = 1,
        metrics : Optional[List[str]] = [rmse, mae, mape, smape, mase],
        ds_col: str = "ds",
        unique_id_col: str = "unique_id",
        target_col: str = "y",
    ) -> Dict[str, float]:
        """Perform backtesting using the specified validation strategy.

        Args:
            df: DataFrame with time series data
            validation_strategy: One of ['rolling', 'expanding']
            window_size: Size of the rolling window (required for 'rolling')
            horizon: Forecast horizon for validation
            step_size: Number of steps between each validation fold
            ds_col: Name of the date column
            unique_id_col: Name of the unique ID column
            target_col: Name of the target column

        Returns:
            Dictionary with performance metrics
        """
        if validation_strategy not in ["rolling", "expanding"]:
            raise ValueError(
                "validation_strategy must be one of ['rolling', 'expanding']"
            )

        if validation_strategy == "rolling" and window_size is None:
            raise ValueError("window_size is required for rolling validation")

        # Sort data by date and series
        df = df.sort_values([ds_col, unique_id_col]).reset_index(drop=True)

        # Get unique dates and series IDs
        dates = df[ds_col].unique()
        series_ids = df[unique_id_col].unique()

        # Calculate number of folds
        n_splits = self._get_n_splits(
            df, window_size or 0, horizon, validation_strategy, step_size, ds_col
        )

        if self.debug:
            logger.debug(f"Starting backtesting with {n_splits} splits...")


        results = {m.__name__: [] for m in metrics}

        for i in range(n_splits):
            if self.debug:
                logger.debug(f"Backtest fold {i + 1}/{n_splits}")
            if validation_strategy == "rolling":
                train_start = i * step_size
                train_end = train_start + window_size
            else:  # expanding
                train_start = 0
                train_end = i + window_size if window_size else len(dates) - horizon

            # Get train/test splits
            train_dates = dates[train_start:train_end]
            test_dates = dates[train_end : train_end + horizon]

            train_data = df[df[ds_col].isin(train_dates)]
            test_data = df[df[ds_col].isin(test_dates)]

            # Fit model on training data
            self.fit(train_data)

            # Generate predictions for test data
            predictions = self.predict(
                X_df=test_data[[unique_id_col, ds_col] + self.static_features + self.dynamic_features],
                ds_col=ds_col,
                unique_id_col=unique_id_col,
                target_col=target_col,
            )

            # Calculate metrics
            for series_id in series_ids:
                actual = test_data[test_data[unique_id_col] == series_id][target_col].values
                pred = predictions[predictions[unique_id_col] == series_id][
                    "preds"
                ].values

                if len(actual) > 0 and len(pred) > 0:
                    for metric in metrics:
                        metric_value = metric(actual, pred)
                        results[metric.__name__].append(metric_value)
                        
        # Return average metrics
        return {metric: float(np.mean(values)) for metric, values in results.items()}
