import pandas as pd
from typing import List


def generate_future_dates(
    last_date: pd.Timestamp, horizon: int, freq: str
) -> pd.DatetimeIndex:
    """
    Generate future dates for forecasting.

    Args:
        last_date: The last known date in the time series.
        horizon: The number of future steps to generate.
        freq: The frequency of the time series (e.g., 'D', 'W', 'M').

    Returns:
        A DatetimeIndex with the future dates.
    """
    return pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]


def create_lag_features(
    df: pd.DataFrame, target_col: str, lag_values: List[int]
) -> pd.DataFrame:
    """
    Create lag features for a time series dataframe.

    This function assumes the dataframe is sorted by date for each time series.
    It performs a simple shift and does not account for gaps in time.

    Args:
        df: DataFrame containing time series data, must include a 'unique_id' column.
        target_col: The name of the column to lag.
        lag_values: A list of integers representing the lag periods.

    Returns:
        DataFrame with added lag feature columns.
    """
    df = df.copy()
    if "unique_id" not in df.columns:
        raise ValueError(
            "Input DataFrame must contain a 'unique_id' column for grouping."
        )

    for lag in lag_values:
        df[f"lag_{lag}"] = df.groupby("unique_id")[target_col].shift(lag)
    return df
