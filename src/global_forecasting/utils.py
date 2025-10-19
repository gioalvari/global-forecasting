import pandas as pd
import numpy as np
from datetime import datetime


def generate_sample_data(
    start_date: datetime = datetime(2023, 1, 1),
    end_date: datetime = datetime(2023, 12, 31),
    freq: str = "D",
    n_series: int = 40,
    pct_of_missing: float = 0.1,
    no_of_dynamic_features: int = 1,
    no_of_static_features: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic time series data with varying date ranges and features.

    Creates multiple time series with different start dates, lengths, and characteristics.
    Each series includes a target variable (y), dynamic features (temperature), and
    static features (series type).

    Parameters
    ----------
    start_date : datetime, default=datetime(2023, 1, 1)
        Start date for the overall date range.
    end_date : datetime, default=datetime(2023, 12, 31)
        End date for the overall date range.
    freq : str, default='D'
        Frequency of the time series data.
    n_series : int, default=40
        Number of time series to generate.
    pct_of_missing : float, default=0.1
        Pct of rows dropped randomly to simulate missing data.
    no_of_dynamic_features : int, default=1
        Number of dynamic features to generate (currently only 1 supported).
    no_of_static_features : int, default=1
        Number of static features to generate (currently only 1 supported).
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame with the following columns:
        - ds : datetime
            Date timestamp for each observation
        - unique_id : str
            Unique identifier for each series (e.g., 'Serie_1', 'Serie_2')
        - y : float
            Target variable with trend, seasonality, and noise
        - dynamic_feature : float
            Simulated temperature with annual seasonality
        - static_feature : str
            Categorical series type ('A', 'B', or 'C')
        - start_date : datetime
            Start date of the series
        - end_date : datetime
            End date of the series
        - series_length : int
            Number of observations in the series

    Examples
    --------
    >>> data = generate_sample_data(n_series=10, seed=123)
    >>> print(data.head())
    >>> print(data.groupby('unique_id')['ds'].agg(['min', 'max', 'count']))

    Notes
    -----
    - Series lengths vary between 90 and 250 days
    - Start dates are randomly distributed across 2023
    - Each series has unique trend, seasonality, and base value parameters
    - Temperature feature follows an annual sinusoidal pattern with daily noise
    """
    np.random.seed(seed)

    # Define the overall date range for possible series
    overall_start = pd.to_datetime(start_date)
    overall_end = pd.to_datetime(end_date)

    # Generate random start dates for each series
    # FIX: Use list comprehension to generate one start date per series
    start_dates = [
        pd.to_datetime(
            np.random.choice(pd.date_range(overall_start, overall_end, freq=freq))
        )
        for _ in range(n_series)
    ]

    # FIX: Generate end dates for each series individually
    # Each end date must be after its corresponding start date
    end_dates = [
        pd.to_datetime(
            np.random.choice(pd.date_range(start_dates[i], overall_end, freq=freq))
        )
        for i in range(n_series)
    ]

    # Generate random parameters for each series
    base_values = np.random.uniform(50, 150, n_series)
    trend_coefs = np.random.uniform(-0.1, 0.1, n_series)
    seasonality_amps = np.random.uniform(2, 10, n_series)

    # Static feature: series type (categorical)
    series_types = np.random.choice([1, 2, 3], n_series)

    df_long = pd.DataFrame()

    for i in range(n_series):
        # Generate date range for this series
        dates = pd.date_range(start=start_dates[i], end=end_dates[i], freq=freq)

        # FIX: Use actual length of dates array to ensure consistent dimensions
        n_points = len(dates)

        # Generate trend component
        trend = (
            np.linspace(0, (end_dates[i] - start_dates[i]).days, n_points)
            * trend_coefs[i]
        )

        # Generate seasonality component (adaptive to frequency)
        # Use a cycle that repeats every ~12 periods (works for any frequency)
        seasonality = np.sin(2 * np.pi * np.arange(n_points) / 12) * seasonality_amps[i]

        # Generate random noise
        noise = np.random.normal(0, 0.1, n_points)

        # Combine components to create target variable
        values = base_values[i] + trend + seasonality + noise

        # Create dataframe for this series
        temp_df = pd.DataFrame(
            {
                "ds": dates,
                "unique_id": f"Serie_{i + 1}",
                "y": values,
            }
        )

        # Generate dynamic features (simulated temperature with annual seasonality)
        # Calculate fractional year position (0 to 1) for any frequency
        start_of_year = pd.Timestamp(dates[0].year, 1, 1)
        end_of_year = pd.Timestamp(dates[0].year, 12, 31)
        year_length = (end_of_year - start_of_year).days + 1
        days_from_start = (dates - start_of_year).days
        year_fraction = days_from_start / year_length

        for j in range(no_of_dynamic_features):
            # Annual cycle with base temperature of 20 and amplitude of 10
            dynamic_feature = 20 + 10 * np.sin(2 * np.pi * year_fraction)
            # Add variability with correct array length
            dynamic_feature += np.random.normal(0, 2, n_points)
            temp_df[f"dynamic_feature_{j + 1}"] = dynamic_feature

        # Add static features (categorical series type)
        for k in range(no_of_static_features):
            temp_df[f"static_feature_{k + 1}"] = series_types[i]

        # Append to main dataframe
        df_long = pd.concat([df_long, temp_df], ignore_index=True)

    # Introduce missing values randomly if specified
    if pct_of_missing > 0:
        n_rows = df_long.shape[0]
        n_missing = int(n_rows * pct_of_missing)
        missing_indices = np.random.choice(n_rows, n_missing, replace=False)
        df_long = df_long.drop(index=missing_indices).reset_index(drop=True)

    return df_long


def get_period(freq, n=1):
    """return a DateOffset or offset object corresponding to the given frequency and number of periods.

    Args:
        freq: The frequency to use. Supported frequencies:
            - 'D' or 'Day': Daily
            - 'B' or 'BusinessDay': Business days (excluding weekends)
            - 'W' or 'Week': Weekly
            - 'M' or 'Month': Monthly (end of month)
            - 'MS' or 'MonthStart': Monthly (start of month)
            - 'Q' or 'Quarter': Quarterly
            - 'QS' or 'QuarterStart': Quarterly (start of quarter)
            - 'A' or 'Y' or 'Year': Yearly (end of year)
            - 'AS' or 'YS' or 'YearStart': Yearly (start of year)
            - 'H' or 'Hour': Hourly
            - 'T' or 'Min': Minutely
            - 'S' or 'Second': Secondly
            - 'SM' or 'SemiMonth': Semi-monthly (15th and end of month)
        n: The number of periods to add (can be negative for subtraction)

    Returns:
        DateOffset or offset object corresponding to the frequency and n

    """

    # Normalizza la frequenza (maiuscolo)
    freq = freq.upper() if isinstance(freq, str) else freq

    # Gestione delle frequenze
    if freq in ("D", "DAY"):
        return pd.DateOffset(days=n)

    elif freq in ("B", "BUSINESSDAY"):
        return pd.offsets.BDay(n)

    elif freq in ("W", "WEEK"):
        return pd.DateOffset(weeks=n)

    elif freq in ("ME", "MONTHEND"):
        return pd.offsets.MonthEnd(n)

    elif freq in ("M", "MONTH"):
        return pd.offsets.MonthEnd(n)

    elif freq in ("MS", "MONTHSTART"):
        return pd.offsets.MonthBegin(n)

    elif freq in ("Q", "QUARTER"):
        return pd.offsets.QuarterEnd(n)

    elif freq in ("QS", "QUARTERSTART"):
        return pd.offsets.QuarterBegin(n)

    elif freq in ("A", "Y", "YEAR"):
        return pd.offsets.YearEnd(n)

    elif freq in ("AS", "YS", "YEARSTART"):
        return pd.offsets.YearBegin(n)

    elif freq in ("H", "HOUR"):
        return pd.DateOffset(hours=n)

    elif freq in ("T", "MIN", "MINUTE"):
        return pd.DateOffset(minutes=n)

    elif freq in ("S", "SECOND"):
        return pd.DateOffset(seconds=n)

    elif freq in ("SM", "SEMIMONTH"):
        return pd.offsets.SemiMonthEnd(n)

    else:
        raise ValueError(
            f"Unsupported frequency: {freq}. "
            f"Supported: D, B, W, M, ME, MS, Q, QS, A/Y, AS/YS, H, T/Min, S, SM"
        )
