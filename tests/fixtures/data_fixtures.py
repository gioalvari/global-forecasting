import numpy as np
import pandas as pd
import pytest
from src.global_forecasting.utils import generate_sample_data


@pytest.fixture
def sample_data() -> pd.DataFrame:
    return generate_sample_data()


@pytest.fixture
def sample_data_with_missing() -> pd.DataFrame:
    return generate_sample_data(pct_of_missing=0.5)


@pytest.fixture
def sample_data_multiple_dynamic_static() -> pd.DataFrame:
    return generate_sample_data(
        no_of_dynamic_features=2,
        no_of_static_features=2,
    )


@pytest.fixture
def sample_data_no_missing() -> pd.DataFrame:
    return generate_sample_data(pct_of_missing=0.0)


@pytest.fixture
def short_sample_data() -> pd.DataFrame:
    return generate_sample_data(n_series=5, end_date=pd.Timestamp("2023-03-31"))


@pytest.fixture
def long_sample_data() -> pd.DataFrame:
    return generate_sample_data(n_series=10, end_date=pd.Timestamp("2024-12-31"))


@pytest.fixture
def sample_data_no_static() -> pd.DataFrame:
    return generate_sample_data(no_of_static_features=0)


@pytest.fixture
def sample_data_no_dynamic() -> pd.DataFrame:
    return generate_sample_data(no_of_dynamic_features=0)


@pytest.fixture
def sample_data_no_features() -> pd.DataFrame:
    return generate_sample_data(no_of_dynamic_features=0, no_of_static_features=0)
