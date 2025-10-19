import pytest
from sklearn.ensemble import RandomForestRegressor


@pytest.fixture
def random_forest_model() -> RandomForestRegressor:
    """Return a RandomForestRegressor model with a fixed random state."""
    return RandomForestRegressor(random_state=42)
