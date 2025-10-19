# Global Forecasting Library

A Python library for training machine learning models on multiple time series data with support for zero-shot forecasting.

## Features

- Train a single model on multiple time series
- Support for static and dynamic features
- Automatic lag feature generation
- Time series cross-validation
- Feature importance analysis
- Zero-shot forecasting capabilities

## Installation

Using `uv`:

```bash
uv pip install global-forecasting
```

## Quick Start

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from global_forecasting import GlobalForecaster

# Prepare your data
df = pd.DataFrame({
    'ds': [...],        # datetime
    'y': [...],         # target values
    'unique_id': [...], # series identifiers
    'static_feat': [...],
    'dynamic_feat': [...]
})

# Initialize forecaster
model = RandomForestRegressor()
forecaster = GlobalForecaster(
    model=model,
    lags=[1, 7, 14, 28],  # Use 1-day, 1-week, 2-week, and 4-week lags
    static_features=['static_feat'],
    dynamic_features=['dynamic_feat']
)

# Train the model
forecaster.fit(df)

# Prepare future dynamic features
future_dynamic = pd.DataFrame({
    'ds': [...],          # future dates
    'unique_id': [...],   # series identifiers
    'dynamic_feat': [...] # future feature values
})

# Make predictions
predictions = forecaster.predict(
    X_df=df,
)
```

## Development

1. Clone the repository:
```bash
git clone https://github.com/gioalvari/global-forecasting.git
cd global-forecasting
```

2. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

3. Run tests:
```bash
pytest tests/ --cov=global_forecasting
```

4. Format code:
```bash
black global_forecasting tests
ruff check global_forecasting tests
mypy global_forecasting tests
```

## License

MIT License
