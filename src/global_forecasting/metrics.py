import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE) between true and predicted values."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE) between true and predicted values."""
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE) between true and predicted values."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mase(y_true: np.ndarray, y_pred: np.ndarray, seasonal_period: int = 1) -> float:
    """Calculate Mean Absolute Scaled Error (MASE) between true and predicted values."""
    n = len(y_true)
    d = np.abs(np.diff(y_true, n=seasonal_period)).sum() / (n - seasonal_period)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (sMAPE) between true and predicted values."""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared (Coefficient of Determination) between true and predicted values."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)