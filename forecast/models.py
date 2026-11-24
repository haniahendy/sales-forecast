import numpy as np
import pandas as pd


def make_series(days: int = 365, seed: int = 5) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=days, freq="D")
    t = np.arange(days)
    y = 100 + 0.1 * t + 15 * np.sin(2 * np.pi * idx.dayofweek / 7) + rng.normal(0, 4, days)
    return pd.Series(y, index=idx, name="sales")


def naive_last(train: pd.Series, h: int) -> np.ndarray:
    return np.repeat(train.iloc[-1], h)


def seasonal_naive(train: pd.Series, h: int, period: int = 7) -> np.ndarray:
    last = train.iloc[-period:].to_numpy()
    return np.resize(last, h)


def holt_winters(train: pd.Series, h: int, period: int = 7) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    fit = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=period).fit()
    return np.asarray(fit.forecast(h))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rolling_origin_eval(series: pd.Series, h: int = 14, folds: int = 4) -> dict:
    """Average MAE over several forecast origins (no peeking at the future)."""
    methods = {"naive_last": naive_last, "seasonal_naive": seasonal_naive, "holt_winters": holt_winters}
    scores = {k: [] for k in methods}
    for f in range(folds):
        cut = len(series) - h * (folds - f)
        train, test = series.iloc[:cut], series.iloc[cut:cut + h]
        for name, fn in methods.items():
            scores[name].append(mae(test, fn(train, h)))
    return {k: round(float(np.mean(v)), 2) for k, v in scores.items()}
