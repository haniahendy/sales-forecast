# sales-forecast

Daily sales forecasting baselines compared with rolling-origin evaluation: last-value naive, seasonal naive (repeat last week), and Holt-Winters (statsmodels).

## Run
```bash
pip install -r requirements.txt
python -m pytest -q    # 5 tests, including a no-leakage check
python run.py
```

## Measured result (23 Nov 2026)
Mean MAE over 4 forecast origins, 14-day horizon, on a synthetic series (trend + weekly cycle + noise):

| method | MAE |
|---|---|
| naive (last value) | 16.29 |
| seasonal naive | 4.83 |
| Holt-Winters | 3.36 |

## Limits
Synthetic series with a clean weekly pattern, so seasonal methods are expected to win. No holidays, promotions, or external regressors. No prediction intervals. Holt-Winters is not tuned. Try real data before drawing conclusions.

## Provenance
A learning and portfolio project.
