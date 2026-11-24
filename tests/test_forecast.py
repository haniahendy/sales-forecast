import numpy as np

from forecast.models import holt_winters, make_series, mae, naive_last, rolling_origin_eval, seasonal_naive


def test_shapes():
    s = make_series()
    for fn in (naive_last, seasonal_naive, holt_winters):
        assert len(fn(s, 10)) == 10


def test_seasonal_naive_repeats_last_week():
    s = make_series()
    f = seasonal_naive(s, 14)
    assert np.allclose(f[:7], s.iloc[-7:]) and np.allclose(f[7:], s.iloc[-7:])


def test_mae_value():
    assert mae([1, 2, 3], [2, 2, 5]) == 1.0


def test_no_leakage_between_train_and_test():
    s = make_series()
    # a forecaster must not change when future values change
    s2 = s.copy()
    s2.iloc[-14:] += 1000
    assert np.allclose(seasonal_naive(s.iloc[:-14], 14), seasonal_naive(s2.iloc[:-14], 14))


def test_seasonal_methods_beat_naive_on_seasonal_data():
    r = rolling_origin_eval(make_series())
    assert r["seasonal_naive"] < r["naive_last"]
    assert r["holt_winters"] < r["naive_last"]
