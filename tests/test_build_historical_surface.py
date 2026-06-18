import numpy as np

from scripts.build_historical_surface import aggregate_annual_values


def test_tdep_aggregation_uses_annual_max_per_point():
    year_values = [
        np.array([1.0, np.nan, 5.0]),
        np.array([3.0, np.nan, 2.0]),
        np.array([2.0, np.nan, np.nan]),
    ]

    out = aggregate_annual_values(year_values, "max")

    assert out[0] == 3.0
    assert np.isnan(out[1])
    assert out[2] == 5.0
