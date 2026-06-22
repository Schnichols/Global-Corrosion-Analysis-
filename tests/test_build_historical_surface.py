from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from scripts.build_historical_surface import aggregate_annual_values, build_surface, sample_deposition_csv_to_points


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


def test_sample_deposition_csv_linear_interpolates_to_points(tmp_path):
    csv_path = tmp_path / "deposition.csv"
    pd.DataFrame(
        {
            "lat": [0.0, 0.0, 1.0, 1.0],
            "lon": [0.0, 1.0, 0.0, 1.0],
            "Pd_mg_m2_d": [10.0, 11.0, 11.0, 12.0],
            "Sd_mg_m2_d": [20.0, 22.0, 21.0, 23.0],
        }
    ).to_csv(csv_path, index=False)
    points = pd.DataFrame({"lat": [0.5], "lon": [0.5]})

    Pd, Sd, distances = sample_deposition_csv_to_points(csv_path, points, interpolation="linear")

    assert np.allclose(Pd, [11.0])
    assert np.allclose(Sd, [21.5])
    assert distances[0] > 0


def test_sample_deposition_csv_can_limit_extrapolation_distance(tmp_path):
    csv_path = tmp_path / "deposition.csv"
    pd.DataFrame(
        {
            "lat": [0.0],
            "lon": [0.0],
            "Pd_mg_m2_d": [10.0],
            "Sd_mg_m2_d": [20.0],
        }
    ).to_csv(csv_path, index=False)
    points = pd.DataFrame({"lat": [5.0], "lon": [5.0]})

    Pd, Sd, _ = sample_deposition_csv_to_points(
        csv_path,
        points,
        interpolation="nearest",
        max_nearest_distance_deg=1.0,
    )

    assert np.isnan(Pd[0])
    assert np.isnan(Sd[0])


def test_build_surface_allows_non_conus_with_deposition_csv_and_power_weather(tmp_path):
    power_path = tmp_path / "power.nc"
    lats = np.array([6.0, 7.0])
    lons = np.array([68.0, 69.0])
    times = pd.date_range("2013-01-01", periods=2, freq="MS")
    xr.Dataset(
        {
            "T2M": (("time", "lat", "lon"), np.full((2, 2, 2), 25.0)),
            "RH2M": (("time", "lat", "lon"), np.full((2, 2, 2), 70.0)),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    ).to_netcdf(power_path)

    deposition_path = tmp_path / "india_deposition.csv"
    pd.DataFrame(
        {
            "lat": [6.0, 6.0, 7.0, 7.0],
            "lon": [68.0, 69.0, 68.0, 69.0],
            "Pd_mg_m2_d": [1.0, 1.0, 1.0, 1.0],
            "Sd_mg_m2_d": [2.0, 2.0, 2.0, 2.0],
        }
    ).to_csv(deposition_path, index=False)

    args = SimpleNamespace(
        region="india",
        start_year=2013,
        end_year=2022,
        resolution=1.0,
        lon_min=68.0,
        lon_max=69.0,
        lat_min=6.0,
        lat_max=7.0,
        deposition_source="auto",
        deposition_csv=str(deposition_path),
        deposition_label="unit test deposition",
        deposition_csv_interpolation="nearest",
        deposition_max_nearest_distance_deg=None,
        sulfur_variable="so2_dw",
        chloride_variable="cl_tw",
        cache_dir=str(tmp_path / "cache"),
        power_netcdf=str(power_path),
        clip_to_iso_intervals=False,
        allow_non_conus_tdep=False,
        output=None,
    )

    out = build_surface(args)

    assert len(out) == 4
    assert set(out["region"]) == {"india"}
    assert set(out["deposition_source_type"]) == {"csv"}
    assert set(out["deposition_source"]) == {"unit test deposition"}
    assert np.allclose(out["T_C"], 25.0)
    assert np.allclose(out["RH_pct"], 70.0)
