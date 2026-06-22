import numpy as np
import pandas as pd

from scripts.build_cams_eac4_corrosion_inputs import (
    CAMS_VARIABLES,
    DepositionAssumptions,
    add_deposition_proxy_columns,
    build_3hourly_request,
    build_monthly_request,
    mixing_ratio_to_deposition_mg_m2_d,
    parse_month,
)


def test_build_monthly_request_uses_eac4_variables_and_region_area():
    request = build_monthly_request("middle_east", 2020, [1, 2], model_level=60, area_pad_degrees=0.0)

    assert request["variable"] == CAMS_VARIABLES
    assert request["model_level"] == ["60"]
    assert request["year"] == ["2020"]
    assert request["month"] == ["01", "02"]
    assert request["product_type"] == ["monthly_mean"]
    assert request["area"] == [42.0, 25.0, 12.0, 65.0]


def test_build_3hourly_request_matches_user_supplied_time_steps():
    request = build_3hourly_request("india", parse_month("2024-02"), model_level=60, area_pad_degrees=0.0)

    assert request["date"] == ["2024-02-01/2024-02-29"]
    assert request["time"] == ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]
    assert request["model_level"] == ["60"]


def test_mixing_ratio_to_deposition_proxy_unit_conversion():
    out = mixing_ratio_to_deposition_mg_m2_d(
        np.array([1e-9]),
        air_density_kg_m3=1.2,
        deposition_velocity_m_s=0.005,
    )

    assert np.allclose(out, [0.5184])


def test_add_deposition_proxy_columns_sums_three_sea_salt_bins():
    df = {
        "so2_kg_kg": [1e-9],
        "sea_salt_0p03_0p5um_kg_kg": [1e-9],
        "sea_salt_0p5_5um_kg_kg": [2e-9],
        "sea_salt_5_20um_kg_kg": [3e-9],
    }

    out = add_deposition_proxy_columns(pd.DataFrame(df), DepositionAssumptions())

    assert np.allclose(out["sea_salt_total_kg_kg"], [6e-9])
    assert out["Pd_mg_m2_d"].iloc[0] > 0
    assert out["Sd_mg_m2_d"].iloc[0] > out["Sd_medium_mg_m2_d"].iloc[0]
