import pandas as pd

from app import filter_surface_to_region
from regions import REGION_ORDER, REGIONS, get_region, production_surface_path


def test_requested_regions_are_configured():
    assert set(REGION_ORDER) == {"conus", "middle_east", "india", "europe", "australia", "south_america"}
    for region_id in REGION_ORDER:
        region = get_region(region_id)
        assert region.lon_min < region.lon_max
        assert region.lat_min < region.lat_max
        assert region.contains(region.default_lat, region.default_lon)


def test_production_surface_paths_are_region_specific():
    assert production_surface_path("europe").as_posix() == "data/europe_zinc_surface_2013_2022.csv"
    assert production_surface_path("middle_east").as_posix() == "data/middle_east_zinc_surface_2013_2022.csv"
    assert production_surface_path("india").as_posix() == "data/india_zinc_surface_2013_2022.csv"


def test_filter_surface_to_region_prefers_region_column():
    df = pd.DataFrame(
        {
            "region": ["europe", "australia"],
            "lat": [48.0, -30.0],
            "lon": [2.0, 140.0],
            "T_C": [10.0, 22.0],
            "RH_pct": [75.0, 55.0],
            "Pd_mg_m2_d": [3.0, 1.5],
            "Sd_mg_m2_d": [5.0, 8.0],
            "Rcorr_um_y": [1.0, 1.0],
        }
    )

    out = filter_surface_to_region(df, REGIONS["australia"])

    assert out["region"].tolist() == ["australia"]
    assert out["lat"].tolist() == [-30.0]
