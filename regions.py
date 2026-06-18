from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegionSpec:
    id: str
    label: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    default_lat: float
    default_lon: float

    @property
    def lon_range(self) -> list[float]:
        return [self.lon_min, self.lon_max]

    @property
    def lat_range(self) -> list[float]:
        return [self.lat_min, self.lat_max]

    def contains(self, lat: float, lon: float) -> bool:
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max


REGION_ORDER = ("conus", "middle_east", "india", "europe", "australia", "south_america")

REGIONS: dict[str, RegionSpec] = {
    "conus": RegionSpec(
        id="conus",
        label="CONUS",
        lon_min=-125.0,
        lon_max=-66.0,
        lat_min=24.0,
        lat_max=50.0,
        default_lat=30.2672,
        default_lon=-97.7431,
    ),
    "middle_east": RegionSpec(
        id="middle_east",
        label="Middle East",
        lon_min=25.0,
        lon_max=65.0,
        lat_min=12.0,
        lat_max=42.0,
        default_lat=25.2048,
        default_lon=55.2708,
    ),
    "india": RegionSpec(
        id="india",
        label="India",
        lon_min=68.0,
        lon_max=98.0,
        lat_min=6.0,
        lat_max=38.0,
        default_lat=28.6139,
        default_lon=77.2090,
    ),
    "europe": RegionSpec(
        id="europe",
        label="Europe",
        lon_min=-11.0,
        lon_max=45.0,
        lat_min=35.0,
        lat_max=72.0,
        default_lat=48.8566,
        default_lon=2.3522,
    ),
    "australia": RegionSpec(
        id="australia",
        label="Australia",
        lon_min=112.0,
        lon_max=154.0,
        lat_min=-44.0,
        lat_max=-10.0,
        default_lat=-33.8688,
        default_lon=151.2093,
    ),
    "south_america": RegionSpec(
        id="south_america",
        label="South America",
        lon_min=-82.0,
        lon_max=-34.0,
        lat_min=-56.0,
        lat_max=13.0,
        default_lat=-23.5505,
        default_lon=-46.6333,
    ),
}

DEFAULT_DEMO_SURFACE = Path("data/sample_demo_global_zinc_surfaces.csv")


def get_region(region_id: str | None) -> RegionSpec:
    key = (region_id or "conus").lower()
    if key not in REGIONS:
        raise ValueError(f"Unknown region {region_id!r}. Expected one of {list(REGIONS)}.")
    return REGIONS[key]


def region_options() -> list[tuple[str, str]]:
    return [(region_id, REGIONS[region_id].label) for region_id in REGION_ORDER]


def production_surface_path(region_id: str) -> Path:
    return Path("data") / f"{get_region(region_id).id}_zinc_surface_2013_2022.csv"


def pick_default_data_path(region_id: str) -> Path:
    production = production_surface_path(region_id)
    if production.exists():
        return production
    legacy_conus = Path("data/conus_zinc_surface_2013_2022.csv")
    if region_id == "conus" and legacy_conus.exists():
        return legacy_conus
    return DEFAULT_DEMO_SURFACE
