from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


RISK_RANK = {"medium": 1, "high": 2}

SOIL_RISK_LEVEL_STYLES = {
    "medium": {
        "label": "Medium",
        "line": "#b45309",
        "fill": "rgba(245, 158, 11, 0.17)",
    },
    "high": {
        "label": "High",
        "line": "#991b1b",
        "fill": "rgba(220, 38, 38, 0.20)",
    },
}


@dataclass(frozen=True)
class SoilRiskRegion:
    id: str
    region_id: str
    label: str
    risk_level: str
    polygon: tuple[tuple[float, float], ...]
    basis: str

    @property
    def risk_label(self) -> str:
        return SOIL_RISK_LEVEL_STYLES[self.risk_level]["label"]

    @property
    def rank(self) -> int:
        return RISK_RANK[self.risk_level]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        lons = [point[0] for point in self.polygon]
        lats = [point[1] for point in self.polygon]
        return min(lons), min(lats), max(lons), max(lats)


def _rectangle(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> tuple[tuple[float, float], ...]:
    return (
        (lon_min, lat_min),
        (lon_max, lat_min),
        (lon_max, lat_max),
        (lon_min, lat_max),
        (lon_min, lat_min),
    )


SOIL_RISK_REGIONS: tuple[SoilRiskRegion, ...] = (
    SoilRiskRegion(
        id="arabian_gulf_coastal_sabkha",
        region_id="middle_east",
        label="Arabian Gulf coastal sabkha belt",
        risk_level="high",
        polygon=_rectangle(45.2, 23.0, 57.5, 30.7),
        basis="Broad coastal sabkha and evaporite-flat screening zone along the Arabian Gulf.",
    ),
    SoilRiskRegion(
        id="kuwait_southern_iraq_saline_lowlands",
        region_id="middle_east",
        label="Kuwait / southern Iraq saline lowlands",
        risk_level="high",
        polygon=_rectangle(46.0, 28.4, 48.8, 31.5),
        basis="Low-lying saline soils and sabkha/playa terrain screening zone.",
    ),
    SoilRiskRegion(
        id="qatar_bahrain_sabkha_lowlands",
        region_id="middle_east",
        label="Qatar / Bahrain sabkha lowlands",
        risk_level="high",
        polygon=_rectangle(50.1, 24.5, 51.9, 26.8),
        basis="Peninsular Gulf sabkha and coastal evaporite-flat screening zone.",
    ),
    SoilRiskRegion(
        id="uae_western_coastal_sabkha",
        region_id="middle_east",
        label="UAE western coastal sabkha / Liwa edge",
        risk_level="high",
        polygon=_rectangle(52.5, 22.4, 55.6, 24.9),
        basis="Abu Dhabi western coastal sabkha and inland salt-flat screening zone.",
    ),
    SoilRiskRegion(
        id="red_sea_coastal_sabkha",
        region_id="middle_east",
        label="Red Sea coastal sabkha belt",
        risk_level="medium",
        polygon=(
            (35.0, 16.5),
            (41.2, 16.5),
            (40.2, 21.5),
            (38.8, 24.0),
            (37.3, 27.8),
            (36.5, 29.0),
            (35.0, 29.0),
            (35.0, 16.5),
        ),
        basis="Coastal sabkha, saline dust, and marine aerosol screening zone.",
    ),
    SoilRiskRegion(
        id="dead_sea_jordan_valley",
        region_id="middle_east",
        label="Dead Sea / Jordan Valley evaporite basin",
        risk_level="high",
        polygon=_rectangle(35.1, 30.7, 36.0, 32.5),
        basis="Evaporite basin and salt-affected lowland screening zone.",
    ),
    SoilRiskRegion(
        id="dasht_e_kavir",
        region_id="middle_east",
        label="Dasht-e Kavir salt desert",
        risk_level="high",
        polygon=_rectangle(51.0, 32.0, 58.5, 36.5),
        basis="Major Iranian salt desert/playa screening zone.",
    ),
    SoilRiskRegion(
        id="dasht_e_lut",
        region_id="middle_east",
        label="Dasht-e Lut playa / salt desert",
        risk_level="high",
        polygon=_rectangle(56.0, 28.5, 61.5, 33.5),
        basis="Interior playa and evaporite desert screening zone.",
    ),
    SoilRiskRegion(
        id="lake_urmia_basin",
        region_id="middle_east",
        label="Lake Urmia salt-affected basin",
        risk_level="high",
        polygon=_rectangle(44.0, 37.0, 46.5, 38.6),
        basis="Salt-lake basin and wind-blown saline sediment screening zone.",
    ),
    SoilRiskRegion(
        id="rann_of_kutch",
        region_id="india",
        label="Great and Little Rann of Kutch",
        risk_level="high",
        polygon=_rectangle(68.2, 22.3, 72.7, 24.8),
        basis="Large salt marsh/salt-flat system in Gujarat screening zone.",
    ),
    SoilRiskRegion(
        id="gujarat_saurashtra_coastal_saline_belt",
        region_id="india",
        label="Gujarat / Saurashtra coastal saline belt",
        risk_level="high",
        polygon=_rectangle(68.5, 20.2, 73.5, 22.8),
        basis="Coastal saline soil and marine salt/dust screening zone.",
    ),
    SoilRiskRegion(
        id="thar_nw_rajasthan_saline_playas",
        region_id="india",
        label="Thar / northwest Rajasthan saline playas",
        risk_level="medium",
        polygon=_rectangle(70.0, 24.0, 76.5, 30.0),
        basis="Arid inland saline playa and salt-affected soil screening zone.",
    ),
    SoilRiskRegion(
        id="haryana_punjab_western_up_sodic_belt",
        region_id="india",
        label="Haryana / Punjab / western Uttar Pradesh sodic-saline belt",
        risk_level="medium",
        polygon=_rectangle(74.0, 26.0, 81.5, 30.5),
        basis="Irrigated alluvial sodic and salt-affected soil screening zone.",
    ),
    SoilRiskRegion(
        id="indo_gangetic_sodic_soils",
        region_id="india",
        label="Indo-Gangetic sodic soil belt",
        risk_level="medium",
        polygon=_rectangle(78.0, 24.5, 85.0, 28.5),
        basis="Sodic and salt-affected alluvial soil screening zone.",
    ),
    SoilRiskRegion(
        id="sundarbans_delta_coastal_salinity",
        region_id="india",
        label="Sundarbans / Ganges-Brahmaputra delta coastal salinity",
        risk_level="high",
        polygon=_rectangle(87.5, 21.3, 89.8, 23.4),
        basis="Deltaic coastal salinity and marine inundation screening zone.",
    ),
    SoilRiskRegion(
        id="krishna_godavari_coastal_salinity",
        region_id="india",
        label="Krishna-Godavari coastal salinity",
        risk_level="medium",
        polygon=_rectangle(80.3, 15.3, 82.6, 17.8),
        basis="Coastal alluvial salinity and marine aerosol screening zone.",
    ),
    SoilRiskRegion(
        id="tamil_nadu_coromandel_coastal_salinity",
        region_id="india",
        label="Tamil Nadu / Coromandel coastal salinity",
        risk_level="medium",
        polygon=_rectangle(79.0, 8.8, 80.5, 13.6),
        basis="Coastal saline soil and marine aerosol screening zone.",
    ),
)


def soil_risk_regions_for_region(region_id: str) -> list[SoilRiskRegion]:
    return [region for region in SOIL_RISK_REGIONS if region.region_id == region_id]


def _point_in_polygon(lon: float, lat: float, polygon: Iterable[tuple[float, float]]) -> bool:
    points = list(polygon)
    if len(points) < 3:
        return False

    inside = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        on_lat_band = (yi > lat) != (yj > lat)
        if on_lat_band:
            x_intersection = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_intersection:
                inside = not inside
        j = i
    return inside


def matching_soil_risk_regions(region_id: str, lat: float, lon: float) -> list[SoilRiskRegion]:
    matches: list[SoilRiskRegion] = []
    for region in soil_risk_regions_for_region(region_id):
        lon_min, lat_min, lon_max, lat_max = region.bbox
        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            continue
        if _point_in_polygon(lon, lat, region.polygon):
            matches.append(region)
    return sorted(matches, key=lambda item: (-item.rank, item.label))


def highest_soil_risk_for_point(region_id: str, lat: float, lon: float) -> SoilRiskRegion | None:
    matches = matching_soil_risk_regions(region_id, lat, lon)
    return matches[0] if matches else None
