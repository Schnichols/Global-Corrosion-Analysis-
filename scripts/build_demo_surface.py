#!/usr/bin/env python3
"""Create dense synthetic demo surfaces for all supported regions.

The demo data is only for UI smoke-testing. It is intentionally smooth and
plausible-looking, but it is not a historical corrosion surface.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corrosion_model import (  # noqa: E402
    ZINC_B1,
    ZINC_B2,
    iso_9223_zinc_rcorr,
    iso_9224_zinc_loss_um,
    zinc_corrosivity_category,
)
from regions import DEFAULT_DEMO_SURFACE, REGION_ORDER, REGIONS  # noqa: E402


def _gaussian(lon, lat, lon0, lat0, sx, sy, amp):
    return amp * np.exp(-(((lon - lon0) / sx) ** 2 + ((lat - lat0) / sy) ** 2))


def _region_environment(region_id: str, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon_mid = (REGIONS[region_id].lon_min + REGIONS[region_id].lon_max) / 2.0
    lat_mid = (REGIONS[region_id].lat_min + REGIONS[region_id].lat_max) / 2.0

    if region_id == "middle_east":
        coast_east = 1.0 / (1.0 + np.exp(-(lon - 54.0) / 2.5))
        mediterranean = np.exp(-((lon - 34.0) / 6.5) ** 2 - ((lat - 32.0) / 5.5) ** 2)
        gulf = np.exp(-((lon - 56.0) / 7.0) ** 2 - ((lat - 25.0) / 5.5) ** 2)
        T = 28.0 - 0.42 * (lat - lat_mid) + _gaussian(lon, lat, 45.0, 22.0, 11.0, 6.0, 4.0)
        RH = np.clip(34.0 + 19.0 * coast_east + 17.0 * mediterranean + 12.0 * gulf, 28.0, 88.0)
        Pd = np.clip(2.0 + _gaussian(lon, lat, 44.0, 31.0, 8.0, 5.5, 3.0) + _gaussian(lon, lat, 51.0, 25.0, 6.0, 3.8, 4.2), 0.4, 14.0)
        Sd = np.clip(1.0 + 32.0 * gulf + 22.0 * mediterranean + 18.0 * coast_east, 0.4, 120.0)
    elif region_id == "india":
        west_coast = np.exp(-((lon - 73.0) / 3.5) ** 2)
        east_coast = np.exp(-((lon - 81.5) / 3.8) ** 2)
        bay_head = _gaussian(lon, lat, 88.0, 22.0, 5.0, 5.0, 1.0)
        urban_industrial = _gaussian(lon, lat, 77.0, 28.5, 5.0, 3.5, 1.0) + _gaussian(lon, lat, 72.9, 19.2, 4.0, 3.0, 1.0)
        south = (REGIONS[region_id].lat_max - lat) / (REGIONS[region_id].lat_max - REGIONS[region_id].lat_min)
        monsoon = np.exp(-((lat - 18.0) / 10.0) ** 2)
        T = 19.0 + 11.0 * south + _gaussian(lon, lat, 77.0, 23.0, 10.0, 7.0, 4.0)
        RH = np.clip(42.0 + 24.0 * monsoon + 18.0 * west_coast + 16.0 * east_coast + 10.0 * bay_head, 30.0, 93.0)
        Pd = np.clip(1.0 + 5.2 * urban_industrial + _gaussian(lon, lat, 83.0, 23.0, 7.0, 4.5, 2.8), 0.3, 18.0)
        Sd = np.clip(0.7 + 36.0 * west_coast + 30.0 * east_coast + 26.0 * bay_head, 0.4, 160.0)
    elif region_id == "europe":
        atlantic = np.exp(-((lon + 5.0) / 8.0) ** 2)
        north = (lat - REGIONS[region_id].lat_min) / (REGIONS[region_id].lat_max - REGIONS[region_id].lat_min)
        T = 20.0 - 18.0 * north + 3.0 * np.exp(-((lat - 42.0) / 5.0) ** 2)
        RH = np.clip(58.0 + 18.0 * atlantic + 8.0 * north, 35.0, 92.0)
        Pd = np.clip(1.4 + _gaussian(lon, lat, 8.0, 50.0, 10.0, 6.0, 6.0) + _gaussian(lon, lat, 30.0, 49.0, 8.0, 5.0, 2.2), 0.4, 16.0)
        Sd = np.clip(0.8 + 30.0 * atlantic + _gaussian(lon, lat, 14.0, 42.0, 9.0, 4.0, 14.0), 0.4, 130.0)
    elif region_id == "australia":
        coast_east = np.exp(-((lon - 152.0) / 4.0) ** 2)
        coast_west = np.exp(-((lon - 115.0) / 4.5) ** 2)
        north = (lat - REGIONS[region_id].lat_min) / (REGIONS[region_id].lat_max - REGIONS[region_id].lat_min)
        T = 12.0 + 18.0 * north + _gaussian(lon, lat, 133.0, -24.0, 12.0, 7.0, 4.0)
        RH = np.clip(38.0 + 30.0 * coast_east + 16.0 * coast_west + 12.0 * north, 25.0, 90.0)
        Pd = np.clip(0.8 + _gaussian(lon, lat, 151.0, -33.0, 4.0, 4.0, 3.0) + _gaussian(lon, lat, 145.0, -38.0, 5.0, 4.0, 2.0), 0.3, 10.0)
        Sd = np.clip(0.6 + 34.0 * coast_east + 24.0 * coast_west + _gaussian(lon, lat, 153.0, -27.0, 4.0, 5.0, 22.0), 0.4, 140.0)
    elif region_id == "south_america":
        pacific = np.exp(-((lon + 78.0) / 4.0) ** 2)
        atlantic = np.exp(-((lon + 38.0) / 4.5) ** 2)
        tropical = np.exp(-((lat - 0.0) / 18.0) ** 2)
        T = 25.0 * tropical + 7.0 * (1.0 - tropical) + _gaussian(lon, lat, -60.0, -23.0, 12.0, 8.0, 4.5)
        RH = np.clip(55.0 + 22.0 * tropical + 18.0 * atlantic + 10.0 * pacific, 32.0, 93.0)
        Pd = np.clip(0.9 + _gaussian(lon, lat, -46.0, -23.0, 6.0, 5.0, 5.0) + _gaussian(lon, lat, -58.0, -34.0, 7.0, 5.0, 2.4), 0.3, 14.0)
        Sd = np.clip(0.7 + 30.0 * pacific + 34.0 * atlantic + _gaussian(lon, lat, -43.0, -22.0, 5.0, 4.0, 22.0), 0.4, 150.0)
    else:
        east = 1.0 / (1.0 + np.exp(-(lon + 97.0) / 5.5))
        gulf = np.exp(-((lat - 29.0) / 3.5) ** 2 - ((lon + 90.0) / 11.0) ** 2)
        west_coast = np.exp(-((lon + 123.0) / 3.0) ** 2)
        T = 30.0 - 0.72 * (lat - 24.0) - _gaussian(lon, lat, -109.0, 41.0, 9.0, 5.0, 5.5)
        RH = np.clip(44.0 + 24.0 * east + 9.0 * gulf + 12.0 * west_coast - _gaussian(lon, lat, -113.0, 34.0, 7.0, 4.0, 18.0), 25.0, 92.0)
        Pd = np.clip(0.75 + _gaussian(lon, lat, -82.0, 39.0, 7.0, 5.0, 4.8) + _gaussian(lon, lat, -91.0, 30.0, 8.0, 3.0, 2.0) + 1.4 * east, 0.2, 14.0)
        Sd = np.clip(0.5 + 22.0 * west_coast + 28.0 * np.exp(-((lon + 76.0) / 3.0) ** 2) + 38.0 * gulf, 0.2, 120.0)

    return T, RH, Pd, Sd


def make_demo(resolution: float = 0.5) -> pd.DataFrame:
    frames = []
    for region_id in REGION_ORDER:
        region = REGIONS[region_id]
        lons = np.round(np.arange(region.lon_min, region.lon_max + resolution * 0.5, resolution), 6)
        lats = np.round(np.arange(region.lat_min, region.lat_max + resolution * 0.5, resolution), 6)
        lon2, lat2 = np.meshgrid(lons, lats)
        lon = lon2.ravel()
        lat = lat2.ravel()
        T, RH, Pd, Sd = _region_environment(region_id, lon, lat)
        R = iso_9223_zinc_rcorr(T, RH, Pd, Sd)
        frames.append(
            pd.DataFrame(
                {
                    "region": region.id,
                    "region_label": region.label,
                    "lat": lat,
                    "lon": lon,
                    "T_C": T,
                    "RH_pct": RH,
                    "Pd_mg_m2_d": Pd,
                    "Sd_mg_m2_d": Sd,
                    "Rcorr_um_y": R,
                    "category": zinc_corrosivity_category(R),
                    "D30_B1_um": iso_9224_zinc_loss_um(R, 30, b=ZINC_B1, after_20="linear"),
                    "D30_B2_um": iso_9224_zinc_loss_um(R, 30, b=ZINC_B2, after_20="linear"),
                    "source_years": "synthetic-demo",
                    "sulfur_variable": "synthetic",
                    "chloride_variable": "synthetic",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    out = ROOT / DEFAULT_DEMO_SURFACE
    out.parent.mkdir(parents=True, exist_ok=True)
    demo = make_demo()
    demo.to_csv(out, index=False)
    print(f"Wrote {len(demo):,} rows to {out}")
