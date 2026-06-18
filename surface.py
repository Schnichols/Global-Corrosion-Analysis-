from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from corrosion_model import calculate_point

REQUIRED_COLUMNS = {
    "lat",
    "lon",
    "T_C",
    "RH_pct",
    "Pd_mg_m2_d",
    "Sd_mg_m2_d",
    "Rcorr_um_y",
}


class ZincSurface:
    """In-memory interpolator for a zinc corrosion surface."""

    def __init__(self, df: pd.DataFrame):
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Surface data missing columns: {sorted(missing)}")
        self.df = df.copy()
        self.df["lat"] = self.df["lat"].astype(float)
        self.df["lon"] = self.df["lon"].astype(float)
        self.regular = self._is_regular_grid()
        self._interpolators = {}
        if self.regular:
            self._build_regular_interpolators()
        else:
            self._tree = cKDTree(self.df[["lat", "lon"]].to_numpy(float))

    @classmethod
    def from_csv(cls, path: str | Path) -> "ZincSurface":
        return cls(pd.read_csv(path))

    def _is_regular_grid(self) -> bool:
        n_unique = self.df["lat"].nunique() * self.df["lon"].nunique()
        return n_unique == len(self.df)

    def _build_regular_interpolators(self):
        lats = np.sort(self.df["lat"].unique())
        lons = np.sort(self.df["lon"].unique())
        self._lats = lats
        self._lons = lons
        for col in ["T_C", "RH_pct", "Pd_mg_m2_d", "Sd_mg_m2_d", "Rcorr_um_y"]:
            grid = self.df.pivot(index="lat", columns="lon", values=col).sort_index().to_numpy(float)
            self._interpolators[col] = RegularGridInterpolator(
                (lats, lons), grid, bounds_error=False, fill_value=np.nan
            )

    def sample(self, latitude: float, longitude: float, *, after_20: str = "linear", clip_to_iso_intervals: bool = False):
        if self.regular:
            vals = {col: float(interp([[latitude, longitude]])[0]) for col, interp in self._interpolators.items()}
            if any(np.isnan(v) for v in vals.values()):
                raise ValueError("Point is outside the available grid extent or falls in a no-data cell.")
        else:
            _, idx = self._tree.query([[latitude, longitude]], k=1)
            row = self.df.iloc[int(idx[0])]
            vals = {col: float(row[col]) for col in ["T_C", "RH_pct", "Pd_mg_m2_d", "Sd_mg_m2_d", "Rcorr_um_y"]}

        return calculate_point(
            latitude=latitude,
            longitude=longitude,
            T_C=vals["T_C"],
            RH_pct=vals["RH_pct"],
            Pd_mg_m2_d=vals["Pd_mg_m2_d"],
            Sd_mg_m2_d=vals["Sd_mg_m2_d"],
            after_20=after_20,
            clip_to_iso_intervals=clip_to_iso_intervals,
        )
