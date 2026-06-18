"""ISO 9223/9224 zinc atmospheric corrosion utilities.

Units used by all public functions:
    T_C: annual mean air temperature, deg C
    RH_pct: annual mean relative humidity, percent
    Pd_mg_m2_d: SO2 deposition rate, mg SO2 / (m2 day)
    Sd_mg_m2_d: chloride deposition rate, mg Cl- / (m2 day)
    Rcorr: first-year zinc corrosion rate, micrometre/year
    Loss: zinc penetration/coating thickness consumed, micrometre
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# ISO 9223 Table 3 calibration intervals for the dose-response functions.
ISO9223_INTERVALS = {
    "T_C": (-17.1, 28.7),
    "RH_pct": (34.0, 93.0),
    "Pd_mg_m2_d": (0.7, 150.4),
    "Sd_mg_m2_d": (0.4, 760.5),
}

# ISO 9223 zinc first-year corrosion-rate category boundaries in micrometres/year.
ZINC_CATEGORY_BOUNDS_UM_Y = [
    ("C1", 0.1),
    ("C2", 0.7),
    ("C3", 2.1),
    ("C4", 4.2),
    ("C5", 8.4),
    ("CX", 25.0),
]

# ISO 9224 Table 2 generalized time exponents for zinc.
ZINC_B1 = 0.813
ZINC_B2 = 0.873

# ISO 9224 Table 2 generalized time exponents for carbon steel.
STEEL_B1 = 0.523
STEEL_B2 = 0.575

# kg/ha/year to mg/m2/day. 1 kg = 1e6 mg, 1 ha = 1e4 m2.
KG_HA_YR_TO_MG_M2_D = 1_000_000.0 / 10_000.0 / 365.25


def kg_s_ha_yr_to_mg_so2_m2_d(value_kg_s_ha_yr):
    """Convert kg-S/ha/year to mg-SO2/m2/day."""
    return np.asarray(value_kg_s_ha_yr, dtype=float) * KG_HA_YR_TO_MG_M2_D * 2.0


def kg_cl_ha_yr_to_mg_cl_m2_d(value_kg_cl_ha_yr):
    """Convert kg-Cl/ha/year to mg-Cl/m2/day."""
    return np.asarray(value_kg_cl_ha_yr, dtype=float) * KG_HA_YR_TO_MG_M2_D


def iso_9223_zinc_rcorr(
    T_C,
    RH_pct,
    Pd_mg_m2_d,
    Sd_mg_m2_d,
    *,
    clip_to_iso_intervals: bool = False,
):
    """Calculate first-year zinc corrosion rate using ISO 9223 Eq. 2."""
    T = np.asarray(T_C, dtype=float)
    RH = np.asarray(RH_pct, dtype=float)
    Pd = np.maximum(np.asarray(Pd_mg_m2_d, dtype=float), 0.0)
    Sd = np.maximum(np.asarray(Sd_mg_m2_d, dtype=float), 0.0)

    if clip_to_iso_intervals:
        T = np.clip(T, *ISO9223_INTERVALS["T_C"])
        RH = np.clip(RH, *ISO9223_INTERVALS["RH_pct"])
        Pd = np.clip(Pd, *ISO9223_INTERVALS["Pd_mg_m2_d"])
        Sd = np.clip(Sd, *ISO9223_INTERVALS["Sd_mg_m2_d"])

    f_zn = np.where(T <= 10.0, 0.038 * (T - 10.0), -0.071 * (T - 10.0))

    sulfur_term = 0.0129 * np.power(Pd, 0.44) * np.exp(0.046 * RH + f_zn)
    chloride_term = 0.0175 * np.power(Sd, 0.57) * np.exp(0.008 * RH + 0.085 * T)
    return sulfur_term + chloride_term


def iso_9223_steel_rcorr(
    T_C,
    RH_pct,
    Pd_mg_m2_d,
    Sd_mg_m2_d,
    *,
    clip_to_iso_intervals: bool = False,
):
    """Calculate first-year carbon steel corrosion loss using ISO 9223 Eq. 1."""
    T = np.asarray(T_C, dtype=float)
    RH = np.asarray(RH_pct, dtype=float)
    Pd = np.maximum(np.asarray(Pd_mg_m2_d, dtype=float), 0.0)
    Sd = np.maximum(np.asarray(Sd_mg_m2_d, dtype=float), 0.0)

    if clip_to_iso_intervals:
        T = np.clip(T, *ISO9223_INTERVALS["T_C"])
        RH = np.clip(RH, *ISO9223_INTERVALS["RH_pct"])
        Pd = np.clip(Pd, *ISO9223_INTERVALS["Pd_mg_m2_d"])
        Sd = np.clip(Sd, *ISO9223_INTERVALS["Sd_mg_m2_d"])

    f_st = np.where(T <= 10.0, 0.150 * (T - 10.0), -0.054 * (T - 10.0))
    sulfur_term = 1.77 * np.power(Pd, 0.52) * np.exp(0.020 * RH + f_st)
    chloride_term = 0.102 * np.power(Sd, 0.62) * np.exp(0.033 * RH + 0.040 * T)
    return sulfur_term + chloride_term


def zinc_corrosivity_category(rcorr_um_y):
    """Return ISO 9223 zinc corrosivity category for Rcorr in micrometres/year."""
    arr = np.asarray(rcorr_um_y, dtype=float)
    out = np.full(arr.shape, ">CX", dtype=object)
    for cat, upper in reversed(ZINC_CATEGORY_BOUNDS_UM_Y):
        out[arr <= upper] = cat
    if out.shape == ():
        return str(out.item())
    return out


def iso_9224_zinc_loss_um(
    rcorr_um_y,
    years,
    *,
    b: float = ZINC_B1,
    after_20: Literal["power", "linear"] = "linear",
):
    """Predict zinc loss/coating thickness consumed using ISO 9224."""
    r = np.asarray(rcorr_um_y, dtype=float)
    t = np.asarray(years, dtype=float)
    if np.any(t < 0):
        raise ValueError("Exposure years must be non-negative.")

    power_loss = r * np.power(t, b)
    if after_20 == "power":
        return power_loss
    if after_20 != "linear":
        raise ValueError("after_20 must be 'power' or 'linear'.")

    linear_tail = r * (np.power(20.0, b) + b * np.power(20.0, b - 1.0) * (t - 20.0))
    return np.where(t <= 20.0, power_loss, linear_tail)


def iso_input_warnings(T_C, RH_pct, Pd_mg_m2_d, Sd_mg_m2_d) -> list[str]:
    """Return human-readable warnings for inputs outside ISO 9223 intervals."""
    vals = {
        "T_C": float(np.asarray(T_C)),
        "RH_pct": float(np.asarray(RH_pct)),
        "Pd_mg_m2_d": float(np.asarray(Pd_mg_m2_d)),
        "Sd_mg_m2_d": float(np.asarray(Sd_mg_m2_d)),
    }
    labels = {
        "T_C": "temperature",
        "RH_pct": "relative humidity",
        "Pd_mg_m2_d": "SO2 deposition",
        "Sd_mg_m2_d": "chloride deposition",
    }
    units = {
        "T_C": "C",
        "RH_pct": "%",
        "Pd_mg_m2_d": "mg/(m2 d)",
        "Sd_mg_m2_d": "mg/(m2 d)",
    }
    warnings = []
    for key, value in vals.items():
        lo, hi = ISO9223_INTERVALS[key]
        if value < lo or value > hi:
            warnings.append(
                f"{labels[key]} = {value:.3g} {units[key]} is outside the ISO 9223 calibration interval "
                f"[{lo:g}, {hi:g}] {units[key]}."
            )
    return warnings


@dataclass(frozen=True)
class ZincPointResult:
    latitude: float
    longitude: float
    T_C: float
    RH_pct: float
    Pd_mg_m2_d: float
    Sd_mg_m2_d: float
    rcorr_um_y: float
    category: str
    d30_b1_um: float
    d30_b2_um: float


def calculate_point(
    latitude: float,
    longitude: float,
    T_C: float,
    RH_pct: float,
    Pd_mg_m2_d: float,
    Sd_mg_m2_d: float,
    *,
    after_20: Literal["power", "linear"] = "linear",
    clip_to_iso_intervals: bool = False,
) -> ZincPointResult:
    r = float(
        iso_9223_zinc_rcorr(
            T_C,
            RH_pct,
            Pd_mg_m2_d,
            Sd_mg_m2_d,
            clip_to_iso_intervals=clip_to_iso_intervals,
        )
    )
    return ZincPointResult(
        latitude=latitude,
        longitude=longitude,
        T_C=float(T_C),
        RH_pct=float(RH_pct),
        Pd_mg_m2_d=float(Pd_mg_m2_d),
        Sd_mg_m2_d=float(Sd_mg_m2_d),
        rcorr_um_y=r,
        category=zinc_corrosivity_category(r),
        d30_b1_um=float(iso_9224_zinc_loss_um(r, 30, b=ZINC_B1, after_20=after_20)),
        d30_b2_um=float(iso_9224_zinc_loss_um(r, 30, b=ZINC_B2, after_20=after_20)),
    )
