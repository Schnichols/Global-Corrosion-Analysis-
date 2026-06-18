from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ZINC_DENSITY_G_CM3 = 7.14
OZ_FT2_TO_G_M2 = 28.349523125 / 0.09290304
G90_TOTAL_MASS_G_M2 = 0.90 * OZ_FT2_TO_G_M2
G90_ZINC_UM_PER_SIDE = G90_TOTAL_MASS_G_M2 / (2.0 * ZINC_DENSITY_G_CM3)


@dataclass(frozen=True)
class MaterialSpec:
    id: str
    label: str
    short_label: str
    kind: str
    default_relative_factor: float


@dataclass(frozen=True)
class GalvanizedDesignation:
    designation: str
    weight_oz_ft2: float
    thickness_um_per_side: float

    @property
    def thickness_mils_per_side(self) -> float:
        return self.thickness_um_per_side / 25.4

    @property
    def display_um(self) -> int:
        return int(round(self.thickness_um_per_side))

    @property
    def display_mils(self) -> float:
        return round(self.thickness_mils_per_side, 2)


def galvanized_thickness_um_per_side(weight_oz_ft2: float) -> float:
    """Convert total both-side galvanized coating weight to zinc thickness per side."""
    return float(weight_oz_ft2) * OZ_FT2_TO_G_M2 / (2.0 * ZINC_DENSITY_G_CM3)


STANDARD_GALVANIZED_DESIGNATIONS: tuple[GalvanizedDesignation, ...] = tuple(
    GalvanizedDesignation(
        designation=f"G{designation}",
        weight_oz_ft2=designation / 100.0,
        thickness_um_per_side=galvanized_thickness_um_per_side(designation / 100.0),
    )
    for designation in (90, 115, 140, 165, 185, 235)
)


MATERIALS: dict[str, MaterialSpec] = {
    "zinc_hdg": MaterialSpec(
        id="zinc_hdg",
        label="Zinc / HDG / G90 baseline",
        short_label="Zinc/HDG",
        kind="zinc_coating",
        default_relative_factor=1.0,
    ),
    "magnelis": MaterialSpec(
        id="magnelis",
        label="Magnelis Zn-Al-Mg",
        short_label="Magnelis",
        kind="zinc_coating",
        default_relative_factor=2.5,
    ),
    "zam": MaterialSpec(
        id="zam",
        label="ZAM Zn-Al-Mg",
        short_label="ZAM",
        kind="zinc_coating",
        default_relative_factor=2.5,
    ),
    "carbon_steel": MaterialSpec(
        id="carbon_steel",
        label="Bare carbon steel allowance",
        short_label="Carbon steel",
        kind="steel_allowance",
        default_relative_factor=1.0,
    ),
}

MATERIAL_ORDER = ("zinc_hdg", "magnelis", "zam", "carbon_steel")


def material_options() -> list[str]:
    return [MATERIALS[key].label for key in MATERIAL_ORDER]


def material_from_label(label: str) -> MaterialSpec:
    for material_id in MATERIAL_ORDER:
        material = MATERIALS[material_id]
        if material.label == label:
            return material
    raise ValueError(f"Unknown material label {label!r}.")


def required_coating_um(zinc_loss_um: Any, relative_factor: float = 1.0):
    """Return coating thickness needed for a zinc-based coating system."""
    factor = max(float(relative_factor), 1e-9)
    return np.asarray(zinc_loss_um, dtype=float) / factor


def g90_equivalents(coating_um: Any):
    """Return G90 equivalents, based on zinc coating thickness per exposed side."""
    return np.asarray(coating_um, dtype=float) / G90_ZINC_UM_PER_SIDE


def galvanized_designation_for_thickness(thickness_um: float) -> GalvanizedDesignation | None:
    """Return the lightest standard galvanized designation meeting a per-side thickness."""
    for designation in STANDARD_GALVANIZED_DESIGNATIONS:
        if float(thickness_um) <= designation.thickness_um_per_side:
            return designation
    return None


def galvanized_designation_label(thickness_um: float) -> str:
    designation = galvanized_designation_for_thickness(thickness_um)
    if designation is None:
        return f">{STANDARD_GALVANIZED_DESIGNATIONS[-1].designation}"
    return designation.designation
