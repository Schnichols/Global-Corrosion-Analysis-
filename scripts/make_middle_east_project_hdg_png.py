from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import apply_sulfate_floor, filter_surface_to_region
from coating import (
    STANDARD_GALVANIZED_DESIGNATIONS,
    galvanized_designation_label,
    required_coating_um,
)
from make_middle_east_project_png import PROJECT_LOCATIONS, add_soil_risk_overlay, country_boundary_rings, grid
from plotting import GALVANIZED_THICKNESS_BANDS
from regions import DEFAULT_DEMO_SURFACE, REGIONS
from surface import ZincSurface

OUTPUT_PATH = ROOT / "outputs" / "middle_east_project_hdg_designations.png"


def hdg_colormap(max_required_um: float) -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm, list[float]]:
    colors = [color for _, _, _, color in GALVANIZED_THICKNESS_BANDS]
    finite_highs = [high for _, _, high, _ in GALVANIZED_THICKNESS_BANDS if not np.isinf(high)]
    upper = max(float(max_required_um), finite_highs[-1]) * 1.001
    bounds = [GALVANIZED_THICKNESS_BANDS[0][1], *finite_highs, upper]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N, clip=True)
    return cmap, norm, bounds


def hdg_tick_labels() -> list[str]:
    labels: list[str] = []
    for designation in STANDARD_GALVANIZED_DESIGNATIONS:
        labels.append(f"{designation.designation}\n{designation.display_um} um/side")
    top = STANDARD_GALVANIZED_DESIGNATIONS[-1]
    labels.append(f">{top.designation}\n>{top.display_um} um/side")
    return labels


def main() -> None:
    region = REGIONS["middle_east"]
    source = ROOT / DEFAULT_DEMO_SURFACE
    raw_df = pd.read_csv(source)
    df = filter_surface_to_region(raw_df, region)
    analysis_df = apply_sulfate_floor(df, 5.0, project_life_years=30, after_20="linear")
    surface = ZincSurface(analysis_df)

    hdg_df = analysis_df[["lat", "lon"]].copy()
    hdg_df["Rcorr_um_y"] = required_coating_um(analysis_df["D30_B2_um"], 1.0)
    x, y, z = grid(hdg_df)
    cmap, norm, bounds = hdg_colormap(float(np.nanmax(z)))

    fig, ax = plt.subplots(figsize=(18, 12), dpi=180)
    filled = ax.contourf(x, y, z, levels=bounds, cmap=cmap, norm=norm, extend="max", alpha=0.88)
    contour_levels = [item.thickness_um_per_side for item in STANDARD_GALVANIZED_DESIGNATIONS]
    category_lines = ax.contour(x, y, z, levels=contour_levels, colors="#334155", linewidths=0.65, alpha=0.7)
    ax.clabel(category_lines, fmt=lambda value: f"{value:.0f}", fontsize=8, inline=True)
    add_soil_risk_overlay(ax, region.id)

    region_bbox = [region.lon_min, region.lat_min, region.lon_max, region.lat_max]
    for xs, ys in country_boundary_rings(region_bbox):
        ax.plot(xs, ys, color="#111827", linewidth=0.65, alpha=0.82)

    for project in PROJECT_LOCATIONS:
        result = surface.sample(project["lat"], project["lon"], after_20="linear")
        required_um = result.d30_b2_um
        min_designation = galvanized_designation_label(required_um)
        ax.scatter(
            project["lon"],
            project["lat"],
            s=70,
            marker="o",
            facecolor="#0f172a",
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        )
        label = f"{project['name']}\n{min_designation}, {required_um:.1f} um/side"
        ax.annotate(
            label,
            xy=(project["lon"], project["lat"]),
            xytext=project["offset"],
            textcoords="offset points",
            ha="left" if project["offset"][0] >= 0 else "right",
            va="center",
            fontsize=8.5,
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#94a3b8", alpha=0.92),
            arrowprops=dict(arrowstyle="-", color="#334155", lw=0.85, shrinkA=0, shrinkB=4),
            zorder=6,
        )

    cbar = fig.colorbar(filled, ax=ax, shrink=0.72, pad=0.018)
    cbar.set_label("Minimum HDG/G designation by required zinc thickness")
    tick_values = [(low + high) / 2 for _, low, high, _ in GALVANIZED_THICKNESS_BANDS[:-1]]
    tick_values.append((STANDARD_GALVANIZED_DESIGNATIONS[-1].thickness_um_per_side + bounds[-1]) / 2)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels(hdg_tick_labels())

    ax.set_title("Middle East Minimum HDG Coating Map with Project Locations", fontsize=18, weight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(region.lon_min, region.lon_max)
    ax.set_ylim(region.lat_min, region.lat_max)
    ax.grid(color="#94a3b8", linewidth=0.45, alpha=0.45)
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.01,
        0.01,
        "Synthetic demo surface, 30-year B2 zinc loss, sulfate floor = 5.0 mg/m2/day; labels show minimum standard G designation and required zinc thickness per side.",
        transform=ax.transAxes,
        fontsize=8,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cbd5e1", alpha=0.86),
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
