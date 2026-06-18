from __future__ import annotations

from pathlib import Path
import sys

import json

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import apply_sulfate_floor, filter_surface_to_region
from plotting import COUNTRY_BOUNDARY_PATH, RCORR_CATEGORY_BANDS
from regions import DEFAULT_DEMO_SURFACE, REGIONS
from soil_risk import SOIL_RISK_LEVEL_STYLES, soil_risk_regions_for_region
from surface import ZincSurface

OUTPUT_PATH = ROOT / "outputs" / "middle_east_project_rcorr_locations.png"

PROJECT_LOCATIONS = [
    {"name": "PIF 7/8 - Bisha proxy", "lat": 19.9863, "lon": 42.3938, "offset": (22, -16)},
    {
        "name": "PIF Humaij proxy / SPPC South Al Ula",
        "lat": 26.5524,
        "lon": 37.9683,
        "offset": (-132, 52),
    },
    {"name": "PIF 7/8 - Khulis", "lat": 22.1441, "lon": 39.3643, "offset": (-122, -30)},
    {"name": "PIF 7/8 - Afif", "lat": 23.8826, "lon": 42.9206, "offset": (20, 26)},
    {"name": "SPPC 7 - Tabarjal 2 proxy", "lat": 30.5000, "lon": 38.2167, "offset": (-130, 34)},
    {"name": "SPPC 7 - Mawqaq proxy", "lat": 27.3786, "lon": 41.1950, "offset": (22, 34)},
    {"name": "SPPC 7 - Tathleeth", "lat": 19.5226, "lon": 43.5551, "offset": (22, 24)},
    {"name": "Shaqaya / Al Shagaya", "lat": 29.2025, "lon": 47.0708, "offset": (24, 26)},
    {"name": "NEOM Shigry/Shiqri proxy", "lat": 27.9865, "lon": 35.9969, "offset": (-116, -34)},
    {"name": "MOD King Khalid Air Base", "lat": 18.2973, "lon": 42.8035, "offset": (24, -38)},
]


def grid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = df.pivot_table(index="lat", columns="lon", values="Rcorr_um_y", aggfunc="mean").sort_index()
    return pivot.columns.to_numpy(float), pivot.index.to_numpy(float), pivot.to_numpy(float)


def country_boundary_rings(region_bbox: list[float]) -> list[tuple[list[float], list[float]]]:
    if not COUNTRY_BOUNDARY_PATH.exists():
        return []

    payload = json.loads(COUNTRY_BOUNDARY_PATH.read_text(encoding="utf-8"))
    rings: list[tuple[list[float], list[float]]] = []
    for feature in payload.get("features", []):
        bbox = feature["bbox"]
        intersects = not (
            bbox[2] < region_bbox[0] - 1.0
            or bbox[0] > region_bbox[2] + 1.0
            or bbox[3] < region_bbox[1] - 1.0
            or bbox[1] > region_bbox[3] + 1.0
        )
        if not intersects:
            continue
        for ring in feature["rings"]:
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            rings.append((xs, ys))
    return rings


def stepped_colormap() -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm, list[float]]:
    colors = [color for _, _, _, color in RCORR_CATEGORY_BANDS]
    bounds = [RCORR_CATEGORY_BANDS[0][1], *[high for _, _, high, _ in RCORR_CATEGORY_BANDS]]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N, clip=True)
    return cmap, norm, bounds


def add_soil_risk_overlay(ax: plt.Axes, region_id: str) -> None:
    handles: list[mpatches.Patch] = []
    shown_levels: set[str] = set()
    for risk_region in soil_risk_regions_for_region(region_id):
        style = SOIL_RISK_LEVEL_STYLES[risk_region.risk_level]
        xs = [point[0] for point in risk_region.polygon]
        ys = [point[1] for point in risk_region.polygon]
        alpha = 0.16 if risk_region.risk_level == "high" else 0.12
        ax.fill(xs, ys, facecolor=style["line"], alpha=alpha, zorder=2)
        ax.plot(xs, ys, color=style["line"], linewidth=0.9, linestyle="--", alpha=0.82, zorder=3)
        if risk_region.risk_level not in shown_levels:
            handles.append(
                mpatches.Patch(
                    facecolor=style["line"],
                    edgecolor=style["line"],
                    alpha=alpha,
                    label=f"Soil/salt risk - {style['label']}",
                )
            )
            shown_levels.add(risk_region.risk_level)
    if handles:
        ax.legend(
            handles=handles,
            loc="lower right",
            fontsize=8,
            title="Screening overlay",
            title_fontsize=8,
            framealpha=0.9,
        )


def main() -> None:
    region = REGIONS["middle_east"]
    source = ROOT / DEFAULT_DEMO_SURFACE
    raw_df = pd.read_csv(source)
    df = filter_surface_to_region(raw_df, region)
    analysis_df = apply_sulfate_floor(df, 5.0, project_life_years=30, after_20="linear")
    surface = ZincSurface(analysis_df)

    x, y, z = grid(analysis_df)
    cmap, norm, bounds = stepped_colormap()

    fig, ax = plt.subplots(figsize=(18, 12), dpi=180)
    filled = ax.contourf(x, y, z, levels=bounds, cmap=cmap, norm=norm, extend="max", alpha=0.88)
    category_lines = ax.contour(x, y, z, levels=bounds[1:-1], colors="#334155", linewidths=0.65, alpha=0.7)
    ax.clabel(category_lines, fmt="%.1f", fontsize=8, inline=True)
    add_soil_risk_overlay(ax, region.id)

    region_bbox = [region.lon_min, region.lat_min, region.lon_max, region.lat_max]
    for xs, ys in country_boundary_rings(region_bbox):
        ax.plot(xs, ys, color="#111827", linewidth=0.65, alpha=0.82)

    for project in PROJECT_LOCATIONS:
        result = surface.sample(project["lat"], project["lon"], after_20="linear")
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
        label = f"{project['name']}\nRcorr={result.rcorr_um_y:.3f} um/y"
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
    cbar.set_label("ISO 9223 zinc Rcorr category (um/year)")
    cbar.set_ticks([(low + high) / 2.0 for _, low, high, _ in RCORR_CATEGORY_BANDS])
    cbar.set_ticklabels([f"{cat}\n{low:g}-{high:g}" for cat, low, high, _ in RCORR_CATEGORY_BANDS])

    ax.set_title("Middle East Zinc Rcorr Map with Project Locations", fontsize=18, weight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(region.lon_min, region.lon_max)
    ax.set_ylim(region.lat_min, region.lat_max)
    ax.grid(color="#94a3b8", linewidth=0.45, alpha=0.45)
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.01,
        0.01,
        "Synthetic demo surface with sulfate floor = 5.0 mg/m2/day; locations include public proxies where exact coordinates are not published.",
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
