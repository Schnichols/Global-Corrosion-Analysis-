from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

from coating import MaterialSpec, STANDARD_GALVANIZED_DESIGNATIONS, required_coating_um
from corrosion_model import ZINC_B1, ZINC_B2, iso_9224_zinc_loss_um
from regions import RegionSpec
from soil_risk import SOIL_RISK_LEVEL_STYLES, SoilRiskRegion

PRODUCT_RECOMMENDATION_LABELS = {
    0: "Low corrosion product",
    1: "Medium corrosion product",
    2: "High corrosion product",
}

COUNTRY_BOUNDARY_PATH = Path(__file__).resolve().parent / "data" / "country_boundary_rings.json"
COUNTRY_BOUNDARY_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_50m_admin_0_countries.geojson"
)
RCORR_COLOR_MAX = 25.0
RCORR_CATEGORY_BANDS = [
    ("C1", 0.0, 0.1, "#16a34a"),
    ("C2", 0.1, 0.7, "#86efac"),
    ("C3", 0.7, 2.1, "#fde047"),
    ("C4", 2.1, 4.2, "#f97316"),
    ("C5", 4.2, 8.4, "#dc2626"),
    ("CX", 8.4, RCORR_COLOR_MAX, "#7e22ce"),
]
RCORR_CATEGORY_TICKTEXT = [
    "C1<br>0-0.1",
    "C2<br>0.1-0.7",
    "C3<br>0.7-2.1",
    "C4<br>2.1-4.2",
    "C5<br>4.2-8.4",
    "CX<br>8.4-25",
]

GALVANIZED_THICKNESS_COLORS = (
    "#16a34a",
    "#86efac",
    "#fde047",
    "#fdba74",
    "#f97316",
    "#dc2626",
    "#7e22ce",
)


def _galvanized_thickness_bands() -> list[tuple[str, float, float, str]]:
    bands: list[tuple[str, float, float, str]] = []
    low = 0.0
    for idx, designation in enumerate(STANDARD_GALVANIZED_DESIGNATIONS):
        label = f"{designation.designation} {designation.display_um} um"
        bands.append((label, low, designation.thickness_um_per_side, GALVANIZED_THICKNESS_COLORS[idx]))
        low = designation.thickness_um_per_side
    top = STANDARD_GALVANIZED_DESIGNATIONS[-1]
    bands.append(
        (
            f">{top.designation} >{top.display_um} um",
            low,
            np.inf,
            GALVANIZED_THICKNESS_COLORS[-1],
        )
    )
    return bands


GALVANIZED_THICKNESS_BANDS = _galvanized_thickness_bands()

STEEL_ALLOWANCE_BANDS = [
    ("<=100 um", 0.0, 100.0, "#16a34a"),
    ("100-200 um", 100.0, 200.0, "#fde047"),
    ("200-400 um", 200.0, 400.0, "#fdba74"),
    ("400-650 um", 400.0, 650.0, "#c2410c"),
    ("650-1500 um", 650.0, 1500.0, "#7f1d1d"),
    (">1500 um", 1500.0, np.inf, "#7e22ce"),
]


def product_recommendation_code(loss_b2_um):
    """Classify project-life B2 zinc loss into product recommendation bands."""
    loss = np.asarray(loss_b2_um, dtype=float)
    out = np.full(loss.shape, np.nan)
    out[loss < 20.0] = 0
    out[(loss >= 20.0) & (loss <= 45.0)] = 1
    out[loss > 45.0] = 2
    if out.shape == ():
        return float(out.item())
    return out


def product_recommendation_label(loss_b2_um: float) -> str:
    code = product_recommendation_code(loss_b2_um)
    if np.isnan(code):
        return "Unavailable"
    return PRODUCT_RECOMMENDATION_LABELS[int(code)]


def _grid(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = df.pivot_table(index="lat", columns="lon", values=value_col, aggfunc="mean").sort_index()
    return grid.index.to_numpy(dtype=float), grid.columns.to_numpy(dtype=float), grid.to_numpy(dtype=float)


def rcorr_category_colorscale() -> list[list[float | str]]:
    """Return a stepped numeric Plotly colorscale matching ISO 9223 zinc categories."""
    colorscale: list[list[float | str]] = []
    for _, low, high, color in RCORR_CATEGORY_BANDS:
        colorscale.append([low / RCORR_COLOR_MAX, color])
        colorscale.append([high / RCORR_COLOR_MAX, color])
    return colorscale


def rcorr_category_code(rcorr_um_y: Any):
    """Classify Rcorr into a 0-based category code for equal-width map legends."""
    values = np.asarray(rcorr_um_y, dtype=float)
    codes = np.full(values.shape, np.nan)
    previous_high = -np.inf
    for idx, (_, _, high, _) in enumerate(RCORR_CATEGORY_BANDS):
        codes[(values > previous_high) & (values <= high)] = idx
        previous_high = high
    codes[values > RCORR_COLOR_MAX] = len(RCORR_CATEGORY_BANDS) - 1
    if codes.shape == ():
        return float(codes.item())
    return codes


def rcorr_category_text(category_codes: Any):
    codes = np.asarray(category_codes, dtype=float)
    text = np.full(codes.shape, "", dtype=object)
    for idx, (label, _, _, _) in enumerate(RCORR_CATEGORY_BANDS):
        text[codes == idx] = label
    if text.shape == ():
        return str(text.item())
    return text


def rcorr_discrete_colorscale() -> list[list[float | str]]:
    """Return an equal-width stepped colorscale for category-code maps."""
    colors = [color for _, _, _, color in RCORR_CATEGORY_BANDS]
    n = len(colors)
    colorscale: list[list[float | str]] = []
    for idx, color in enumerate(colors):
        low = idx / n
        high = (idx + 1) / n
        colorscale.append([low, color])
        colorscale.append([high, color])
    return colorscale


def rcorr_category_colorbar() -> dict[str, Any]:
    """Return Plotly colorbar settings for the Rcorr category palette."""
    tickvals = [idx + 0.5 for idx in range(len(RCORR_CATEGORY_BANDS))]
    return dict(title="Rcorr<br>um/year", tickmode="array", tickvals=tickvals, ticktext=RCORR_CATEGORY_TICKTEXT)


def _band_code(values: Any, bands: list[tuple[str, float, float, str]]):
    arr = np.asarray(values, dtype=float)
    codes = np.full(arr.shape, np.nan)
    for idx, (_, low, high, _) in enumerate(bands):
        if np.isinf(high):
            mask = arr > low
        elif idx == 0:
            mask = (arr >= low) & (arr <= high)
        else:
            mask = (arr > low) & (arr <= high)
        codes[mask] = idx
    if codes.shape == ():
        return float(codes.item())
    return codes


def _band_colorscale(bands: list[tuple[str, float, float, str]]) -> list[list[float | str]]:
    colorscale: list[list[float | str]] = []
    n = len(bands)
    for idx, (_, _, _, color) in enumerate(bands):
        colorscale.append([idx / n, color])
        colorscale.append([(idx + 1) / n, color])
    return colorscale


def _band_colorbar(title: str, bands: list[tuple[str, float, float, str]]) -> dict[str, Any]:
    tickvals = [idx + 0.5 for idx in range(len(bands))]
    ticktext = [label.replace(" ", "<br>") for label, _, _, _ in bands]
    return dict(title=title, tickmode="array", tickvals=tickvals, ticktext=ticktext)


def _band_text(codes: Any, bands: list[tuple[str, float, float, str]]):
    arr = np.asarray(codes, dtype=float)
    text = np.full(arr.shape, "", dtype=object)
    for idx, (label, _, _, _) in enumerate(bands):
        text[arr == idx] = label
    if text.shape == ():
        return str(text.item())
    return text


def _axis_ranges(
    lats: np.ndarray,
    lons: np.ndarray,
    region: RegionSpec | None,
) -> tuple[list[float], list[float]]:
    if region is not None:
        return region.lon_range, region.lat_range
    lon_pad = max((float(np.nanmax(lons)) - float(np.nanmin(lons))) * 0.03, 0.5)
    lat_pad = max((float(np.nanmax(lats)) - float(np.nanmin(lats))) * 0.03, 0.5)
    return [float(np.nanmin(lons)) - lon_pad, float(np.nanmax(lons)) + lon_pad], [
        float(np.nanmin(lats)) - lat_pad,
        float(np.nanmax(lats)) + lat_pad,
    ]


def _selected_marker(selected_lat: float | None, selected_lon: float | None) -> go.Scatter | None:
    if selected_lat is None or selected_lon is None:
        return None
    return go.Scatter(
        x=[selected_lon],
        y=[selected_lat],
        name="selected",
        mode="markers+text",
        text=["selected"],
        textposition="top center",
        marker=dict(symbol="x", size=12, line=dict(width=2)),
        hovertemplate="selected<br>lon=%{x:.4f}<br>lat=%{y:.4f}<extra></extra>",
        showlegend=False,
    )


@lru_cache(maxsize=1)
def _load_country_boundaries() -> tuple[dict[str, Any], ...]:
    if COUNTRY_BOUNDARY_PATH.exists():
        payload = json.loads(COUNTRY_BOUNDARY_PATH.read_text(encoding="utf-8"))
        return tuple(payload.get("features", ()))

    try:
        response = requests.get(COUNTRY_BOUNDARY_URL, timeout=20)
        response.raise_for_status()
    except requests.RequestException:
        return tuple()

    return _country_features_from_geojson(response.json())


def _geometry_rings(geometry: dict[str, Any]) -> list[list[list[float]]]:
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    rings: list[list[list[float]]] = []

    if geom_type == "Polygon":
        polygons = [coordinates]
    elif geom_type == "MultiPolygon":
        polygons = coordinates
    else:
        return rings

    for polygon in polygons:
        for ring in polygon:
            clean_ring = [[float(point[0]), float(point[1])] for point in ring if len(point) >= 2]
            if len(clean_ring) >= 2:
                rings.append(clean_ring)
    return rings


def _country_features_from_geojson(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    features: list[dict[str, Any]] = []
    for raw_feature in payload.get("features", []):
        rings = _geometry_rings(raw_feature.get("geometry") or {})
        if not rings:
            continue
        xs = [point[0] for ring in rings for point in ring]
        ys = [point[1] for ring in rings for point in ring]
        properties = raw_feature.get("properties") or {}
        features.append(
            {
                "name": properties.get("NAME") or properties.get("ADMIN") or "country boundary",
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "rings": rings,
            }
        )
    return tuple(features)


def _bbox_intersects(a: list[float], b: list[float], *, pad: float = 0.0) -> bool:
    return not (
        a[2] < b[0] - pad
        or a[0] > b[2] + pad
        or a[3] < b[1] - pad
        or a[1] > b[3] + pad
    )


def _add_country_boundaries(fig: go.Figure, region: RegionSpec | None) -> None:
    if region is None:
        return
    region_bbox = [region.lon_min, region.lat_min, region.lon_max, region.lat_max]
    for feature in _load_country_boundaries():
        if not _bbox_intersects(feature["bbox"], region_bbox, pad=1.0):
            continue
        for ring in feature["rings"]:
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            ring_bbox = [min(xs), min(ys), max(xs), max(ys)]
            if not _bbox_intersects(ring_bbox, region_bbox, pad=1.0):
                continue
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name=feature.get("name", "country boundary"),
                    line=dict(color="rgba(30, 41, 59, 0.82)", width=0.8),
                    hoverinfo="skip",
                    hovertemplate=None,
                    showlegend=False,
                )
            )


def _add_soil_risk_regions(fig: go.Figure, soil_risk_regions: list[SoilRiskRegion] | None) -> None:
    if not soil_risk_regions:
        return

    shown_levels: set[str] = set()
    for risk_region in soil_risk_regions:
        style = SOIL_RISK_LEVEL_STYLES[risk_region.risk_level]
        xs = [point[0] for point in risk_region.polygon]
        ys = [point[1] for point in risk_region.polygon]
        showlegend = risk_region.risk_level not in shown_levels
        shown_levels.add(risk_region.risk_level)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                name=f"Soil/salt risk - {style['label']}",
                line=dict(color=style["line"], width=1.2, dash="dash"),
                fillcolor=style["fill"],
                hovertemplate=(
                    f"{risk_region.label}<br>"
                    f"Risk={style['label']}<br>"
                    f"{risk_region.basis}<extra></extra>"
                ),
                showlegend=showlegend,
                legendgroup=f"soil-risk-{risk_region.risk_level}",
            )
        )


def make_contour_figure(
    df: pd.DataFrame,
    selected_lat: float | None = None,
    selected_lon: float | None = None,
    *,
    region: RegionSpec | None = None,
    soil_risk_regions: list[SoilRiskRegion] | None = None,
):
    """Create a Plotly contour figure for a zinc Rcorr surface."""
    required = {"lat", "lon", "Rcorr_um_y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Surface CSV missing columns: {sorted(missing)}")

    lats, lons, z = _grid(df, "Rcorr_um_y")
    category_codes = rcorr_category_code(z)
    category_text = rcorr_category_text(category_codes)
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=lons,
            y=lats,
            z=category_codes + 0.5,
            text=category_text,
            customdata=z,
            colorscale=rcorr_discrete_colorscale(),
            zmin=0.0,
            zmax=len(RCORR_CATEGORY_BANDS),
            colorbar=rcorr_category_colorbar(),
            hovertemplate=(
                "lon=%{x:.3f}<br>lat=%{y:.3f}<br>"
                "Rcorr=%{customdata:.3f} um/y<br>Category=%{text}<extra></extra>"
            ),
            hoverongaps=False,
        )
    )
    fig.add_trace(
        go.Contour(
            x=lons,
            y=lats,
            z=z,
            contours=dict(showlabels=True, labelfont=dict(size=10), coloring="lines"),
            colorscale=[[0.0, "rgba(31, 41, 55, 0.65)"], [1.0, "rgba(31, 41, 55, 0.65)"]],
            showscale=False,
            hoverinfo="skip",
            ncontours=18,
            connectgaps=False,
            line=dict(width=0.7, color="rgba(31, 41, 55, 0.55)", smoothing=0.55),
        )
    )
    _add_soil_risk_regions(fig, soil_risk_regions)
    _add_country_boundaries(fig, region)
    marker = _selected_marker(selected_lat, selected_lon)
    if marker is not None:
        fig.add_trace(marker)

    title_region = region.label if region is not None else "Selected Region"
    lon_range, lat_range = _axis_ranges(lats, lons, region)
    fig.update_layout(
        title=f"{title_region} Zinc First-Year Corrosion Rate, ISO 9223 Rcorr",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=680,
        plot_bgcolor="rgba(248,250,252,0.1)",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(range=lon_range)
    fig.update_yaxes(range=lat_range)
    return fig


def make_product_recommendation_figure(
    df: pd.DataFrame,
    selected_lat: float | None = None,
    selected_lon: float | None = None,
    *,
    loss_col: str = "D30_B2_um",
    project_life_years: int = 30,
    region: RegionSpec | None = None,
    soil_risk_regions: list[SoilRiskRegion] | None = None,
):
    """Create a discrete product recommendation map from a project-life B2 loss column."""
    required = {"lat", "lon", loss_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Surface CSV missing columns: {sorted(missing)}")

    lats, lons, loss = _grid(df, loss_col)
    z = product_recommendation_code(loss)
    text = np.full(z.shape, "", dtype=object)
    for code, label in PRODUCT_RECOMMENDATION_LABELS.items():
        text[z == code] = label

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=lons,
            y=lats,
            z=z,
            zmin=0,
            zmax=2,
            text=text,
            customdata=loss,
            colorscale=[
                [0.0, "#2f855a"],
                [0.25, "#2f855a"],
                [0.25, "#d69e2e"],
                [0.75, "#d69e2e"],
                [0.75, "#c53030"],
                [1.0, "#c53030"],
            ],
            colorbar=dict(
                title="Product",
                tickmode="array",
                tickvals=[0, 1, 2],
                ticktext=["Low<br><20 um", "Medium<br>20-45 um", "High<br>>45 um"],
            ),
            hoverongaps=False,
            hovertemplate=(
                "lon=%{x:.3f}<br>lat=%{y:.3f}<br>%{text}<br>"
                f"D{project_life_years} B2=%{{customdata:.1f}} um<extra></extra>"
            ),
        )
    )
    _add_soil_risk_regions(fig, soil_risk_regions)
    _add_country_boundaries(fig, region)
    marker = _selected_marker(selected_lat, selected_lon)
    if marker is not None:
        fig.add_trace(marker)

    title_region = region.label if region is not None else "Selected Region"
    lon_range, lat_range = _axis_ranges(lats, lons, region)
    fig.update_layout(
        title=f"{title_region} Zinc Product Recommendation by {project_life_years}-Year B2 Loss",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=680,
        plot_bgcolor="rgba(248,250,252,0.1)",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(range=lon_range)
    fig.update_yaxes(range=lat_range)
    return fig


def make_coating_requirement_figure(
    df: pd.DataFrame,
    selected_lat: float | None = None,
    selected_lon: float | None = None,
    *,
    loss_col: str = "D30_B2_um",
    steel_loss_col: str = "Steel_D30_B2_um",
    project_life_years: int = 30,
    material: MaterialSpec,
    relative_factor: float = 1.0,
    region: RegionSpec | None = None,
    soil_risk_regions: list[SoilRiskRegion] | None = None,
):
    """Create a material demand map for project-life coating thickness or steel allowance."""
    required = {"lat", "lon", loss_col}
    if material.kind == "steel_allowance":
        required.add(steel_loss_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Surface CSV missing columns: {sorted(missing)}")

    plot_df = df[["lat", "lon", loss_col]].copy()
    if material.kind == "steel_allowance":
        plot_df["plot_value"] = pd.to_numeric(df[steel_loss_col], errors="coerce")
        plot_df["display_um"] = plot_df["plot_value"]
        plot_df["g90_eq"] = np.nan
        bands = STEEL_ALLOWANCE_BANDS
        colorbar_title = "Steel<br>allowance"
        hover_value = f"D{project_life_years} steel B2=%{{customdata[1]:.0f}} um"
        title_suffix = "Carbon Steel Atmospheric Corrosion Allowance"
    else:
        plot_df["display_um"] = required_coating_um(df[loss_col], relative_factor)
        plot_df["plot_value"] = plot_df["display_um"]
        bands = GALVANIZED_THICKNESS_BANDS
        colorbar_title = "Min.<br>galv<br>thickness"
        hover_value = (
            f"D{project_life_years} zinc B2=%{{customdata[0]:.1f}} um<br>"
            f"Required {material.short_label}=%{{customdata[1]:.1f}} um/side<br>"
            "Minimum standard=%{text}"
        )
        if material.id == "zinc_hdg":
            title_suffix = f"Minimum Galvanized Coating by {project_life_years}-Year B2 Loss"
        else:
            title_suffix = f"{material.short_label} Required Coating by {project_life_years}-Year B2 Loss"

    lats, lons, plot_value = _grid(plot_df, "plot_value")
    _, _, source_loss = _grid(plot_df, loss_col)
    _, _, display_um = _grid(plot_df, "display_um")
    customdata = np.dstack([source_loss, display_um])
    category_codes = _band_code(plot_value, bands)
    category_text = _band_text(category_codes, bands)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=lons,
            y=lats,
            z=category_codes + 0.5,
            text=category_text,
            customdata=customdata,
            colorscale=_band_colorscale(bands),
            zmin=0.0,
            zmax=len(bands),
            colorbar=_band_colorbar(colorbar_title, bands),
            hovertemplate=f"lon=%{{x:.3f}}<br>lat=%{{y:.3f}}<br>{hover_value}<extra></extra>",
            hoverongaps=False,
        )
    )
    fig.add_trace(
        go.Contour(
            x=lons,
            y=lats,
            z=plot_value,
            contours=dict(showlabels=True, labelfont=dict(size=10), coloring="lines"),
            colorscale=[[0.0, "rgba(31, 41, 55, 0.65)"], [1.0, "rgba(31, 41, 55, 0.65)"]],
            showscale=False,
            hoverinfo="skip",
            ncontours=12,
            connectgaps=False,
            line=dict(width=0.7, color="rgba(31, 41, 55, 0.55)", smoothing=0.55),
        )
    )
    _add_soil_risk_regions(fig, soil_risk_regions)
    _add_country_boundaries(fig, region)
    marker = _selected_marker(selected_lat, selected_lon)
    if marker is not None:
        fig.add_trace(marker)

    title_region = region.label if region is not None else "Selected Region"
    lon_range, lat_range = _axis_ranges(lats, lons, region)
    fig.update_layout(
        title=f"{title_region} {title_suffix}",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=680,
        plot_bgcolor="rgba(248,250,252,0.1)",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(range=lon_range)
    fig.update_yaxes(range=lat_range)
    return fig


def make_life_chart(
    rcorr_um_y: float,
    *,
    project_life_years: int = 30,
    title_prefix: str = "Selected site",
    after_20: str = "linear",
):
    """Return a Matplotlib project-life figure."""
    years = np.arange(1, max(41, project_life_years + 11), dtype=float)
    selected_b1 = iso_9224_zinc_loss_um(rcorr_um_y, years, b=ZINC_B1, after_20=after_20)
    selected_b2 = iso_9224_zinc_loss_um(rcorr_um_y, years, b=ZINC_B2, after_20=after_20)

    refs: list[tuple[str, float, Any]] = [
        ("C2/C3 boundary: 0.7 um/y", 0.7, "--"),
        ("C3/C4 boundary: 2.1 um/y", 2.1, "--"),
        ("C4/C5 boundary: 4.2 um/y", 4.2, "--"),
        ("C5/CX boundary: 8.4 um/y", 8.4, "--"),
    ]

    d_b1 = float(iso_9224_zinc_loss_um(rcorr_um_y, project_life_years, b=ZINC_B1, after_20=after_20))
    d_b2 = float(iso_9224_zinc_loss_um(rcorr_um_y, project_life_years, b=ZINC_B2, after_20=after_20))
    xmax = max(60.0, np.nanmax(selected_b2) * 1.15, d_b2 * 1.3)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=140)
    for label, r, style in refs:
        x = iso_9224_zinc_loss_um(r, years, b=ZINC_B1, after_20=after_20)
        ax.plot(x, years, linestyle=style, linewidth=1.5, label=label, alpha=0.75)

    ax.plot(selected_b1, years, marker="o", markersize=3.2, linewidth=2.6, label=f"{title_prefix} B1, Rcorr={rcorr_um_y:.3f} um/y")
    ax.plot(selected_b2, years, linestyle="--", linewidth=2.2, label=f"{title_prefix} B2 upper")
    ax.axhline(project_life_years, color="black", linewidth=1.8)
    ax.scatter([d_b1, d_b2], [project_life_years, project_life_years], s=45, zorder=5)
    ax.annotate(
        f"{project_life_years}-yr zinc loss\nB1 {d_b1:.1f} um\nB2 {d_b2:.1f} um",
        xy=(d_b2, project_life_years),
        xytext=(min(xmax * 0.72, d_b2 + xmax * 0.08), project_life_years + 2.5),
        arrowprops=dict(arrowstyle="->", lw=1.2),
        fontsize=10,
        ha="left",
    )

    ax.set_title(f"{title_prefix} - Zinc Coating Thickness Required vs Project Life")
    ax.set_xlabel("Zinc Coating Thickness Required / Zinc Loss (um)")
    ax.set_ylabel("Project Life (years)")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, max(40, project_life_years + 8))
    ax.grid(True, which="major", alpha=0.45)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig
