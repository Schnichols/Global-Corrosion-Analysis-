from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import streamlit as st

from corrosion_model import (
    ISO9223_INTERVALS,
    STEEL_B1,
    STEEL_B2,
    ZINC_B1,
    ZINC_B2,
    iso_9223_steel_rcorr,
    iso_9223_zinc_rcorr,
    iso_9224_zinc_loss_um,
    iso_input_warnings,
    zinc_corrosivity_category,
)
from coating import (
    STANDARD_GALVANIZED_DESIGNATIONS,
    galvanized_designation_for_thickness,
    galvanized_designation_label,
    material_from_label,
    material_options,
    required_coating_um,
)
from plotting import (
    make_coating_requirement_figure,
    make_contour_figure,
    make_life_chart,
    make_product_recommendation_figure,
    product_recommendation_label,
)
from project_presets import project_preset_by_label, project_presets_for_region
from regions import DEFAULT_DEMO_SURFACE, REGION_ORDER, REGIONS, RegionSpec, get_region, pick_default_data_path
from soil_risk import highest_soil_risk_for_point, soil_risk_regions_for_region
from surface import ZincSurface


@st.cache_data(show_spinner=False)
def load_df(path: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_csv(path)


def _is_default_demo_path(path: Path) -> bool:
    return path.as_posix().replace("\\", "/").endswith(DEFAULT_DEMO_SURFACE.as_posix())


def ensure_surface_csv(path: Path) -> str | None:
    if path.exists():
        return None
    if not _is_default_demo_path(path):
        return None

    from scripts.build_demo_surface import make_demo

    path.parent.mkdir(parents=True, exist_ok=True)
    demo = make_demo()
    demo.to_csv(path, index=False)
    return f"Generated synthetic demo surface at {path}."


def project_loss_columns(project_life_years: int) -> tuple[str, str]:
    years = int(project_life_years)
    return f"D{years}_B1_um", f"D{years}_B2_um"


def steel_loss_columns(project_life_years: int) -> tuple[str, str]:
    years = int(project_life_years)
    return f"Steel_D{years}_B1_um", f"Steel_D{years}_B2_um"


def filter_surface_to_region(df: pd.DataFrame, region: RegionSpec) -> pd.DataFrame:
    """Return rows for a selected region from single-region or multi-region surfaces."""
    if "region" in df.columns:
        region_values = df["region"].astype(str).str.lower()
        by_region = df[region_values == region.id].copy()
        if not by_region.empty:
            return by_region

    bounded = df[
        df["lat"].between(region.lat_min, region.lat_max)
        & df["lon"].between(region.lon_min, region.lon_max)
    ].copy()
    if not bounded.empty:
        return bounded
    return df.copy()


def apply_sulfate_floor(
    df: pd.DataFrame,
    sulfate_floor: float,
    *,
    project_life_years: int = 30,
    after_20: str = "linear",
) -> pd.DataFrame:
    """Return an analysis surface with Pd floored and project-life values recalculated."""
    out = df.copy()
    out["Pd_mg_m2_d"] = pd.to_numeric(out["Pd_mg_m2_d"], errors="coerce").clip(lower=float(sulfate_floor))
    rcorr = iso_9223_zinc_rcorr(
        out["T_C"],
        out["RH_pct"],
        out["Pd_mg_m2_d"],
        out["Sd_mg_m2_d"],
    )
    out["Rcorr_um_y"] = rcorr
    out["category"] = zinc_corrosivity_category(rcorr)
    d_b1_col, d_b2_col = project_loss_columns(project_life_years)
    out[d_b1_col] = iso_9224_zinc_loss_um(rcorr, project_life_years, b=ZINC_B1, after_20=after_20)
    out[d_b2_col] = iso_9224_zinc_loss_um(rcorr, project_life_years, b=ZINC_B2, after_20=after_20)
    steel_rcorr = iso_9223_steel_rcorr(
        out["T_C"],
        out["RH_pct"],
        out["Pd_mg_m2_d"],
        out["Sd_mg_m2_d"],
    )
    steel_b1_col, steel_b2_col = steel_loss_columns(project_life_years)
    out["Steel_Rcorr_um_y"] = steel_rcorr
    out[steel_b1_col] = iso_9224_zinc_loss_um(steel_rcorr, project_life_years, b=STEEL_B1, after_20=after_20)
    out[steel_b2_col] = iso_9224_zinc_loss_um(steel_rcorr, project_life_years, b=STEEL_B2, after_20=after_20)
    out["sulfate_floor_mg_m2_d"] = float(sulfate_floor)
    out["project_life_years"] = int(project_life_years)
    return out


def _region_index(region_id: str) -> int:
    try:
        return REGION_ORDER.index(region_id)
    except ValueError:
        return 0


def main(data_path: str | None = None, region_id: str = "conus"):
    st.set_page_config(page_title="Regional Zinc Corrosion Model", layout="wide")
    st.title("Regional zinc corrosion model - ISO 9223 / ISO 9224")

    with st.sidebar:
        st.header("Data and location")
        selected_region_label = st.selectbox(
            "Region",
            options=[REGIONS[key].label for key in REGION_ORDER],
            index=_region_index(region_id),
        )
        selected_region_id = next(key for key in REGION_ORDER if REGIONS[key].label == selected_region_label)
        region = get_region(selected_region_id)
        region_soil_risk_regions = soil_risk_regions_for_region(region.id)
        selected_path = Path(data_path) if data_path else pick_default_data_path(region.id)
        data_file = st.text_input("Surface CSV", value=str(selected_path), key=f"surface_csv_{region.id}")
        region_project_presets = project_presets_for_region(region.id)
        use_project_preset = st.checkbox(
            "Use project / portfolio preset",
            value=False,
            disabled=not region_project_presets,
            key=f"use_project_preset_{region.id}",
        )
        selected_project = None
        if use_project_preset and region_project_presets:
            selected_project_label = st.selectbox(
                "Project / owner portfolio",
                options=[preset.label for preset in region_project_presets],
            )
            selected_project = project_preset_by_label(region.id, selected_project_label)
            lat = float(selected_project.latitude)
            lon = float(selected_project.longitude)
            st.text_input("Preset latitude", value=f"{lat:.5f}", disabled=True)
            st.text_input("Preset longitude", value=f"{lon:.5f}", disabled=True)
            st.caption(selected_project.location_basis)
            if selected_project.notes:
                st.caption(selected_project.notes)
        else:
            if not region_project_presets:
                st.caption("No project presets are configured for this region yet.")
            lat = st.number_input(
                "Latitude",
                value=float(region.default_lat),
                min_value=float(region.lat_min),
                max_value=float(region.lat_max),
                step=0.01,
                format="%.5f",
                key=f"lat_{region.id}",
            )
            lon = st.number_input(
                "Longitude",
                value=float(region.default_lon),
                min_value=float(region.lon_min),
                max_value=float(region.lon_max),
                step=0.01,
                format="%.5f",
                key=f"lon_{region.id}",
            )
        project_life = st.number_input("Project life (years)", value=30, min_value=1, max_value=100, step=1)
        sulfate_floor = st.number_input(
            "Sulfate floor (mg/m2/day)",
            value=5.0,
            min_value=0.0,
            step=0.1,
            format="%.2f",
            help="Raises Pd_mg_m2_d values below this floor before calculating corrosion.",
        )
        after_20 = st.radio(
            "ISO 9224 treatment after 20 years",
            options=["linear", "power"],
            index=0,
            help="linear uses ISO 9224 Eq. 3 beyond 20 years; power uses Eq. 1 for all years.",
        )
        clip = st.checkbox(
            "Clip environmental inputs to ISO 9223 calibration intervals before calculating",
            value=False,
        )
        show_coating_map = st.toggle(
            "Show coating thickness map",
            value=False,
            help="Swaps the Rcorr category map for a project-life coating demand map with standard galvanized thickness bands.",
        )
        material_label = st.selectbox(
            "Material / coating",
            options=material_options(),
            disabled=not show_coating_map,
        )
        selected_material = material_from_label(material_label)
        if selected_material.kind == "zinc_coating":
            relative_factor = st.number_input(
                "Durability factor vs zinc",
                value=float(selected_material.default_relative_factor),
                min_value=0.1,
                max_value=20.0,
                step=0.1,
                format="%.2f",
                disabled=not show_coating_map,
                help="Required coating thickness is zinc B2 project-life loss divided by this factor.",
            )
        else:
            relative_factor = 1.0
        show_product_map = st.toggle(
            "Show product recommendation map",
            value=False,
            disabled=show_coating_map,
            help="Bands the project-life B2 loss as low (<20 um), medium (20-45 um), or high (>45 um).",
        )
        show_soil_risk_overlay = st.checkbox(
            "Show soil/salt-flat risk overlay",
            value=bool(region_soil_risk_regions),
            disabled=not region_soil_risk_regions,
            help=(
                "Screens broad salt flats, sabkha, and saline/sodic soil regions that may create "
                "wind-blown corrosive dust. This is not a soil corrosivity replacement."
            ),
        )

    path = Path(data_file)
    generated_demo_message = ensure_surface_csv(path)
    if not path.exists():
        st.error(
            f"Surface CSV not found: {path}. Use {DEFAULT_DEMO_SURFACE} for a synthetic smoke test "
            "or build a production surface first."
        )
        st.stop()

    is_demo = "sample_demo" in path.name.lower()
    if generated_demo_message:
        st.info(generated_demo_message)
    if is_demo:
        st.warning(
            "This app is running with a packaged synthetic demo surface so the UI can be smoke-tested. "
            "Do not use demo outputs for engineering decisions."
        )
    elif region.id != "conus":
        st.info(
            "For non-CONUS regions, confirm the surface CSV uses vetted SO2 and chloride deposition inputs. "
            "The bundled EPA/NADP TDep downloader is CONUS-focused."
        )

    mtime_ns = path.stat().st_mtime_ns
    raw_df = load_df(str(path), mtime_ns)
    df = filter_surface_to_region(raw_df, region)
    if df.empty:
        st.error(f"No data rows are available for {region.label} in {path}.")
        st.stop()

    project_life_years = int(project_life)
    d_b1_col, d_b2_col = project_loss_columns(project_life_years)
    steel_b1_col, steel_b2_col = steel_loss_columns(project_life_years)
    analysis_df = apply_sulfate_floor(
        df,
        sulfate_floor,
        project_life_years=project_life_years,
        after_20=after_20,
    )
    surface = ZincSurface(analysis_df)

    try:
        result = surface.sample(lat, lon, after_20=after_20, clip_to_iso_intervals=clip)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    d_project_b1 = float(iso_9224_zinc_loss_um(result.rcorr_um_y, project_life_years, b=ZINC_B1, after_20=after_20))
    d_project_b2 = float(iso_9224_zinc_loss_um(result.rcorr_um_y, project_life_years, b=ZINC_B2, after_20=after_20))
    steel_rcorr = float(
        iso_9223_steel_rcorr(
            result.T_C,
            result.RH_pct,
            result.Pd_mg_m2_d,
            result.Sd_mg_m2_d,
            clip_to_iso_intervals=clip,
        )
    )
    steel_project_b2 = float(
        iso_9224_zinc_loss_um(steel_rcorr, project_life_years, b=STEEL_B2, after_20=after_20)
    )
    selected_coating_um = float(required_coating_um(d_project_b2, relative_factor))
    selected_galv_designation = galvanized_designation_for_thickness(selected_coating_um)
    selected_galv_label = galvanized_designation_label(selected_coating_um)
    selected_standard_um = (
        selected_galv_designation.thickness_um_per_side if selected_galv_designation is not None else None
    )
    selected_product = product_recommendation_label(d_project_b2)
    selected_product_short = selected_product.replace(" corrosion product", "")
    soil_risk_match = highest_soil_risk_for_point(region.id, lat, lon)
    visible_soil_risk_regions = region_soil_risk_regions if show_soil_risk_overlay else []
    soil_risk_metric = soil_risk_match.risk_label if soil_risk_match else "Not flagged"
    if not region_soil_risk_regions:
        soil_risk_metric = "No layer"

    kpi_cols = st.columns(8)
    kpi_cols[0].metric("Rcorr", f"{result.rcorr_um_y:.3f} um/y")
    kpi_cols[1].metric("Category", result.category)
    kpi_cols[2].metric("Product", selected_product_short)
    kpi_cols[3].metric(f"{project_life}-yr loss B1/B2", f"{d_project_b1:.1f} / {d_project_b2:.1f} um")
    kpi_cols[4].metric("Temp", f"{result.T_C:.1f} C")
    kpi_cols[5].metric("RH", f"{result.RH_pct:.1f}%")
    kpi_cols[6].metric("SO2 / Cl dep.", f"{result.Pd_mg_m2_d:.2f} / {result.Sd_mg_m2_d:.2f}")
    kpi_cols[7].metric("Soil/salt risk", soil_risk_metric)

    warnings = iso_input_warnings(result.T_C, result.RH_pct, result.Pd_mg_m2_d, result.Sd_mg_m2_d)
    for warning in warnings:
        st.warning(warning)

    if soil_risk_match is not None:
        st.warning(
            f"Soil/salt-flat risk flag: {soil_risk_match.label} ({soil_risk_match.risk_label}). "
            f"{soil_risk_match.basis} Treat this as a wind-blown salt/dust screening flag; confirm with "
            "site soil resistivity/pH/chlorides/sulfates and dust deposition evidence before changing specs."
        )

    if show_coating_map:
        coating_cols = st.columns(4)
        if selected_material.kind == "steel_allowance":
            coating_cols[0].metric("Steel allowance", f"{steel_project_b2:.0f} um")
            coating_cols[1].metric("Steel Rcorr", f"{steel_rcorr:.1f} um/y")
            coating_cols[2].metric("Design life", f"{project_life_years} yr")
            coating_cols[3].metric("Basis", "B2 upper")
        else:
            coating_cols[0].metric("Min. standard", selected_galv_label)
            coating_cols[1].metric(f"{selected_material.short_label} required", f"{selected_coating_um:.1f} um/side")
            if selected_standard_um is None:
                top_um = STANDARD_GALVANIZED_DESIGNATIONS[-1].thickness_um_per_side
                coating_cols[2].metric("Standard thickness", f">{top_um:.0f} um/side")
            else:
                coating_cols[2].metric("Standard thickness", f"{selected_standard_um:.0f} um/side")
            coating_cols[3].metric("Material factor", f"{relative_factor:.2f}x")

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        if show_coating_map:
            st.subheader("Required coating map")
            st.plotly_chart(
                make_coating_requirement_figure(
                    analysis_df,
                    selected_lat=lat,
                    selected_lon=lon,
                    loss_col=d_b2_col,
                    steel_loss_col=steel_b2_col,
                    project_life_years=project_life_years,
                    material=selected_material,
                    relative_factor=relative_factor,
                    region=region,
                    soil_risk_regions=visible_soil_risk_regions,
                ),
                use_container_width=True,
            )
        elif show_product_map:
            st.subheader("Product recommendation map")
            st.plotly_chart(
                make_product_recommendation_figure(
                    analysis_df,
                    selected_lat=lat,
                    selected_lon=lon,
                    loss_col=d_b2_col,
                    project_life_years=project_life_years,
                    region=region,
                    soil_risk_regions=visible_soil_risk_regions,
                ),
                use_container_width=True,
            )
        else:
            st.subheader("Zinc Rcorr contour map")
            st.plotly_chart(
                make_contour_figure(
                    analysis_df,
                    selected_lat=lat,
                    selected_lon=lon,
                    region=region,
                    soil_risk_regions=visible_soil_risk_regions,
                ),
                use_container_width=True,
            )

    with right:
        st.subheader("Project-life overlay")
        fig = make_life_chart(
            result.rcorr_um_y,
            project_life_years=int(project_life),
            title_prefix=f"{region.label}: lat {lat:.4f}, lon {lon:.4f}",
            after_20=after_20,
        )
        st.pyplot(fig, clear_figure=True)

    with st.expander("Point calculation details", expanded=False):
        st.write(
            {
                "region": region.label,
                "project_preset": selected_project.label if selected_project else None,
                "project_location_basis": selected_project.location_basis if selected_project else None,
                "soil_salt_risk": soil_risk_match.risk_label if soil_risk_match else None,
                "soil_salt_risk_region": soil_risk_match.label if soil_risk_match else None,
                "soil_salt_risk_basis": soil_risk_match.basis if soil_risk_match else None,
                "latitude": result.latitude,
                "longitude": result.longitude,
                "temperature_C": result.T_C,
                "relative_humidity_pct": result.RH_pct,
                "Pd_SO2_mg_m2_day": result.Pd_mg_m2_d,
                "sulfate_floor_mg_m2_day": sulfate_floor,
                "Sd_chloride_mg_m2_day": result.Sd_mg_m2_d,
                "Rcorr_um_per_year": result.rcorr_um_y,
                "category": result.category,
                d_b1_col: d_project_b1,
                d_b2_col: d_project_b2,
                "Steel_Rcorr_um_per_year": steel_rcorr,
                steel_b2_col: steel_project_b2,
                "selected_material": selected_material.label,
                "material_relative_factor": relative_factor,
                "required_coating_um": selected_coating_um if selected_material.kind == "zinc_coating" else None,
                "minimum_standard_galv_designation": selected_galv_label
                if selected_material.kind == "zinc_coating"
                else None,
                "minimum_standard_thickness_um_per_side": selected_standard_um
                if selected_material.kind == "zinc_coating"
                else None,
                "product_recommendation": selected_product,
                "after_20_method": after_20,
            }
        )
        st.caption(f"ISO 9223 calibration intervals: {ISO9223_INTERVALS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data", default=None)
    parser.add_argument("--region", default="conus", choices=list(REGIONS))
    args, _ = parser.parse_known_args()
    main(args.data, args.region)
