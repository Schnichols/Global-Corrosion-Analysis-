import numpy as np
import pandas as pd

from coating import (
    G90_ZINC_UM_PER_SIDE,
    MATERIALS,
    STANDARD_GALVANIZED_DESIGNATIONS,
    galvanized_designation_label,
    g90_equivalents,
    required_coating_um,
)
from plotting import (
    RCORR_CATEGORY_BANDS,
    make_coating_requirement_figure,
    make_contour_figure,
    make_product_recommendation_figure,
    product_recommendation_code,
    product_recommendation_label,
    rcorr_category_code,
)
from regions import REGIONS
from soil_risk import soil_risk_regions_for_region


def test_product_recommendation_thresholds():
    values = np.array([19.999, 20.0, 45.0, 45.001])

    assert product_recommendation_code(values).tolist() == [0.0, 1.0, 1.0, 2.0]
    assert product_recommendation_label(19.999) == "Low corrosion product"
    assert product_recommendation_label(20.0) == "Medium corrosion product"
    assert product_recommendation_label(45.001) == "High corrosion product"


def test_product_recommendation_figure_uses_region_title():
    df = pd.DataFrame(
        {
            "lat": [35.0, 35.0, 36.0, 36.0],
            "lon": [10.0, 11.0, 10.0, 11.0],
            "D45_B2_um": [19.0, 46.0, 21.0, 42.0],
        }
    )

    fig = make_product_recommendation_figure(
        df,
        loss_col="D45_B2_um",
        project_life_years=45,
        region=REGIONS["europe"],
    )

    assert "Europe" in fig.layout.title.text
    assert "45-Year" in fig.layout.title.text
    assert "D45 B2" in fig.data[0].hovertemplate


def test_contour_figure_uses_iso_category_rcorr_colors():
    df = pd.DataFrame(
        {
            "lat": [20.0, 20.0, 21.0, 21.0],
            "lon": [40.0, 41.0, 40.0, 41.0],
            "Rcorr_um_y": [0.05, 0.4, 1.2, 3.0],
        }
    )

    fig = make_contour_figure(df, region=REGIONS["middle_east"])
    heatmap = fig.data[0]

    assert heatmap.type == "heatmap"
    assert heatmap.zmin == 0.0
    assert heatmap.zmax == len(RCORR_CATEGORY_BANDS)
    assert "C1" in heatmap.colorbar.ticktext[0]
    assert "CX" in heatmap.colorbar.ticktext[-1]
    assert len(heatmap.colorscale) == len(RCORR_CATEGORY_BANDS) * 2
    assert dict((label, color) for label, _, _, color in RCORR_CATEGORY_BANDS)["C2"] == "#86efac"
    assert dict((label, color) for label, _, _, color in RCORR_CATEGORY_BANDS)["C3"] == "#fde047"
    assert fig.data[1].type == "contour"
    assert fig.data[1].showscale is False


def test_contour_figure_can_show_soil_risk_overlay():
    df = pd.DataFrame(
        {
            "lat": [20.0, 20.0, 21.0, 21.0],
            "lon": [40.0, 41.0, 40.0, 41.0],
            "Rcorr_um_y": [0.05, 0.4, 1.2, 3.0],
        }
    )

    fig = make_contour_figure(
        df,
        region=REGIONS["middle_east"],
        soil_risk_regions=soil_risk_regions_for_region("middle_east"),
    )

    overlay_names = [trace.name for trace in fig.data if str(trace.name).startswith("Soil/salt risk")]
    assert "Soil/salt risk - High" in overlay_names
    assert "Soil/salt risk - Medium" in overlay_names


def test_rcorr_category_code_thresholds():
    values = np.array([0.1, 0.1001, 0.7, 0.7001, 2.1, 4.2, 8.4, 30.0])

    assert rcorr_category_code(values).tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0]


def test_g90_equivalent_conversion():
    assert round(G90_ZINC_UM_PER_SIDE) == 19
    assert float(g90_equivalents(G90_ZINC_UM_PER_SIDE)) == 1.0
    assert float(required_coating_um(40.0, 2.0)) == 20.0


def test_standard_galvanized_designation_table():
    by_name = {item.designation: item for item in STANDARD_GALVANIZED_DESIGNATIONS}

    assert by_name["G90"].display_um == 19
    assert by_name["G115"].display_um == 25
    assert by_name["G140"].display_um == 30
    assert by_name["G165"].display_um == 35
    assert by_name["G185"].display_um == 40
    assert by_name["G235"].display_um == 50
    assert galvanized_designation_label(10.0) == "G90"
    assert galvanized_designation_label(20.0) == "G115"


def test_coating_requirement_figure_uses_material_map():
    df = pd.DataFrame(
        {
            "lat": [20.0, 20.0, 21.0, 21.0],
            "lon": [40.0, 41.0, 40.0, 41.0],
            "D30_B2_um": [10.0, 20.0, 30.0, 40.0],
            "Steel_D30_B2_um": [100.0, 200.0, 300.0, 400.0],
        }
    )

    fig = make_coating_requirement_figure(
        df,
        material=MATERIALS["zinc_hdg"],
        loss_col="D30_B2_um",
        steel_loss_col="Steel_D30_B2_um",
        project_life_years=30,
        region=REGIONS["middle_east"],
    )

    assert "Minimum Galvanized Coating" in fig.layout.title.text
    assert fig.data[0].type == "heatmap"
    assert "galv" in fig.data[0].colorbar.title.text
    assert "G90" in fig.data[0].colorbar.ticktext[0]
    assert "G140" in fig.data[0].colorbar.ticktext[2]
    assert "G165" in fig.data[0].colorbar.ticktext[3]
