from soil_risk import (
    highest_soil_risk_for_point,
    matching_soil_risk_regions,
    soil_risk_regions_for_region,
)


def test_middle_east_gulf_point_flags_high_risk():
    risk = highest_soil_risk_for_point("middle_east", 29.2025, 47.0708)

    assert risk is not None
    assert risk.risk_level == "high"
    assert "Gulf" in risk.label or "Kuwait" in risk.label


def test_middle_east_interior_point_is_not_flagged():
    assert highest_soil_risk_for_point("middle_east", 27.5, 41.7) is None


def test_india_kutch_point_flags_high_risk():
    risk = highest_soil_risk_for_point("india", 23.5, 70.5)

    assert risk is not None
    assert risk.risk_level == "high"
    assert "Kutch" in risk.label


def test_india_delhi_area_flags_medium_risk():
    risk = highest_soil_risk_for_point("india", 28.6139, 77.2090)

    assert risk is not None
    assert risk.risk_level == "medium"


def test_matching_regions_are_sorted_by_risk_rank():
    matches = matching_soil_risk_regions("india", 24.5, 70.5)

    assert matches
    assert matches[0].risk_level == "high"


def test_soil_risk_regions_only_configured_for_target_regions():
    assert soil_risk_regions_for_region("middle_east")
    assert soil_risk_regions_for_region("india")
    assert soil_risk_regions_for_region("conus") == []
