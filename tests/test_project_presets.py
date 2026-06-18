from project_presets import project_preset_by_label, project_presets_for_region
from regions import REGIONS


def test_middle_east_project_presets_are_inside_region():
    presets = project_presets_for_region("middle_east")

    assert presets
    for preset in presets:
        assert REGIONS[preset.region_id].contains(preset.latitude, preset.longitude)


def test_project_preset_lookup_by_label():
    preset = project_preset_by_label("middle_east", "SPPC 7 portfolio - conservative proxy")

    assert preset.latitude == 30.5
    assert preset.longitude == 38.2167


def test_india_has_no_presets_yet():
    assert project_presets_for_region("india") == []
