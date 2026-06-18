from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectPreset:
    id: str
    label: str
    region_id: str
    latitude: float
    longitude: float
    location_basis: str
    notes: str = ""


PROJECT_PRESETS: tuple[ProjectPreset, ...] = (
    ProjectPreset(
        id="pif_7_8_portfolio",
        label="PIF 7 and 8 portfolio - conservative proxy",
        region_id="middle_east",
        latitude=26.5524,
        longitude=37.9683,
        location_basis="Humaij / northwest KSA public proxy; highest sampled PIF proxy in this screening set.",
    ),
    ProjectPreset(
        id="pif_7_8_bisha",
        label="PIF 7 and 8 - Bisha proxy",
        region_id="middle_east",
        latitude=19.9863,
        longitude=42.3938,
        location_basis="Bisha, Asir Province public location proxy.",
    ),
    ProjectPreset(
        id="pif_7_8_humaij",
        label="PIF 7 and 8 - Humaij proxy",
        region_id="middle_east",
        latitude=26.5524,
        longitude=37.9683,
        location_basis="Humaij / Al Ula-area public proxy.",
    ),
    ProjectPreset(
        id="pif_7_8_khulis",
        label="PIF 7 and 8 - Khulis",
        region_id="middle_east",
        latitude=22.1441,
        longitude=39.3643,
        location_basis="Khulis public project coordinate proxy.",
    ),
    ProjectPreset(
        id="pif_7_8_afif",
        label="PIF 7 and 8 - Afif",
        region_id="middle_east",
        latitude=23.8826,
        longitude=42.9206,
        location_basis="Afif public project coordinate proxy.",
    ),
    ProjectPreset(
        id="sppc_7_portfolio",
        label="SPPC 7 portfolio - conservative proxy",
        region_id="middle_east",
        latitude=30.5,
        longitude=38.2167,
        location_basis="Tabarjal 2 public city/project proxy; highest sampled SPPC 7 proxy in this screening set.",
    ),
    ProjectPreset(
        id="sppc_7_tabarjal_2",
        label="SPPC 7 - Tabarjal 2 proxy",
        region_id="middle_east",
        latitude=30.5,
        longitude=38.2167,
        location_basis="Tabarjal, Al Jouf public location proxy.",
    ),
    ProjectPreset(
        id="sppc_7_mawqaq",
        label="SPPC 7 - Mawqaq proxy",
        region_id="middle_east",
        latitude=27.3786,
        longitude=41.195,
        location_basis="Mawqaq, Hail public location proxy.",
    ),
    ProjectPreset(
        id="sppc_7_tathleeth",
        label="SPPC 7 - Tathleeth",
        region_id="middle_east",
        latitude=19.5226,
        longitude=43.5551,
        location_basis="Tathleeth public project coordinate proxy.",
    ),
    ProjectPreset(
        id="sppc_7_south_al_ula",
        label="SPPC 7 - South Al Ula",
        region_id="middle_east",
        latitude=26.5524,
        longitude=37.9683,
        location_basis="South Al Ula public project coordinate proxy.",
    ),
    ProjectPreset(
        id="shaqaya_al_shagaya",
        label="Shaqaya / Al Shagaya, Kuwait",
        region_id="middle_east",
        latitude=29.2025,
        longitude=47.0708,
        location_basis="Al Shagaya renewable energy area, west of Kuwait City.",
    ),
    ProjectPreset(
        id="neom_shigry",
        label="NEOM Shigry / Shiqri proxy",
        region_id="middle_east",
        latitude=27.9865,
        longitude=35.9969,
        location_basis="Shigry / Shiqri public NEOM proxy in Tabuk, Saudi Arabia.",
        notes="Some source tables list UAE, but public NEOM/Shigry references point to Saudi Arabia.",
    ),
    ProjectPreset(
        id="mod_king_khalid_air_base",
        label="MOD King Khalid Air Base, Khamis Mushait",
        region_id="middle_east",
        latitude=18.2973,
        longitude=42.8035,
        location_basis="King Khalid Air Base public airport coordinates.",
    ),
)


def project_presets_for_region(region_id: str) -> list[ProjectPreset]:
    return [preset for preset in PROJECT_PRESETS if preset.region_id == region_id]


def project_preset_by_label(region_id: str, label: str) -> ProjectPreset:
    for preset in project_presets_for_region(region_id):
        if preset.label == label:
            return preset
    raise ValueError(f"Unknown project preset {label!r} for region {region_id!r}.")
