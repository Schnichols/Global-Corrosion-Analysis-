#!/usr/bin/env python3
"""Download CAMS EAC4 SO2/sea-salt fields and build corrosion screening CSVs.

The CAMS EAC4 variables used here are model-level mass mixing ratios in kg/kg,
not ISO 9223 deposition measurements. This script converts them to deposition
proxies with configurable deposition velocities so the outputs remain clearly
labeled as screening estimates.
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corrosion_model import (  # noqa: E402
    ZINC_B1,
    ZINC_B2,
    iso_9223_zinc_rcorr,
    iso_9224_zinc_loss_um,
    zinc_corrosivity_category,
)
from regions import DEFAULT_DEMO_SURFACE, REGION_ORDER, REGIONS, get_region, production_surface_path  # noqa: E402
from scripts.build_historical_surface import sample_deposition_csv_to_points  # noqa: E402

CAMS_MONTHLY_DATASET = "cams-global-reanalysis-eac4-monthly"
CAMS_3HOURLY_DATASET = "cams-global-reanalysis-eac4"
CAMS_VARIABLES = [
    "sulphur_dioxide",
    "sea_salt_aerosol_0.03-0.5um_mixing_ratio",
    "sea_salt_aerosol_0.5-5um_mixing_ratio",
    "sea_salt_aerosol_5-20um_mixing_ratio",
]
CAMS_ALIASES = {
    "so2_kg_kg": ["so2", "sulphur_dioxide"],
    "sea_salt_0p03_0p5um_kg_kg": ["aermr01", "sea_salt_aerosol_0.03-0.5um_mixing_ratio"],
    "sea_salt_0p5_5um_kg_kg": ["aermr02", "sea_salt_aerosol_0.5-5um_mixing_ratio"],
    "sea_salt_5_20um_kg_kg": ["aermr03", "sea_salt_aerosol_5-20um_mixing_ratio"],
}
TIME_DIMS = {"time", "valid_time", "date"}
LAT_NAMES = {"lat", "latitude"}
LON_NAMES = {"lon", "longitude"}
MG_PER_KG = 1_000_000.0
SECONDS_PER_DAY = 86_400.0
DEFAULT_REGIONS = tuple(REGION_ORDER)


@dataclass(frozen=True)
class DepositionAssumptions:
    air_density_kg_m3: float = 1.2
    so2_vd_m_s: float = 0.005
    sea_salt_small_vd_m_s: float = 0.001
    sea_salt_medium_vd_m_s: float = 0.005
    sea_salt_large_vd_m_s: float = 0.03
    chloride_mass_fraction: float = 0.55


def parse_month(value: str) -> pd.Period:
    try:
        return pd.Period(value, freq="M")
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM, got {value!r}") from exc


def months_between(start: pd.Period, end: pd.Period) -> list[pd.Period]:
    if end < start:
        raise ValueError("end month must be the same as or later than start month")
    return list(pd.period_range(start, end, freq="M"))


def _area_for_region(region_id: str, pad_degrees: float = 0.0) -> list[float]:
    region = get_region(region_id)
    return [
        min(90.0, region.lat_max + pad_degrees),
        max(-180.0, region.lon_min - pad_degrees),
        max(-90.0, region.lat_min - pad_degrees),
        min(180.0, region.lon_max + pad_degrees),
    ]


def build_monthly_request(
    region_id: str,
    year: int,
    months: Iterable[int],
    *,
    model_level: int = 60,
    data_format: str = "netcdf_zip",
    area_pad_degrees: float = 0.75,
) -> dict:
    month_values = [f"{int(month):02d}" for month in months]
    if not month_values:
        raise ValueError("At least one month is required")
    return {
        "variable": CAMS_VARIABLES,
        "model_level": [str(model_level)],
        "year": [str(year)],
        "month": month_values,
        "product_type": ["monthly_mean"],
        "data_format": data_format,
        "area": _area_for_region(region_id, area_pad_degrees),
    }


def build_3hourly_request(
    region_id: str,
    month: pd.Period,
    *,
    model_level: int = 60,
    data_format: str = "netcdf_zip",
    area_pad_degrees: float = 0.75,
) -> dict:
    last_day = calendar.monthrange(month.year, month.month)[1]
    return {
        "variable": CAMS_VARIABLES,
        "model_level": [str(model_level)],
        "date": [f"{month.year:04d}-{month.month:02d}-01/{month.year:04d}-{month.month:02d}-{last_day:02d}"],
        "time": ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"],
        "data_format": data_format,
        "area": _area_for_region(region_id, area_pad_degrees),
    }


def build_download_jobs(args: argparse.Namespace) -> list[tuple[str, dict, Path]]:
    start = parse_month(args.start_month)
    end = parse_month(args.end_month)
    months = months_between(start, end)
    jobs: list[tuple[str, dict, Path]] = []
    for region_id in args.regions:
        region_dir = Path(args.raw_dir) / region_id
        if args.temporal == "monthly":
            dataset = CAMS_MONTHLY_DATASET
            by_year: dict[int, list[int]] = {}
            for month in months:
                by_year.setdefault(month.year, []).append(month.month)
            for year, month_numbers in by_year.items():
                request = build_monthly_request(
                    region_id,
                    year,
                    month_numbers,
                    model_level=args.model_level,
                    data_format=args.data_format,
                    area_pad_degrees=args.area_pad_degrees,
                )
                target = region_dir / f"cams_eac4_monthly_ml{args.model_level}_{region_id}_{year}.{_download_suffix(args.data_format)}"
                jobs.append((dataset, request, target))
        else:
            dataset = CAMS_3HOURLY_DATASET
            for month in months:
                request = build_3hourly_request(
                    region_id,
                    month,
                    model_level=args.model_level,
                    data_format=args.data_format,
                    area_pad_degrees=args.area_pad_degrees,
                )
                target = region_dir / (
                    f"cams_eac4_3hourly_ml{args.model_level}_{region_id}_{month.year:04d}_{month.month:02d}."
                    f"{_download_suffix(args.data_format)}"
                )
                jobs.append((dataset, request, target))
    return jobs


def _download_suffix(data_format: str) -> str:
    if data_format.endswith("_zip"):
        return "zip"
    if data_format == "netcdf":
        return "nc"
    return data_format


def _make_cds_client(args: argparse.Namespace):
    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError("cdsapi is not installed. Run `python -m pip install cdsapi`.") from exc

    url = args.ads_url or os.environ.get("ADS_API_URL") or os.environ.get("CDSAPI_URL")
    key = args.ads_key or os.environ.get("ADS_API_KEY") or os.environ.get("CDSAPI_KEY")
    if url and key:
        return cdsapi.Client(url=url, key=key)
    try:
        return cdsapi.Client()
    except Exception as exc:
        raise RuntimeError(
            "Missing ADS/CDS API credentials. Create a ~/.cdsapirc file using the values "
            "from https://ads.atmosphere.copernicus.eu/how-to-api, or pass --ads-url and "
            "--ads-key / set ADS_API_URL and ADS_API_KEY. You may also need to accept the "
            "CAMS EAC4 licence once in the ADS web UI before API downloads work."
        ) from exc


def download_jobs(args: argparse.Namespace) -> list[Path]:
    jobs = build_download_jobs(args)
    if args.dry_run:
        for dataset, request, target in jobs:
            print(json.dumps({"dataset": dataset, "target": str(target), "request": request}, indent=2))
        return []

    client = _make_cds_client(args)
    downloaded: list[Path] = []
    for dataset, request, target in jobs:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not args.overwrite:
            print(f"Using cached CAMS file {target}")
            downloaded.append(target)
            continue
        print(f"Retrieving {dataset} -> {target}")
        client.retrieve(dataset, request).download(str(target))
        downloaded.append(target)
    return downloaded


def _extract_netcdfs(path: Path, extract_dir: Path) -> list[Path]:
    if path.suffix.lower() == ".nc":
        return [path]
    if path.suffix.lower() != ".zip":
        raise ValueError(f"Cannot read {path}; use --data-format netcdf or netcdf_zip.")

    out_dir = extract_dir / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        names = [name for name in zf.namelist() if name.lower().endswith(".nc")]
        if not names:
            raise ValueError(f"No NetCDF file found inside {path}")
        outputs = []
        for name in names:
            out = out_dir / Path(name).name
            if not out.exists():
                out.write_bytes(zf.read(name))
            outputs.append(out)
        return outputs


def _coord_name(ds: xr.Dataset, candidates: set[str]) -> str:
    for name in list(ds.coords) + list(ds.dims):
        if name.lower() in candidates:
            return name
    raise ValueError(f"Could not find coordinate matching {sorted(candidates)} in {list(ds.coords)}")


def _data_var(ds: xr.Dataset, canonical_name: str) -> xr.DataArray:
    aliases = {alias.lower() for alias in CAMS_ALIASES[canonical_name]}
    for name, da in ds.data_vars.items():
        keys = {name.lower()}
        for attr in ("GRIB_shortName", "GRIB_cfVarName", "long_name", "standard_name"):
            value = da.attrs.get(attr)
            if value:
                keys.add(str(value).lower())
        if keys.intersection(aliases):
            return da
    raise ValueError(f"Could not find {canonical_name}; available variables are {list(ds.data_vars)}")


def _mean_field(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    mean_dims = [dim for dim in da.dims if dim.lower() in TIME_DIMS]
    if mean_dims:
        da = da.mean(dim=mean_dims, skipna=True)
    for dim in list(da.dims):
        if dim not in {lat_name, lon_name}:
            da = da.isel({dim: 0}, drop=True)
    return da


def cams_files_to_dataframe(paths: Iterable[Path], *, extract_dir: Path) -> pd.DataFrame:
    frames = []
    for path in paths:
        for nc_path in _extract_netcdfs(Path(path), extract_dir):
            ds = xr.open_dataset(nc_path)
            try:
                lat_name = _coord_name(ds, LAT_NAMES)
                lon_name = _coord_name(ds, LON_NAMES)
                fields = {}
                for canonical_name in CAMS_ALIASES:
                    fields[canonical_name] = _mean_field(_data_var(ds, canonical_name), lat_name, lon_name)
                out = xr.Dataset(fields)
                df = out.to_dataframe().reset_index()
                df = df.rename(columns={lat_name: "lat", lon_name: "lon"})
                df["lon"] = np.where(df["lon"] > 180.0, df["lon"] - 360.0, df["lon"])
                frames.append(df[["lat", "lon", *CAMS_ALIASES.keys()]])
            finally:
                ds.close()

    if not frames:
        raise ValueError("No CAMS NetCDF files were available for conversion.")
    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby(["lat", "lon"], as_index=False)[list(CAMS_ALIASES)].mean()


def mixing_ratio_to_deposition_mg_m2_d(
    mixing_ratio_kg_kg,
    *,
    air_density_kg_m3: float,
    deposition_velocity_m_s: float,
    mass_fraction: float = 1.0,
):
    return np.asarray(mixing_ratio_kg_kg, dtype=float) * air_density_kg_m3 * deposition_velocity_m_s * SECONDS_PER_DAY * MG_PER_KG * mass_fraction


def add_deposition_proxy_columns(df: pd.DataFrame, assumptions: DepositionAssumptions) -> pd.DataFrame:
    out = df.copy()
    out["sea_salt_total_kg_kg"] = (
        out["sea_salt_0p03_0p5um_kg_kg"]
        + out["sea_salt_0p5_5um_kg_kg"]
        + out["sea_salt_5_20um_kg_kg"]
    )
    out["Pd_mg_m2_d"] = mixing_ratio_to_deposition_mg_m2_d(
        out["so2_kg_kg"],
        air_density_kg_m3=assumptions.air_density_kg_m3,
        deposition_velocity_m_s=assumptions.so2_vd_m_s,
    )
    out["Sd_small_mg_m2_d"] = mixing_ratio_to_deposition_mg_m2_d(
        out["sea_salt_0p03_0p5um_kg_kg"],
        air_density_kg_m3=assumptions.air_density_kg_m3,
        deposition_velocity_m_s=assumptions.sea_salt_small_vd_m_s,
        mass_fraction=assumptions.chloride_mass_fraction,
    )
    out["Sd_medium_mg_m2_d"] = mixing_ratio_to_deposition_mg_m2_d(
        out["sea_salt_0p5_5um_kg_kg"],
        air_density_kg_m3=assumptions.air_density_kg_m3,
        deposition_velocity_m_s=assumptions.sea_salt_medium_vd_m_s,
        mass_fraction=assumptions.chloride_mass_fraction,
    )
    out["Sd_large_mg_m2_d"] = mixing_ratio_to_deposition_mg_m2_d(
        out["sea_salt_5_20um_kg_kg"],
        air_density_kg_m3=assumptions.air_density_kg_m3,
        deposition_velocity_m_s=assumptions.sea_salt_large_vd_m_s,
        mass_fraction=assumptions.chloride_mass_fraction,
    )
    out["Sd_mg_m2_d"] = out["Sd_small_mg_m2_d"] + out["Sd_medium_mg_m2_d"] + out["Sd_large_mg_m2_d"]
    out["air_density_kg_m3_assumed"] = assumptions.air_density_kg_m3
    out["so2_vd_m_s_assumed"] = assumptions.so2_vd_m_s
    out["sea_salt_small_vd_m_s_assumed"] = assumptions.sea_salt_small_vd_m_s
    out["sea_salt_medium_vd_m_s_assumed"] = assumptions.sea_salt_medium_vd_m_s
    out["sea_salt_large_vd_m_s_assumed"] = assumptions.sea_salt_large_vd_m_s
    out["chloride_mass_fraction_assumed"] = assumptions.chloride_mass_fraction
    out["deposition_method"] = "CAMS EAC4 model-level mixing ratio converted with assumed deposition velocity"
    return out


def _template_weather(region_id: str) -> pd.DataFrame:
    region = get_region(region_id)
    path = production_surface_path(region_id)
    if not path.exists():
        path = DEFAULT_DEMO_SURFACE
    df = pd.read_csv(path)
    if "region" in df.columns:
        df = df[df["region"].eq(region_id)].copy()
    else:
        df = df[df["lat"].between(region.lat_min, region.lat_max) & df["lon"].between(region.lon_min, region.lon_max)].copy()
    missing = {"lat", "lon", "T_C", "RH_pct"}.difference(df.columns)
    if missing:
        raise ValueError(f"Weather template {path} is missing columns {sorted(missing)}")
    if df.empty:
        raise ValueError(f"No weather-template rows found for {region_id} in {path}")
    return df[["lat", "lon", "T_C", "RH_pct"]].copy()


def reassess_region_surface(
    region_id: str,
    deposition_csv: Path,
    output_csv: Path,
    *,
    interpolation: str = "linear",
    max_nearest_distance_deg: float | None = 1.5,
    project_life_years: int = 30,
    after_20: str = "linear",
    clip_to_iso_intervals: bool = False,
) -> pd.DataFrame:
    weather = _template_weather(region_id)
    Pd, Sd, distances = sample_deposition_csv_to_points(
        deposition_csv,
        weather[["lat", "lon"]],
        interpolation=interpolation,
        max_nearest_distance_deg=max_nearest_distance_deg,
    )
    r = iso_9223_zinc_rcorr(
        weather["T_C"],
        weather["RH_pct"],
        Pd,
        Sd,
        clip_to_iso_intervals=clip_to_iso_intervals,
    )
    out = weather.copy()
    out["region"] = region_id
    out["region_label"] = get_region(region_id).label
    out["Pd_mg_m2_d"] = Pd
    out["Sd_mg_m2_d"] = Sd
    out["deposition_nearest_source_distance_deg"] = distances
    out["Rcorr_um_y"] = r
    out["category"] = zinc_corrosivity_category(r)
    out[f"D{project_life_years}_B1_um"] = iso_9224_zinc_loss_um(r, project_life_years, b=ZINC_B1, after_20=after_20)
    out[f"D{project_life_years}_B2_um"] = iso_9224_zinc_loss_um(r, project_life_years, b=ZINC_B2, after_20=after_20)
    out["project_life_years"] = project_life_years
    out["after_20_method"] = after_20
    out["source_years"] = "CAMS EAC4 proxy"
    out["deposition_source"] = str(deposition_csv)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Rcorr_um_y"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def summarize_surface(region_id: str, df: pd.DataFrame) -> dict:
    counts = df["category"].value_counts(normalize=True).to_dict()
    summary = {
        "region": region_id,
        "region_label": get_region(region_id).label,
        "rows": int(len(df)),
        "Rcorr_min_um_y": float(df["Rcorr_um_y"].min()),
        "Rcorr_p50_um_y": float(df["Rcorr_um_y"].quantile(0.50)),
        "Rcorr_p90_um_y": float(df["Rcorr_um_y"].quantile(0.90)),
        "Rcorr_max_um_y": float(df["Rcorr_um_y"].max()),
        "Pd_mean_mg_m2_d": float(df["Pd_mg_m2_d"].mean()),
        "Sd_mean_mg_m2_d": float(df["Sd_mg_m2_d"].mean()),
    }
    for category in ["C1", "C2", "C3", "C4", "C5", "CX"]:
        summary[f"share_{category}"] = float(counts.get(category, 0.0))
    return summary


def _region_files(region_id: str, args: argparse.Namespace) -> list[Path]:
    region_dir = Path(args.raw_dir) / region_id
    suffix = _download_suffix(args.data_format)
    pattern = f"cams_eac4_{args.temporal}_ml{args.model_level}_{region_id}_*.{suffix}"
    return sorted(region_dir.glob(pattern))


def convert_and_reassess(args: argparse.Namespace) -> pd.DataFrame:
    assumptions = DepositionAssumptions(
        air_density_kg_m3=args.air_density_kg_m3,
        so2_vd_m_s=args.so2_vd_m_s,
        sea_salt_small_vd_m_s=args.sea_salt_small_vd_m_s,
        sea_salt_medium_vd_m_s=args.sea_salt_medium_vd_m_s,
        sea_salt_large_vd_m_s=args.sea_salt_large_vd_m_s,
        chloride_mass_fraction=args.chloride_mass_fraction,
    )
    summaries = []
    for region_id in args.regions:
        raw_files = _region_files(region_id, args)
        if not raw_files:
            print(f"No CAMS files found for {region_id}; skipping conversion.")
            continue
        print(f"Converting {len(raw_files)} CAMS files for {region_id}")
        cams = cams_files_to_dataframe(raw_files, extract_dir=Path(args.raw_dir) / "_extracted")
        cams = add_deposition_proxy_columns(cams, assumptions)
        cams["region"] = region_id
        cams["region_label"] = get_region(region_id).label
        cams["cams_dataset"] = CAMS_MONTHLY_DATASET if args.temporal == "monthly" else CAMS_3HOURLY_DATASET
        cams["cams_model_level"] = args.model_level
        cams["cams_temporal"] = args.temporal
        cams["cams_source_months"] = f"{args.start_month}/{args.end_month}"

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cams_csv = out_dir / f"{region_id}_cams_eac4_ml{args.model_level}_{args.start_month}_{args.end_month}_deposition_proxy.csv"
        cams.to_csv(cams_csv, index=False)

        deposition_csv = out_dir / f"{region_id}_cams_eac4_ml{args.model_level}_{args.start_month}_{args.end_month}_iso9223_deposition_proxy.csv"
        cams[["lat", "lon", "Pd_mg_m2_d", "Sd_mg_m2_d"]].to_csv(deposition_csv, index=False)

        surface_csv = out_dir / f"{region_id}_zinc_surface_{args.start_month}_{args.end_month}_cams_eac4_proxy.csv"
        reassessed = reassess_region_surface(
            region_id,
            deposition_csv,
            surface_csv,
            interpolation=args.surface_interpolation,
            max_nearest_distance_deg=args.surface_max_nearest_distance_deg,
            project_life_years=args.project_life_years,
            after_20=args.after_20,
            clip_to_iso_intervals=args.clip_to_iso_intervals,
        )
        summaries.append(summarize_surface(region_id, reassessed))
        print(f"Wrote {cams_csv}")
        print(f"Wrote {deposition_csv}")
        print(f"Wrote {surface_csv}")

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        summary_out = Path(args.output_dir) / f"cams_eac4_regional_corrosivity_summary_{args.start_month}_{args.end_month}.csv"
        summary_df.to_csv(summary_out, index=False)
        print(f"Wrote {summary_out}")
    return summary_df


def _expand_regions(values: list[str]) -> list[str]:
    if values == ["all"] or "all" in values:
        return list(DEFAULT_REGIONS)
    bad = [value for value in values if value not in REGIONS]
    if bad:
        raise argparse.ArgumentTypeError(f"Unknown region(s): {bad}. Expected all or one of {list(REGIONS)}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", nargs="+", default=["all"], help="Regions to process, or all.")
    parser.add_argument("--start-month", default="2015-08", help="Inclusive start month, YYYY-MM.")
    parser.add_argument("--end-month", default="2025-08", help="Inclusive end month, YYYY-MM.")
    parser.add_argument("--temporal", choices=["monthly", "3hourly"], default="monthly")
    parser.add_argument("--model-level", type=int, default=60)
    parser.add_argument("--data-format", choices=["netcdf_zip", "netcdf"], default="netcdf_zip")
    parser.add_argument("--area-pad-degrees", type=float, default=0.75)
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw" / "cams_eac4"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "cams"))
    parser.add_argument("--ads-url", default=None, help="Optional ADS API URL; otherwise use config/env.")
    parser.add_argument("--ads-key", default=None, help="Optional ADS API key; otherwise use config/env.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print CDS API requests without downloading.")
    parser.add_argument("--skip-download", action="store_true", help="Convert/reassess from files already in --raw-dir.")
    parser.add_argument("--skip-convert", action="store_true", help="Only download raw CAMS files.")
    parser.add_argument("--air-density-kg-m3", type=float, default=1.2)
    parser.add_argument("--so2-vd-m-s", type=float, default=0.005)
    parser.add_argument("--sea-salt-small-vd-m-s", type=float, default=0.001)
    parser.add_argument("--sea-salt-medium-vd-m-s", type=float, default=0.005)
    parser.add_argument("--sea-salt-large-vd-m-s", type=float, default=0.03)
    parser.add_argument("--chloride-mass-fraction", type=float, default=0.55)
    parser.add_argument("--surface-interpolation", choices=["linear", "nearest"], default="linear")
    parser.add_argument("--surface-max-nearest-distance-deg", type=float, default=1.5)
    parser.add_argument("--project-life-years", type=int, default=30)
    parser.add_argument("--after-20", choices=["linear", "power"], default="linear")
    parser.add_argument("--clip-to-iso-intervals", action="store_true")
    args = parser.parse_args()
    args.regions = _expand_regions(args.regions)
    parse_month(args.start_month)
    parse_month(args.end_month)
    return args


def main() -> None:
    try:
        args = parse_args()
        if not args.skip_download:
            download_jobs(args)
        if not args.skip_convert and not args.dry_run:
            convert_and_reassess(args)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None


if __name__ == "__main__":
    main()
