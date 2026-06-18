#!/usr/bin/env python3
"""Build a zinc corrosion surface from historical data.

The bundled downloader combines NASA POWER T2M/RH2M with EPA/NADP TDep
deposition grids. The TDep grids are CONUS-focused, so non-CONUS regions should
normally use scripts/build_from_pregridded_csv.py with vetted SO2 and chloride
deposition inputs already converted to ISO 9223 units.
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import xarray as xr
from pyproj import Transformer
from rasterio.io import MemoryFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corrosion_model import (  # noqa: E402
    ZINC_B1,
    ZINC_B2,
    iso_9223_zinc_rcorr,
    iso_9224_zinc_loss_um,
    kg_cl_ha_yr_to_mg_cl_m2_d,
    kg_s_ha_yr_to_mg_so2_m2_d,
    zinc_corrosivity_category,
)
from regions import REGIONS, get_region, production_surface_path  # noqa: E402

TDEP_BASE = "https://gaftp.epa.gov/castnet/tdep/CURRENT_grids"
POWER_MONTHLY_REGIONAL = "https://power.larc.nasa.gov/api/temporal/monthly/regional"
POWER_MAX_REGIONAL_DEGREES = 10.0


def make_target_grid(lon_min=-125.0, lon_max=-66.0, lat_min=24.0, lat_max=50.0, resolution=0.25) -> pd.DataFrame:
    lons = np.round(np.arange(lon_min, lon_max + resolution * 0.5, resolution), 6)
    lats = np.round(np.arange(lat_min, lat_max + resolution * 0.5, resolution), 6)
    lon2, lat2 = np.meshgrid(lons, lats)
    return pd.DataFrame({"lat": lat2.ravel(), "lon": lon2.ravel()})


def download_url(url: str, cache_dir: Path, overwrite: bool = False) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / Path(url).name
    if out.exists() and not overwrite:
        return out
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return out


def first_tif_from_zip(zip_path: Path) -> bytes:
    with zipfile.ZipFile(zip_path) as zf:
        tif_names = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))]
        if not tif_names:
            raise ValueError(f"No GeoTIFF found inside {zip_path}")
        return zf.read(tif_names[0])


def aggregate_annual_values(values: Iterable[np.ndarray], method: str) -> np.ndarray:
    """Aggregate annual samples per point while preserving all-year no-data cells."""
    stack = np.vstack(list(values))
    valid = np.any(~np.isnan(stack), axis=0)
    out = np.full(stack.shape[1], np.nan, dtype=float)
    if method == "mean":
        out[valid] = np.nanmean(stack[:, valid], axis=0)
    elif method == "max":
        out[valid] = np.nanmax(stack[:, valid], axis=0)
    else:
        raise ValueError("Aggregation method must be 'mean' or 'max'.")
    return out


def sample_tdep_variable(
    variable: str,
    years: list[int],
    points: pd.DataFrame,
    cache_dir: Path,
    *,
    aggregation: str = "max",
) -> np.ndarray:
    """Download annual TDep grids and return aggregated annual samples at points."""
    all_year_values = []
    xs = points["lon"].to_numpy(float)
    ys = points["lat"].to_numpy(float)

    for year in years:
        url = f"{TDEP_BASE}/{variable}-{year}.zip"
        zpath = download_url(url, cache_dir)
        tif_bytes = first_tif_from_zip(zpath)
        with MemoryFile(tif_bytes) as memfile:
            with memfile.open() as ds:
                transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
                xp, yp = transformer.transform(xs, ys)
                samples = np.array([v[0] for v in ds.sample(zip(xp, yp))], dtype=float)
                nodata = ds.nodata
                if nodata is not None:
                    samples = np.where(samples == nodata, np.nan, samples)
                samples = np.where(samples < -1e20, np.nan, samples)
                all_year_values.append(samples)

    return aggregate_annual_values(all_year_values, aggregation)


def _write_power_netcdf_response(content: bytes, out: Path) -> None:
    """Save a NASA POWER NetCDF response, unwrapping a zip response if needed."""
    if content.lstrip()[:1] in {b"{", b"["}:
        text = content.decode("utf-8", errors="replace")[:1000]
        raise ValueError(f"NASA POWER returned JSON instead of NetCDF: {text}")

    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            nc_names = [n for n in zf.namelist() if n.lower().endswith(".nc")]
            if not nc_names:
                raise ValueError("NASA POWER response zip did not contain a NetCDF file.")
            out.write_bytes(zf.read(nc_names[0]))
    else:
        out.write_bytes(content)


def _coord_token(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _axis_tiles(min_value: float, max_value: float, max_span: float = POWER_MAX_REGIONAL_DEGREES) -> list[tuple[float, float]]:
    """Split a coordinate axis into API-compliant intervals."""
    if max_value <= min_value:
        raise ValueError(f"Axis maximum must be greater than minimum; got {min_value} to {max_value}")
    if max_span <= 0:
        raise ValueError("max_span must be positive")

    tiles: list[tuple[float, float]] = []
    start = float(min_value)
    stop = float(max_value)
    while start < stop - 1e-9:
        end = min(start + max_span, stop)
        tiles.append((round(start, 6), round(end, 6)))
        start = end
    return tiles


def _find_power_variable(ds: xr.Dataset, parameter: str) -> str:
    if parameter in ds.data_vars:
        return parameter
    lower_map = {name.lower(): name for name in ds.data_vars}
    if parameter.lower() in lower_map:
        return lower_map[parameter.lower()]
    raise ValueError(f"Expected {parameter} in NASA POWER data; found {list(ds.data_vars)}")


def _download_power_monthly_regional_tile(
    parameter: str,
    start_year: int,
    end_year: int,
    cache_dir: Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    overwrite: bool = False,
) -> Path:
    """Download one API-compliant NASA POWER monthly regional tile as NetCDF."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / (
        f"nasa_power_{parameter}_{start_year}_{end_year}_"
        f"lon{_coord_token(lon_min)}_to_{_coord_token(lon_max)}_"
        f"lat{_coord_token(lat_min)}_to_{_coord_token(lat_max)}.nc"
    )
    if out.exists() and not overwrite:
        return out

    params = {
        "parameters": parameter,
        "community": "AG",
        "longitude-min": lon_min,
        "longitude-max": lon_max,
        "latitude-min": lat_min,
        "latitude-max": lat_max,
        "start": start_year,
        "end": end_year,
        "format": "NETCDF",
        "time-standard": "UTC",
    }
    print(
        "Downloading NASA POWER monthly regional NetCDF "
        f"for {parameter}: lon {lon_min} to {lon_max}, lat {lat_min} to {lat_max}"
    )
    r = requests.get(POWER_MONTHLY_REGIONAL, params=params, timeout=300)
    if r.status_code >= 400:
        detail = r.text[:1000].replace("\n", " ")
        raise requests.HTTPError(
            f"NASA POWER request failed for {parameter} tile "
            f"lon {lon_min} to {lon_max}, lat {lat_min} to {lat_max} "
            f"with HTTP {r.status_code}: {detail}",
            response=r,
        )
    _write_power_netcdf_response(r.content, out)
    return out


def _standardize_power_dims(ds: xr.Dataset) -> xr.Dataset:
    renames = {}
    for dim in ds.dims:
        dlow = dim.lower()
        if dlow in {"lat", "latitude"} and dim != "lat":
            renames[dim] = "lat"
        elif dlow in {"lon", "longitude"} and dim != "lon":
            renames[dim] = "lon"
        elif dlow in {"time", "date"} and dim != "time":
            renames[dim] = "time"
    if renames:
        ds = ds.rename(renames)
    return ds


def _merge_power_tile_files(parameter: str, tile_paths: list[Path], out: Path) -> Path:
    """Merge POWER regional tiles into one parameter NetCDF."""
    if not tile_paths:
        raise ValueError(f"No NASA POWER tiles were downloaded for {parameter}")

    frames = []
    for tile_path in tile_paths:
        ds = xr.open_dataset(tile_path)
        try:
            ds = _standardize_power_dims(ds)
            var_name = _find_power_variable(ds, parameter)
            da = ds[var_name].rename(parameter).load()
        finally:
            ds.close()

        df = da.to_dataframe(name=parameter).reset_index()
        required = {"time", "lat", "lon", parameter}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"NASA POWER tile {tile_path} is missing columns {sorted(missing)}")
        df = df[["time", "lat", "lon", parameter]]
        df["lat"] = df["lat"].astype(float).round(6)
        df["lon"] = df["lon"].astype(float).round(6)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["time", "lat", "lon"], keep="first")
    combined = combined.sort_values(["time", "lat", "lon"])
    merged = combined.set_index(["time", "lat", "lon"])[parameter].to_xarray().to_dataset(name=parameter)
    merged.to_netcdf(out)
    return out


def _download_power_monthly_regional_parameter(
    parameter: str,
    start_year: int,
    end_year: int,
    cache_dir: Path,
    lon_min=-125.0,
    lon_max=-66.0,
    lat_min=24.0,
    lat_max=50.0,
    overwrite: bool = False,
) -> Path:
    """Download one NASA POWER monthly regional parameter as tiled NetCDF."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"nasa_power_{parameter}_{start_year}_{end_year}.nc"
    if out.exists() and not overwrite:
        return out

    lon_tiles = _axis_tiles(lon_min, lon_max)
    lat_tiles = _axis_tiles(lat_min, lat_max)
    print(
        f"NASA POWER regional request for {parameter} will use "
        f"{len(lon_tiles) * len(lat_tiles)} tiles "
        f"({len(lon_tiles)} longitude x {len(lat_tiles)} latitude)."
    )

    tile_paths: list[Path] = []
    for tile_lat_min, tile_lat_max in lat_tiles:
        for tile_lon_min, tile_lon_max in lon_tiles:
            tile_paths.append(
                _download_power_monthly_regional_tile(
                    parameter,
                    start_year,
                    end_year,
                    cache_dir,
                    lon_min=tile_lon_min,
                    lon_max=tile_lon_max,
                    lat_min=tile_lat_min,
                    lat_max=tile_lat_max,
                    overwrite=overwrite,
                )
            )

    return _merge_power_tile_files(parameter, tile_paths, out)


def download_power_monthly_regional(
    start_year: int,
    end_year: int,
    cache_dir: Path,
    lon_min=-125.0,
    lon_max=-66.0,
    lat_min=24.0,
    lat_max=50.0,
    overwrite: bool = False,
) -> Path:
    """Download NASA POWER monthly regional T2M/RH2M NetCDF."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"nasa_power_T2M_RH2M_{start_year}_{end_year}.nc"
    if out.exists() and not overwrite:
        return out

    parameter_paths = [
        _download_power_monthly_regional_parameter(
            parameter,
            start_year,
            end_year,
            cache_dir,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            overwrite=overwrite,
        )
        for parameter in ("T2M", "RH2M")
    ]

    loaded = []
    for nc_path in parameter_paths:
        ds = xr.open_dataset(nc_path)
        try:
            loaded.append(_standardize_power_dims(ds).load())
        finally:
            ds.close()

    merged = xr.merge(loaded, compat="override")
    missing = [name for name in ("T2M", "RH2M") if name not in merged.data_vars]
    if missing:
        raise ValueError(
            f"Expected T2M and RH2M in merged NASA POWER data; missing {missing}. "
            f"Available variables: {list(merged.data_vars)}"
        )
    merged[["T2M", "RH2M"]].to_netcdf(out)
    return out


def sample_power_to_points(nc_path: Path, points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ds = xr.open_dataset(nc_path)
    try:
        ds = _standardize_power_dims(ds)
        if "T2M" not in ds or "RH2M" not in ds:
            raise ValueError(f"Expected T2M and RH2M in {nc_path}; found {list(ds.data_vars)}")
        mean = ds[["T2M", "RH2M"]].mean(dim="time", skipna=True)
        lats = xr.DataArray(points["lat"].to_numpy(float), dims="point")
        lons = xr.DataArray(points["lon"].to_numpy(float), dims="point")
        sampled = mean.interp(lat=lats, lon=lons, method="linear")
        return sampled["T2M"].to_numpy(), sampled["RH2M"].to_numpy()
    finally:
        ds.close()


def _apply_region_defaults(args: argparse.Namespace) -> argparse.Namespace:
    region = get_region(args.region)
    if args.lon_min is None:
        args.lon_min = region.lon_min
    if args.lon_max is None:
        args.lon_max = region.lon_max
    if args.lat_min is None:
        args.lat_min = region.lat_min
    if args.lat_max is None:
        args.lat_max = region.lat_max
    if args.output is None:
        args.output = str(ROOT / production_surface_path(region.id))
    return args


def build_surface(args: argparse.Namespace) -> pd.DataFrame:
    args = _apply_region_defaults(args)
    if args.region != "conus" and not args.allow_non_conus_tdep:
        raise ValueError(
            "The historical downloader uses EPA/NADP TDep deposition grids, which are CONUS-focused. "
            "For Middle East, India, Europe, Australia, or South America, build a vetted pre-gridded CSV with "
            "scripts/build_from_pregridded_csv.py. Pass --allow-non-conus-tdep only for an intentional "
            "experimental run."
        )

    years = list(range(args.start_year, args.end_year + 1))
    if len(years) < 10:
        raise ValueError("Use at least 10 years of data, e.g. --start-year 2013 --end-year 2022.")

    points = make_target_grid(
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        resolution=args.resolution,
    )
    cache_dir = Path(args.cache_dir)

    nc_path = Path(args.power_netcdf) if args.power_netcdf else download_power_monthly_regional(
        args.start_year,
        args.end_year,
        cache_dir / "nasa_power",
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
    )
    T_C, RH_pct = sample_power_to_points(nc_path, points)

    so2_kg_s_ha_yr = sample_tdep_variable(args.sulfur_variable, years, points, cache_dir / "tdep", aggregation="max")
    cl_kg_cl_ha_yr = sample_tdep_variable(args.chloride_variable, years, points, cache_dir / "tdep", aggregation="max")

    Pd = kg_s_ha_yr_to_mg_so2_m2_d(so2_kg_s_ha_yr)
    Sd = kg_cl_ha_yr_to_mg_cl_m2_d(cl_kg_cl_ha_yr)
    Rcorr = iso_9223_zinc_rcorr(T_C, RH_pct, Pd, Sd, clip_to_iso_intervals=args.clip_to_iso_intervals)

    out = points.copy()
    out["region"] = args.region
    out["region_label"] = get_region(args.region).label
    out["T_C"] = T_C
    out["RH_pct"] = RH_pct
    out["Pd_mg_m2_d"] = Pd
    out["Sd_mg_m2_d"] = Sd
    out["Rcorr_um_y"] = Rcorr
    out["category"] = zinc_corrosivity_category(Rcorr)
    out["D30_B1_um"] = iso_9224_zinc_loss_um(Rcorr, 30, b=ZINC_B1, after_20="linear")
    out["D30_B2_um"] = iso_9224_zinc_loss_um(Rcorr, 30, b=ZINC_B2, after_20="linear")
    out["source_years"] = f"{args.start_year}-{args.end_year}"
    out["temperature_aggregation"] = "mean"
    out["relative_humidity_aggregation"] = "mean"
    out["sulfur_variable"] = args.sulfur_variable
    out["sulfur_aggregation"] = "annual_max"
    out["chloride_variable"] = args.chloride_variable
    out["chloride_aggregation"] = "annual_max"
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Rcorr_um_y"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=list(REGIONS), default="conus")
    parser.add_argument("--start-year", type=int, default=2013)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--resolution", type=float, default=0.25, help="Target WGS84 grid spacing in degrees.")
    parser.add_argument("--lon-min", type=float, default=None)
    parser.add_argument("--lon-max", type=float, default=None)
    parser.add_argument("--lat-min", type=float, default=None)
    parser.add_argument("--lat-max", type=float, default=None)
    parser.add_argument("--sulfur-variable", default="so2_dw", help="TDep variable name, default so2_dw.")
    parser.add_argument("--chloride-variable", default="cl_tw", help="TDep variable name, default cl_tw; use cl_dw for dry-only.")
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--power-netcdf", default=None, help="Optional local NASA POWER T2M/RH2M NetCDF.")
    parser.add_argument("--clip-to-iso-intervals", action="store_true")
    parser.add_argument(
        "--allow-non-conus-tdep",
        action="store_true",
        help="Allow experimental TDep runs outside CONUS bounds. Not recommended for production.",
    )
    parser.add_argument("--output", default=None)
    return _apply_region_defaults(parser.parse_args())


def main():
    args = parse_args()
    df = build_surface(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} grid cells to {out}")


if __name__ == "__main__":
    main()
