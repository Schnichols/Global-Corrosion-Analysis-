#!/usr/bin/env python3
"""Build a zinc corrosion surface from an already aligned CSV.

Input columns required:
  lat, lon, T_C, RH_pct, Pd_mg_m2_d, Sd_mg_m2_d

This is the recommended production path for Middle East, India, Europe,
Australia, and South America until a vetted global SO2/chloride deposition
downloader is added. The input CSV should already contain deposition values in
ISO 9223 units.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corrosion_model import (  # noqa: E402
    ZINC_B1,
    ZINC_B2,
    iso_9223_zinc_rcorr,
    iso_9224_zinc_loss_um,
    zinc_corrosivity_category,
)
from regions import REGIONS, get_region  # noqa: E402

REQ = {"lat", "lon", "T_C", "RH_pct", "Pd_mg_m2_d", "Sd_mg_m2_d"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_csv")
    p.add_argument("output_csv")
    p.add_argument("--region", choices=list(REGIONS), default=None, help="Annotate and optionally filter rows by region.")
    p.add_argument("--filter-to-region", action="store_true", help="Drop rows outside --region bounds before writing.")
    p.add_argument("--project-life-years", type=int, default=30)
    p.add_argument("--after-20", choices=["linear", "power"], default="linear")
    p.add_argument("--clip-to-iso-intervals", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    missing = REQ - set(df.columns)
    if missing:
        raise SystemExit(f"Missing input columns: {sorted(missing)}")

    if args.region:
        region = get_region(args.region)
        if args.filter_to_region:
            df = df[
                df["lat"].between(region.lat_min, region.lat_max)
                & df["lon"].between(region.lon_min, region.lon_max)
            ].copy()
        df["region"] = region.id
        df["region_label"] = region.label

    if df.empty:
        raise SystemExit("No rows remain after applying input filters.")

    r = iso_9223_zinc_rcorr(
        df["T_C"],
        df["RH_pct"],
        df["Pd_mg_m2_d"],
        df["Sd_mg_m2_d"],
        clip_to_iso_intervals=args.clip_to_iso_intervals,
    )
    df["Rcorr_um_y"] = r
    df["category"] = zinc_corrosivity_category(r)
    d_b1_col = f"D{args.project_life_years}_B1_um"
    d_b2_col = f"D{args.project_life_years}_B2_um"
    df[d_b1_col] = iso_9224_zinc_loss_um(r, args.project_life_years, b=ZINC_B1, after_20=args.after_20)
    df[d_b2_col] = iso_9224_zinc_loss_um(r, args.project_life_years, b=ZINC_B2, after_20=args.after_20)
    df["project_life_years"] = args.project_life_years
    df["after_20_method"] = args.after_20
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
