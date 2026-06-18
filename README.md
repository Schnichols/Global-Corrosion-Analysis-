# Regional Zinc Corrosion Model - ISO 9223 / ISO 9224

This repository is a runnable Streamlit UI plus data-build helpers for zinc atmospheric-corrosion surfaces. It calculates zinc first-year corrosion rate (`Rcorr`, micrometres/year) with ISO 9223 and projects zinc loss/coating thickness required with ISO 9224.

The UI now supports these regions:

- CONUS
- Middle East
- India
- Europe
- Australia
- South America

## What The UI Does

- Shows a regional contour map of zinc `Rcorr`.
- Lets a user choose a region, use a project/portfolio preset when available, or enter latitude/longitude manually.
- Returns the local environmental inputs, zinc `Rcorr`, ISO 9223 zinc corrosivity category, and projected project-life zinc loss.
- Switches the map to required coating demand, including standard galvanized coating designations, Zn-Al-Mg material factors, and bare carbon-steel atmospheric allowance.
- Shows product recommendation bands based on project-life B2 zinc loss.
- Overlays broad Middle East and India soil/salt-flat screening regions for sabkha, salt flats, and saline/sodic soils that may create wind-blown corrosive dust.
- Produces a project-life graphic with the selected corrosion rate overlaid on ISO 9224 zinc-loss curves.

## Quick Start

```bash
cd conus_zinc_corrosion_model
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app can generate `data/sample_demo_global_zinc_surfaces.csv` on first run for UI smoke-testing across all supported regions. The generated demo is synthetic; build or import a production surface before using results for engineering decisions.

## Production Surface Paths

The app looks for region-specific production files first, then falls back to the synthetic demo:

```text
data/conus_zinc_surface_2013_2022.csv
data/middle_east_zinc_surface_2013_2022.csv
data/india_zinc_surface_2013_2022.csv
data/europe_zinc_surface_2013_2022.csv
data/australia_zinc_surface_2013_2022.csv
data/south_america_zinc_surface_2013_2022.csv
```

You can also pass an explicit file and region:

```bash
streamlit run app.py -- --region europe --data data/europe_zinc_surface_2013_2022.csv
```

## Build The CONUS Historical Surface

Default build years are 2013-2022. The CONUS historical script uses:

- NASA POWER Monthly Regional API for historical `T2M` and `RH2M`, averaged over the source years.
- EPA/NADP TDep current grids for `so2_dw` and `cl_tw` annual deposition, using the maximum annual value over the source years at each grid cell.
- A regular WGS84 output grid; default resolution is 0.25 degrees.

```bash
python scripts/build_historical_surface.py \
  --region conus \
  --start-year 2013 \
  --end-year 2022 \
  --resolution 0.25
```

For a dry-only chloride deposition sensitivity run, use:

```bash
python scripts/build_historical_surface.py --region conus --chloride-variable cl_dw --output data/conus_zinc_surface_2013_2022_cl_dw.csv
```

## Build Non-CONUS Surfaces

EPA/NADP TDep is a regional CONUS-focused deposition source. For Middle East, India, Europe, Australia, and South America, use a vetted pre-gridded CSV with the required ISO 9223 inputs already aligned to your grid:

```text
lat, lon, T_C, RH_pct, Pd_mg_m2_d, Sd_mg_m2_d
```

Then run:

```bash
python scripts/build_from_pregridded_csv.py \
  source/europe_inputs.csv \
  data/europe_zinc_surface_2013_2022.csv \
  --region europe \
  --filter-to-region
```

The pre-gridded builder recalculates `Rcorr_um_y`, category, and project-life loss columns. It also annotates rows with `region` and `region_label` so one CSV can contain multiple regional surfaces if desired.

## Model Equations And Units

The pipeline standardizes all inputs to the ISO 9223 dose-response units:

| Variable | Meaning | Unit used in this code |
|---|---:|---:|
| `T_C` | annual mean air temperature | degrees C |
| `RH_pct` | annual mean relative humidity | percent |
| `Pd_mg_m2_d` | SO2 deposition rate | mg SO2/(m2 day) |
| `Sd_mg_m2_d` | chloride deposition rate | mg Cl/(m2 day) |
| `Rcorr_um_y` | first-year zinc corrosion rate | micrometres/year |

TDep aggregation and unit conversions for CONUS:

- `so2_dw` and `cl_tw`/`cl_dw` are sampled for each source year, then aggregated as the maximum annual value per output grid cell before conversion to ISO dose-response units.
- `so2_dw` is kg-S/ha/year. The script converts it to mg SO2/(m2 day) using `kg-S/ha/year * 0.273785 * 2.0`.
- `cl_tw` or `cl_dw` is kg-Cl/ha/year. The script converts it to mg Cl/(m2 day) using `kg-Cl/ha/year * 0.273785`.

ISO 9224 zinc projection:

- Up to 20 years: `D = Rcorr * t^b`.
- For years greater than 20, the UI defaults to the ISO 9224 linear-tail maximum-estimate form. The sidebar can switch to the all-years power law.
- Zinc uses B1 = 0.813 and B2 = 0.873. B1 is the typical exponent; B2 is the upper estimate in the ISO 9224 table.

## Data-Source Notes

This code does not currently download or calculate from Copernicus/CAMS layers. The CONUS historical builder uses NASA POWER for temperature and relative humidity plus EPA/NADP TDep for SO2 and chloride deposition. Non-CONUS surfaces are expected to arrive as a vetted pre-gridded CSV containing the ISO 9223 input columns listed above.

The Middle East and India soil/salt-flat overlay is a screening aid only. It flags broad sabkha, salt flat, saline soil, sodic soil, and coastal salinity regions where wind-blown salt or saline dust could make actual site exposure more severe than the regional atmospheric chloride deposition field. It should not be used as a soil corrosivity model or as a substitute for geotechnical corrosion testing. For embedded posts, confirm soil resistivity, pH, chlorides, sulfates, moisture, drainage, groundwater, and backfill conditions.

Localized urban, industrial, sheltered marine, deicing-salt, splash-zone, and wind-blown saline-dust microenvironments can differ significantly from regional gridded inputs. The UI flags inputs outside the ISO 9223 calibration ranges instead of silently hiding them. For design-critical work, calibrate the model against zinc coupon measurements or site-specific deposition measurements.

## Files

```text
app.py                                   Streamlit UI
regions.py                              Region definitions and default paths
project_presets.py                      Project and owner-portfolio coordinate presets
soil_risk.py                            Middle East and India soil/salt-flat screening polygons
corrosion_model.py                       ISO 9223/9224 equations and unit conversions
plotting.py                              Plotly contour map and project-life chart
surface.py                               CSV loader/interpolator for lat/lon lookups
scripts/build_historical_surface.py      CONUS NASA POWER + TDep downloader/builder
scripts/build_from_pregridded_csv.py     Builder for already-aligned non-CONUS or custom inputs
scripts/build_demo_surface.py            Synthetic dense demo-surface generator
data/sample_demo_global_zinc_surfaces.csv Generated synthetic UI smoke-test surface
data/country_boundary_rings.json         Optional local country-outline cache; app falls back to Natural Earth
tests/test_corrosion_model.py            Equation regression tests
tests/test_regions.py                    Region configuration tests
```

## Run Tests

```bash
python -m pytest tests
```
