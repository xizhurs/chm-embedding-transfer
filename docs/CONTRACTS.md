# Data contracts

All stages communicate through the files below. **Do not change `config.yaml`, `src/hgt/grid.py`, `src/hgt/io.py`
or `src/hgt/config.py`** without the owner's (main agent's) agreement. These files define the geometry that keeps
the handcrafted-vs-AEF comparison fair.

Environment: `uv run python ...` from the repo root (Python 3.12, `.venv`). To add a dependency, use `uv add <pkg>`.

## Sites and grids (`hgt.grid.get_grid(site)`)
| site | CRS | grid bounds (m) | 10 m shape (H, W) | 100 m shape | 1 km blocks |
|---|---|---|---|---|---|
| colombia | EPSG:32618 | 415000, 581000, 426000, 594000 | 1300 × 1100 | 130 × 110 | 143 |
| tanzania | EPSG:32736 | 295000, 9824000, 299000, 9836000 | 1200 × 400 | 120 × 40 | 48 |

- Years: colombia 2021, 2022; tanzania 2022, 2024 (`hgt.config.site_years()`).
- `grid.feature_transform` / `grid.cell_transform` give the Affine transforms. The 10 m, 100 m and 1 km grids share the same origin.
- `cell_id = row * n_cell_cols + col` (row 0 = north); `block_id` = id of the 1 km block (10 × 10 cells). Get both from `grid.cell_table()`.
- WGS84 search bbox: `grid.bbox_wgs84(buffer_m=...)`.

## Stage outputs

### 1a — S1/S2 from Planetary Computer (`src/hgt/download_pc.py`)
- `data/raw/s2/{site}_{year}.tif`: float32 on the 10 m grid, NaN = nodata, band descriptions =
  `s2_{B}_median` for B in config `s2.bands` (reflectance 0–1), then `s2_{IDX}_p{10,50,90}` for IDX in config `s2.indices`.
- `data/raw/s2/{site}_{year}_qa.tif`: band `s2_n_clear` (count of clear observations per pixel).
- `data/raw/s1/{site}_{year}.tif`: bands `s1_{pol}_db_{median,std,p10,p90}` for pol in `vv, vh, vhvv`
  (`vhvv` = VH − VV in dB, computed per scene). Percentiles are computed on the dB time series, and std is the std of the dB series.
- `data/raw/s1/{site}_{year}_qa.tif`: band `s1_n_obs`.
- Time window: calendar year. S1 = descending + Sentinel-1A only. S2 = per-item baseline offset, de-duplicated, SCL 4–7 kept.

### 1b — AlphaEarth from AWS (`src/hgt/download_aef.py`)
- `data/raw/aef/{site}_{year}.tif`: float32 on the 10 m grid, 64 bands `aef_A00` … `aef_A63`, de-quantised
  (`((v/127.5)**2)*sign(v)`), NaN where the source = −128.

### 1d — Tessera from AWS (`src/hgt/download_tessera.py`)
- `data/raw/tessera/{site}_{year}.tif`: float32 on the 10 m grid, bands `tes_E000` … `tes_E127`, de-quantised,
  NaN = nodata. Reprojection (if any) with nearest resampling, so embedding vectors are never blended.

### 1c — Targets (`src/hgt/targets.py`)
- `data/interim/targets/{site}_{year}.parquet`, one row per kept 100 m cell:
  `cell_id:int64, row:int32, col:int32, x:float64, y:float64, block_id:int32, height:float32, valid_frac:float32`.
  `height` = config `target.percentile` (P95) of the valid 0.5 m CHM pixels in the cell. A cell is kept if `valid_frac >= 0.5`.
  (P95 rather than P98 because the Colombia CHMs contain power lines capped at 80 m.)

All rasters must pass `hgt.io.check_on_grid(path, get_grid(site))`. Write them with `hgt.io.write_feature_raster`.

### 2 — Features table (`src/hgt/features.py`)
- `data/tables/{site}_{year}.parquet` = the target table plus:
  - `hc_{band}_mean`, `hc_{band}_std` for every s1/s2 band (10 × 10 px per cell, nan-aware)
  - `hc_glcm_{input}_{prop}` (24 columns, see config `glcm`)
  - `aef_{Axx}_mean` (mean vector re-normalised to unit length), `aef_{Axx}_std`
  - `tes_{Exxx}_mean` (plain per-dimension mean, no re-normalisation), `tes_{Exxx}_std`
- Feature-set column selectors: HC = `hc_*`, AEF = `aef_*`, TES = `tes_*`; HC+AEF and HC+TES = unions.

### 3 — Results (`src/hgt/experiments.py`, `src/hgt/report.py`), one directory per learner
All stage-3 outputs go to `hgt.io.experiment_dir()` = `results/{model.learner}/` (`xgboost` or `tabicl`):
- `metrics.csv` (long): `experiment, train, test, feature_set, repeat, fold, k, draw, variant, metric, value`.
  `fold = -1` = pooled out-of-fold; `k`/`draw = -1` = not applicable. `variant`: `main`; E3 `tz+k` / `k-only`;
  E4 Colombia `overlap` / `full` / `matched`. Metrics: r2, rmse, rrmse, mae, bias, n; extra per-fold rows:
  `n_shot_cells` (E3), `n_selected`, `n_train`, `n_buffer_removed` (E1).
- `run_log.csv`: settings of each run (n_estimators, n_repeats, n_draws, quick, feature_sets, cells, learner,
  buffer_cells, selection).
- `predictions_E1.parquet`, `predictions_E2.parquet`, `importances_E1.csv` (XGBoost only), `figures/*.png`,
  `summary.md`.
- Pipeline settings (`config.yaml` `pipeline`): `buffer_cells` (training cells within n cells of a fold's test
  blocks are dropped; `hgt.splits.buffer_mask`), `cells` (covered / all), `feature_selection` (in-fold,
  XGBoost-SHAP selector for every learner).
- `results/learner_comparison.md` (`scripts/compare_learners.py`): paired comparison of two learners.
- Archives of earlier runs: `results/archive_*/` (each with a README).
