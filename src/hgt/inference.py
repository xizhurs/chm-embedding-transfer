"""Inference: wall-to-wall 100 m canopy-height maps for every site-year.

For each site-year and feature set, the configured learner (with the same in-fold SHAP feature
selection as the experiments, here fitted on all training cells) is trained on all lidar cells
(covered-cell rule of hgt.experiments.cell_subset) and predicts every 100 m cell of the site grid
whose features are complete (TES feature sets are NaN where Tessera has no data).

Outputs in results/{learner}/maps/:
  {site}_{year}.tif   float32 on the 100 m cell grid; bands lidar_p95 (observed, NaN outside the
                      lidar footprint), then pred_{FS} per feature set
  {site}_{year}.png   one panel per band, shared colour scale, lidar footprint outlined
  maps_meta.csv       per site-year / FS: training cells, selected features, predicted cells,
                      in-sample R2 on lidar cells (sanity check only, not an accuracy estimate)

Needs the map feature tables: uv run python -m hgt.features --maps

Usage:
  uv run python -m hgt.inference [--site S --year Y] [--fs HC AEF ...] [--learner xgboost|plsr|tabicl]
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from hgt.config import load_config, site_years
from hgt.experiments import cell_subset, load_table, make_model, xy
from hgt.grid import get_grid
from hgt.io import experiment_dir, map_table_path, write_cell_raster
from hgt.splits import FEATURE_SETS, feature_columns


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_grid(grid, df: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    img = np.full(grid.cell_shape, np.nan, dtype="float32")
    img[df["row"].to_numpy(), df["col"].to_numpy()] = values
    return img


def map_site_year(site: str, year: int, feature_sets: list[str]) -> list[dict]:
    cfg = load_config()
    grid = get_grid(site)
    train = cell_subset(load_table(site, year))
    cells = pd.read_parquet(map_table_path(site, year))
    lidar = load_table(site, year)                       # all lidar cells, for the observed band
    bands, names, meta = [to_grid(grid, lidar, lidar["height"].to_numpy())], ["lidar_p95"], []

    for fs in feature_sets:
        t0 = time.time()
        cols = feature_columns(train, fs)
        X_tr, y_tr = xy(train, fs)
        model = make_model(cfg["model"]["n_estimators"], cfg["model"]["seed"]).fit(X_tr, y_tr)
        X_map = cells[cols].to_numpy(np.float32)
        ok = np.isfinite(X_map).all(axis=1)
        pred = np.full(len(cells), np.nan, dtype="float32")
        pred[ok] = np.maximum(model.predict(X_map[ok]), 0.0)   # heights cannot be negative
        bands.append(to_grid(grid, cells, pred))
        names.append(f"pred_{fs}")
        r2_in = r2_score(y_tr, model.predict(X_tr))
        meta.append({"site": site, "year": year, "learner": cfg["model"]["learner"], "feature_set": fs,
                     "n_train": len(train), "n_selected": len(getattr(model, "selected_", cols)),
                     "n_predicted": int(ok.sum()), "n_cells": len(cells), "r2_in_sample": r2_in,
                     "pred_min": float(np.nanmin(pred)), "pred_max": float(np.nanmax(pred))})
        log(f"{site} {year} {fs}: trained on {len(train)} cells, predicted {int(ok.sum())}/{len(cells)} cells, "
            f"range {np.nanmin(pred):.1f}-{np.nanmax(pred):.1f} m, in-sample R2 {r2_in:.2f} "
            f"({time.time() - t0:.0f} s)")

    out = experiment_dir() / "maps" / f"{site}_{year}.tif"
    write_cell_raster(out, np.stack(bands), names, grid)
    plot_maps(site, year, np.stack(bands), names)
    log(f"{site} {year}: wrote {out}")
    return meta


def plot_maps(site: str, year: int, bands: np.ndarray, names: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = get_grid(site)
    extent = (grid.minx / 1000, grid.maxx / 1000, grid.miny / 1000, grid.maxy / 1000)
    vmax = float(np.nanpercentile(bands[0], 99))
    footprint = np.isfinite(bands[0]).astype(float)
    n = len(names)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * (3.2 * aspect + 1.0), nrow * 3.6), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        im = ax.imshow(bands[i], extent=extent, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
        if i > 0:   # outline the lidar footprint on prediction panels
            ax.contour(footprint, levels=[0.5], colors="white", linewidths=0.6, origin="upper", extent=extent)
        ax.set_title(names[i].replace("pred_", "predicted ").replace("lidar_p95", "lidar P95 (observed)"),
                     fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel(f"x (km, {grid.crs})", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.8, label="canopy height P95 (m)")
    learner = load_config()["model"]["learner"]
    fig.suptitle(f"{site.capitalize()} {year}: 100 m canopy height, {learner} (white = lidar footprint)")
    fig.savefig(experiment_dir() / "maps" / f"{site}_{year}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site")
    ap.add_argument("--year", type=int)
    ap.add_argument("--fs", nargs="+", choices=FEATURE_SETS, default=list(FEATURE_SETS))
    ap.add_argument("--learner", choices=["xgboost", "plsr", "tabicl"], default=None,
                    help="override config model.learner (outputs go to results/{learner}/maps/)")
    args = ap.parse_args()
    if args.learner:
        load_config()["model"]["learner"] = args.learner      # in-memory override of the cached config

    meta = []
    for site, year in site_years():
        if (args.site and site != args.site) or (args.year and year != args.year):
            continue
        meta += map_site_year(site, year, args.fs)

    path = experiment_dir() / "maps" / "maps_meta.csv"
    new = pd.DataFrame(meta)
    if path.exists():   # keep rows of site-years / FS not rerun
        old = pd.read_csv(path)
        key = ["site", "year", "feature_set"]
        old = old[~old.set_index(key).index.isin(new.set_index(key).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
