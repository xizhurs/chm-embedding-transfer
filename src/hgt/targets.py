"""Stage 1c: lidar CHM (0.5 m) -> percentile canopy height (config target.percentile, P95) per 100 m cell.

For each site-year the CHM is warped (nearest neighbour, no smoothing) onto a 0.5 m
grid in the site CRS that shares the site grid origin, so every 100 m cell is an
exact 200 x 200 block of 0.5 m pixels. The warp is read one 1 km block-row at a time
to keep memory low.

Output: data/interim/targets/{site}_{year}.parquet, one row per kept cell:
  cell_id:int64, row:int32, col:int32, x:float64, y:float64, block_id:int32,
  height:float32, valid_frac:float32

Usage:
  uv run python -m hgt.targets                       # all site-years
  uv run python -m hgt.targets --site colombia --year 2022 [--overwrite]
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from hgt.config import load_config, site_years
from hgt.grid import SiteGrid, get_grid
from hgt.io import chm_path, results_dir, target_path

CHM_RES = 0.5          # m
MAX_HEIGHT = 80.0      # m, values above this are treated as nodata

SCHEMA = {
    "cell_id": "int64", "row": "int32", "col": "int32", "x": "float64", "y": "float64",
    "block_id": "int32", "height": "float32", "valid_frac": "float32",
}


def covered_cells(src, grid: SiteGrid) -> tuple[int, int, int, int]:
    """(row0, row1, col0, col1) cell range covering the CHM footprint, clipped to the grid.

    row0 is snapped down to a 1 km block boundary so windows follow block rows."""
    left, bottom, right, top = transform_bounds(src.crs, grid.crs, *src.bounds, densify_pts=21)
    res = grid.cell_res
    nr, nc = grid.cell_shape
    col0 = max(0, math.floor((left - grid.minx) / res))
    col1 = min(nc, math.ceil((right - grid.minx) / res))
    row0 = max(0, math.floor((grid.maxy - top) / res))
    row1 = min(nr, math.ceil((grid.maxy - bottom) / res))
    cpb = grid.cells_per_block
    row0 = (row0 // cpb) * cpb
    return row0, row1, col0, col1


def cell_percentile(blocks: np.ndarray, q: float) -> np.ndarray:
    """Percentile of each row of (n, k) float32 with NaNs, linear interpolation.

    Same result as np.nanpercentile(..., axis=1, method="linear"), but vectorised:
    NaNs sort to the end, so we interpolate inside the first n_valid values."""
    n_valid = np.sum(~np.isnan(blocks), axis=1)
    s = np.sort(blocks, axis=1)
    pos = (n_valid - 1) * (q / 100.0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, n_valid - 1)
    frac = pos - lo
    idx = np.arange(len(s))
    v_lo = s[idx, lo].astype(np.float64)
    v_hi = s[idx, hi].astype(np.float64)
    return v_lo + (v_hi - v_lo) * frac


def compute_targets(site: str, year: int) -> tuple[pd.DataFrame, dict]:
    cfg = load_config()["target"]
    q, min_valid = float(cfg["percentile"]), float(cfg["min_valid_frac"])
    grid = get_grid(site)
    px = int(round(grid.cell_res / CHM_RES))           # 200 px per cell side
    nr_cells, nc_cells = grid.cell_shape
    cpb = grid.cells_per_block

    stats = {"n_negative": 0, "n_too_high": 0, "n_valid_px": 0, "min_raw": np.inf, "max_raw": -np.inf}
    out_rows, out_cols, out_height, out_frac = [], [], [], []

    with rasterio.Env(GDAL_CACHEMAX=256 * 2**20), rasterio.open(chm_path(site, year)) as src:
        row0, row1, col0, col1 = covered_cells(src, grid)
        vrt_opts = dict(
            crs=grid.crs,
            transform=Affine(CHM_RES, 0.0, grid.minx, 0.0, -CHM_RES, grid.maxy),
            width=nc_cells * px, height=nr_cells * px,
            resampling=Resampling.nearest, src_nodata=np.nan, nodata=np.nan,
            dtype="float32", warp_mem_limit=256,
        )
        with WarpedVRT(src, **vrt_opts) as vrt:
            for r in range(row0, row1, cpb):
                r_end = min(r + cpb, row1, nr_cells)
                n_r, n_c = r_end - r, col1 - col0
                win = Window(col0 * px, r * px, n_c * px, n_r * px)
                a = vrt.read(1, window=win, out_dtype="float32")

                # clean values: negatives -> 0, implausibly tall -> NaN
                valid = ~np.isnan(a)
                if valid.any():
                    stats["min_raw"] = min(stats["min_raw"], float(np.nanmin(a)))
                    stats["max_raw"] = max(stats["max_raw"], float(np.nanmax(a)))
                neg = a < 0
                stats["n_negative"] += int(neg.sum())
                a[neg] = 0.0
                high = a > MAX_HEIGHT
                stats["n_too_high"] += int(high.sum())
                a[high] = np.nan
                del neg, high, valid

                # (n_r, px, n_c, px) -> per-cell valid fraction
                blocks = a.reshape(n_r, px, n_c, px)
                n_valid = (~np.isnan(blocks)).sum(axis=(1, 3))
                stats["n_valid_px"] += int(n_valid.sum())
                frac = n_valid / float(px * px)
                rr, cc = np.nonzero(frac >= min_valid)
                if len(rr):
                    kept = blocks[rr, :, cc, :].reshape(len(rr), px * px)
                    out_height.append(cell_percentile(kept, q))
                    out_rows.append(rr + r)
                    out_cols.append(cc + col0)
                    out_frac.append(frac[rr, cc])
                    del kept
                del a, blocks
                print(f"  {site} {year}: cell rows {r}-{r_end - 1} of {row0}-{row1 - 1}, "
                      f"kept {len(rr)}", flush=True)

    stats["window"] = (row0, row1, col0, col1)
    if not out_rows:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in SCHEMA.items()}), stats

    kept = pd.DataFrame({
        "row": np.concatenate(out_rows),
        "col": np.concatenate(out_cols),
        "height": np.concatenate(out_height),
        "valid_frac": np.concatenate(out_frac),
    })
    cells = grid.cell_table()
    df = cells.merge(kept, on=["row", "col"], how="inner", validate="one_to_one")
    df = df[list(SCHEMA)].astype(SCHEMA).sort_values("cell_id").reset_index(drop=True)
    return df, stats


def plot_qa(site: str, year: int) -> Path:
    """Map of target height on the 100 m grid with the 1 km block grid overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = get_grid(site)
    df = pd.read_parquet(target_path(site, year))
    img = np.full(grid.cell_shape, np.nan, dtype="float32")
    img[df["row"].to_numpy(), df["col"].to_numpy()] = df["height"].to_numpy()

    extent = (grid.minx / 1000, grid.maxx / 1000, grid.miny / 1000, grid.maxy / 1000)
    width_km, height_km = extent[1] - extent[0], extent[3] - extent[2]
    fig, ax = plt.subplots(figsize=(2.5 + 6 * width_km / height_km, 7))
    im = ax.imshow(img, extent=extent, cmap="viridis", interpolation="nearest")
    step = grid.block_size / 1000
    for xk in np.arange(extent[0], extent[1] + step / 2, step):
        ax.axvline(xk, color="0.6", lw=0.5)
    for yk in np.arange(extent[2], extent[3] + step / 2, step):
        ax.axhline(yk, color="0.6", lw=0.5)
    pct = load_config()["target"]["percentile"]
    fig.colorbar(im, ax=ax, label=f"P{pct} canopy height (m)", shrink=0.8)
    ax.set_title(f"{site} {year}: P{pct} per 100 m cell (n = {len(df)}), 1 km blocks")
    ax.set_xlabel(f"x (km, {grid.crs})")
    ax.set_ylabel("y (km)")
    out = results_dir() / "figures" / f"qa_targets_{site}_{year}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def run(site: str, year: int, overwrite: bool = False) -> None:
    path = target_path(site, year)
    if path.exists() and not overwrite:
        print(f"{site} {year}: {path} exists, skipping (use --overwrite)")
        plot_qa(site, year)
        return
    t0 = time.time()
    df, stats = compute_targets(site, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    print(f"{site} {year}: {len(df)} cells kept -> {path} ({time.time() - t0:.0f} s)")
    print(f"  raw range [{stats['min_raw']:.2f}, {stats['max_raw']:.2f}] m, "
          f"negative px clipped to 0: {stats['n_negative']}, px > {MAX_HEIGHT:.0f} m set NaN: "
          f"{stats['n_too_high']}, valid px: {stats['n_valid_px']}, cell window {stats['window']}")
    print(f"  QA figure: {plot_qa(site, year)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site")
    ap.add_argument("--year", type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    for site, year in site_years():
        if args.site and site != args.site:
            continue
        if args.year and year != args.year:
            continue
        run(site, year, args.overwrite)


if __name__ == "__main__":
    main()
