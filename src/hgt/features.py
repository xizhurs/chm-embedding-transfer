"""Stage 2: per-cell feature tables (handcrafted S1/S2 features + AlphaEarth and Tessera embeddings).

For each site-year, every kept 100 m cell of the target table gets:
  - hc_{band}_mean, hc_{band}_std   for every S2 and S1 band (10 x 10 px, nan-aware, ddof=0)
  - hc_glcm_{input}_{prop}          one GLCM per cell from its own 10 x 10 patch, quantised with
                                    the fixed global range in config glcm.ranges, props averaged
                                    over the 4 angles (entropy in bits, -sum p log2 p)
  - aef_{Axx}_mean, aef_{Axx}_std   mean of the 64-d embedding over valid pixels, re-normalised
                                    to unit L2 norm; per-dimension nan-std (ddof=0)
  - tes_{Exxx}_mean, tes_{Exxx}_std  Tessera embedding: per-dimension nanmean and nan-std (ddof=0),
                                    no re-normalisation (Tessera vectors are not unit-norm)
Pixel (r, c) of the 10 m grid belongs to cell (r // 10, c // 10) by the grid design.

Output: data/tables/{site}_{year}.parquet = target table (unchanged, sorted by cell_id)
plus the float32 feature columns above.

Usage:
  uv run python -m hgt.features                       # all site-years
  uv run python -m hgt.features --site colombia --year 2022 [--overwrite]
  uv run python -m hgt.features --check               # QA report on the written tables
  uv run python -m hgt.features --maps                # feature tables for ALL grid cells (maps)
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
import rasterio
from skimage.feature import graycomatrix, graycoprops

from hgt.config import load_config, site_years
from hgt.grid import get_grid
from hgt.io import check_on_grid, map_table_path, raw_path, read_feature_raster, table_path, target_path

ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
EMB_CHUNK = 16     # bands per read, keeps an embedding stack under ~100 MB at a time


def cell_mean_std(stack: np.ndarray, px: int) -> tuple[np.ndarray, np.ndarray]:
    """(B, H, W) -> per-cell nanmean and nanstd (ddof=0), each (B, H/px, W/px)."""
    b, h, w = stack.shape
    blocks = stack.reshape(b, h // px, px, w // px, px)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN cells -> NaN
        mean = np.nanmean(blocks, axis=(2, 4))
        std = np.nanstd(blocks, axis=(2, 4))
    return mean.astype("float32"), std.astype("float32")


def handcrafted_stats(kind: str, site: str, year: int, rows, cols, px: int) -> tuple[dict, dict]:
    """hc_{band}_mean/std columns for one S1/S2 stack, plus the stack's GLCM input bands."""
    stack, names = read_feature_raster(raw_path(kind, site, year), site)
    mean, std = cell_mean_std(stack, px)
    cols_out = {}
    for i, name in enumerate(names):
        cols_out[f"hc_{name}_mean"] = mean[i, rows, cols]
        cols_out[f"hc_{name}_std"] = std[i, rows, cols]
    glcm_inputs = load_config()["glcm"]["inputs"]
    bands = {name: stack[i].copy() for i, name in enumerate(names) if name in glcm_inputs}
    return cols_out, bands


def quantise(v: np.ndarray, lo: float, hi: float, levels: int) -> np.ndarray:
    v = np.clip(v, lo, hi)
    q = np.floor((v - lo) / (hi - lo) * levels)
    return np.clip(q, 0, levels - 1).astype(np.uint8)


def glcm_features(band: np.ndarray, name: str, rows, cols, px: int) -> tuple[dict, dict]:
    """hc_glcm_{name}_{prop} columns for one input band, plus QA stats."""
    cfg = load_config()["glcm"]
    lo, hi = cfg["ranges"][name]
    levels, dist, props = int(cfg["levels"]), int(cfg["distance"]), cfg["props"]
    out = {p: np.full(len(rows), np.nan, dtype="float64") for p in props}
    n_uniform = n_all_nan = n_filled = n_px = n_low = n_high = 0

    for k, (r, c) in enumerate(zip(rows, cols)):
        patch = band[r * px:(r + 1) * px, c * px:(c + 1) * px]
        nan = np.isnan(patch)
        if nan.all():
            n_all_nan += 1
            continue
        if nan.any():
            n_filled += 1
            patch = np.where(nan, np.nanmedian(patch), patch)
        n_px += int((~nan).sum())
        n_low += int((patch[~nan] < lo).sum())
        n_high += int((patch[~nan] > hi).sum())

        q = quantise(patch, lo, hi, levels)
        if q.min() == q.max():
            n_uniform += 1
        P = graycomatrix(q, distances=[dist], angles=ANGLES, levels=levels,
                         symmetric=True, normed=True)
        for p in props:
            if p == "entropy":
                # bits, independent of the skimage version (skimage's own uses nats)
                p0 = P[:, :, 0, :]
                logp = np.log2(p0, where=p0 > 0, out=np.zeros_like(p0))
                val = -np.sum(p0 * logp, axis=(0, 1)).mean() + 0.0   # no -0.0
            else:
                v = graycoprops(P, p)[0]
                if p == "correlation":
                    v = np.where(np.isnan(v), 1.0, v)   # uniform patch -> 1
                val = v.mean()
            out[p][k] = val

    cols_out = {f"hc_glcm_{name}_{p}": out[p].astype("float32") for p in props}
    stats = {"input": name, "n_uniform": n_uniform, "n_all_nan": n_all_nan, "n_nan_filled": n_filled,
             "pct_low": 100 * n_low / max(n_px, 1), "pct_high": 100 * n_high / max(n_px, 1)}
    return cols_out, stats


def embedding_features(kind: str, prefix: str, site: str, year: int, rows, cols, px: int,
                       renormalise: bool) -> tuple[dict, float]:
    """{prefix}_{dim}_mean and {prefix}_{dim}_std columns for an embedding stack (read in band
    chunks), plus the median norm of the cell mean vectors. With `renormalise` the mean vector
    is scaled to unit L2 norm (AEF: mean of unit vectors on the sphere)."""
    path = raw_path(kind, site, year)
    check_on_grid(path, get_grid(site))
    means, stds, names = [], [], []
    with rasterio.open(path) as src:
        all_names = list(src.descriptions)
        for b0 in range(0, src.count, EMB_CHUNK):
            idx = list(range(b0 + 1, min(b0 + EMB_CHUNK, src.count) + 1))
            stack = src.read(idx, out_dtype="float32")
            m, s = cell_mean_std(stack, px)
            means.append(m[:, rows, cols])
            stds.append(s[:, rows, cols])
            names += all_names[b0:b0 + len(idx)]
            del stack, m, s
    mean = np.concatenate(means).astype("float64")       # (n_dims, n_cells)
    std = np.concatenate(stds)
    norm = np.sqrt(np.sum(mean ** 2, axis=0))
    if renormalise:
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = mean / norm
    cols_out = {}
    for i, name in enumerate(names):
        cols_out[f"{prefix}_{name.removeprefix(prefix + '_')}_mean"] = mean[i].astype("float32")
    for i, name in enumerate(names):
        cols_out[f"{prefix}_{name.removeprefix(prefix + '_')}_std"] = std[i].astype("float32")
    return cols_out, float(np.nanmedian(norm))


def build_table(site: str, year: int) -> tuple[pd.DataFrame, dict]:
    """Target table (lidar cells) + features."""
    df = pd.read_parquet(target_path(site, year)).sort_values("cell_id").reset_index(drop=True)
    return build_features(site, year, df)


def build_map_table(site: str, year: int) -> pd.DataFrame:
    """All 100 m cells of the site grid + features (for wall-to-wall maps). Cells without
    Tessera data keep NaN tes_* features; S1, S2 and AEF cover the whole grid."""
    cells = get_grid(site).cell_table()
    out, _ = build_features(site, year, cells)
    # the shared feature code must give identical values for the lidar cells
    if table_path(site, year).exists():
        tab = pd.read_parquet(table_path(site, year))
        feat = [c for c in tab.columns if c.startswith(("hc_", "aef_", "tes_"))]
        a = out.set_index("cell_id").loc[tab["cell_id"], feat].to_numpy()
        assert np.array_equal(a, tab[feat].to_numpy(), equal_nan=True), "map features differ from table"
    return out


def build_features(site: str, year: int, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Feature columns for the cells of df (needs row/col on the 100 m grid), appended to df."""
    grid = get_grid(site)
    px = grid.px_per_cell
    rows, cols = df["row"].to_numpy(), df["col"].to_numpy()

    feats, glcm_bands = {}, {}
    for kind in ("s2", "s1"):
        hc, bands = handcrafted_stats(kind, site, year, rows, cols, px)
        feats.update(hc)
        glcm_bands.update(bands)

    glcm_stats = []
    for name in load_config()["glcm"]["inputs"]:
        g, st = glcm_features(glcm_bands.pop(name), name, rows, cols, px)
        feats.update(g)
        glcm_stats.append(st)

    aef, median_norm = embedding_features("aef", "aef", site, year, rows, cols, px, renormalise=True)
    feats.update(aef)
    tes, tes_norm = embedding_features("tessera", "tes", site, year, rows, cols, px, renormalise=False)
    feats.update(tes)

    out = pd.concat([df, pd.DataFrame(feats)], axis=1)
    return out, {"glcm": glcm_stats, "aef_median_raw_norm": median_norm, "tes_median_norm": tes_norm}


def run(site: str, year: int, overwrite: bool = False) -> None:
    path = table_path(site, year)
    if path.exists() and not overwrite:
        print(f"{site} {year}: {path} exists, skipping (use --overwrite)")
        return
    t0 = time.time()
    df, stats = build_table(site, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    print(f"{site} {year}: {len(df)} rows x {df.shape[1]} cols -> {path} ({time.time() - t0:.0f} s)")
    print(f"  AEF median norm of the cell mean vector (before re-normalising): "
          f"{stats['aef_median_raw_norm']:.4f}; Tessera median cell-mean norm {stats['tes_median_norm']:.3f}")
    for st in stats["glcm"]:
        flag = "  <-- >2% clipped" if max(st["pct_low"], st["pct_high"]) > 2 else ""
        print(f"  GLCM {st['input']:16s} clipped low {st['pct_low']:6.2f}%  high {st['pct_high']:6.2f}%  "
              f"uniform cells {st['n_uniform']}  all-NaN {st['n_all_nan']}  "
              f"NaN-filled {st['n_nan_filled']}{flag}")


def run_map(site: str, year: int, overwrite: bool = False) -> None:
    path = map_table_path(site, year)
    if path.exists() and not overwrite:
        print(f"{site} {year}: {path} exists, skipping (use --overwrite)")
        return
    t0 = time.time()
    df = build_map_table(site, year)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    tes = [c for c in df.columns if c.startswith("tes_")]
    print(f"{site} {year}: map table {len(df)} cells x {df.shape[1]} cols -> {path} "
          f"({time.time() - t0:.0f} s); cells without Tessera "
          f"{100 * df[tes].isna().all(axis=1).mean():.1f}%; lidar-cell features identical to the table")


# --- QA ----------------------------------------------------------------------

def column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    hc = [c for c in df.columns if c.startswith("hc_")]
    return {
        "hc_meanstd": [c for c in hc if not c.startswith("hc_glcm_")],
        "hc_glcm": [c for c in hc if c.startswith("hc_glcm_")],
        "aef_mean": [c for c in df.columns if c.startswith("aef_") and c.endswith("_mean")],
        "aef_std": [c for c in df.columns if c.startswith("aef_") and c.endswith("_std")],
        "tes_mean": [c for c in df.columns if c.startswith("tes_") and c.endswith("_mean")],
        "tes_std": [c for c in df.columns if c.startswith("tes_") and c.endswith("_std")],
    }


def check() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    tables = {f"{s}_{y}": pd.read_parquet(table_path(s, y)) for s, y in site_years()}

    print("== 1/2. shape, order, column groups, NaN ==")
    for (s, y), (key, df) in zip(site_years(), tables.items()):
        tgt = pd.read_parquet(target_path(s, y))
        same = len(df) == len(tgt) and np.array_equal(df["cell_id"].to_numpy(), tgt["cell_id"].to_numpy())
        same_cols = df[tgt.columns].equals(tgt)
        groups = column_groups(df)
        feat_cols = sum(groups.values(), [])
        non_f32 = [c for c in feat_cols if df[c].dtype != np.float32]
        print(f"{key}: rows {len(df)}, cell_id order == targets: {same}, target cols unchanged: {same_cols}, "
              f"non-float32 features: {len(non_f32)}")
        for g, cs in groups.items():
            nan = df[cs].isna()
            print(f"  {g:11s} n={len(cs):3d}  NaN cells {100 * nan.any(axis=1).mean():.2f}%  "
                  f"NaN values {100 * nan.to_numpy().mean():.3f}%")
        all_nan = [c for c in feat_cols if df[c].isna().all()]
        const = [c for c in feat_cols if df[c].nunique(dropna=True) <= 1 and c not in all_nan]
        print(f"  all-NaN columns: {all_nan or 'none'}; constant columns: {const or 'none'}")

    print("\n== 3. GLCM prop ranges (min / median / max over cells) ==")
    rows = []
    for key, df in tables.items():
        for c in column_groups(df)["hc_glcm"]:
            v = df[c]
            rows.append({"table": key, "col": c.removeprefix("hc_glcm_"),
                         "min": v.min(), "median": v.median(), "max": v.max()})
    print(pd.DataFrame(rows).pivot(index="col", columns="table",
                                   values=["min", "median", "max"]).round(3).to_string())

    print("\n== 4. AEF mean-vector norm ==")
    for key, df in tables.items():
        m = df[column_groups(df)["aef_mean"]].to_numpy("float64")
        norm = np.sqrt(np.sum(m ** 2, axis=1))
        ok = np.abs(norm[~np.isnan(norm)] - 1) <= 1e-5
        print(f"{key}: max |norm - 1| = {np.nanmax(np.abs(norm - 1)):.2e}, all within 1e-5: {ok.all()}, "
              f"NaN rows {np.isnan(norm).sum()}")

    print("\n== 5. Spearman with height ==")
    rows = []
    for key, df in tables.items():
        r = {"table": key}
        for c in ["hc_s2_NDVI_p50_mean", "hc_s1_vh_db_median_mean", "hc_glcm_s2_B08_median_homogeneity"]:
            r[c] = df["height"].corr(df[c], method="spearman")
        aef = df[column_groups(df)["aef_mean"]].corrwith(df["height"], method="spearman")
        r["max|aef_mean|"] = aef.abs().max()
        r["argmax"] = aef.abs().idxmax()
        rows.append(r)
    print(pd.DataFrame(rows).set_index("table").round(3).to_string())

    print("\n== 6. Mean per site-year (distribution shift preview) ==")
    key_cols = ["height", "hc_s2_B08_median_mean", "hc_s2_NDVI_p50_mean", "hc_s2_B11_median_mean",
                "hc_s1_vv_db_median_mean", "hc_s1_vh_db_median_mean",
                "aef_A00_mean", "aef_A10_mean", "aef_A20_mean", "aef_A40_mean", "aef_A63_mean"]
    print(pd.DataFrame({k: df[key_cols].mean() for k, df in tables.items()}).round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site")
    ap.add_argument("--year", type=int)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check", action="store_true", help="only print the QA report")
    ap.add_argument("--maps", action="store_true",
                    help="build map tables (all grid cells) instead of lidar-cell tables")
    args = ap.parse_args()
    if args.check:
        check()
        return
    for site, year in site_years():
        if args.site and site != args.site:
            continue
        if args.year and year != args.year:
            continue
        if args.maps:
            run_map(site, year, args.overwrite)
        else:
            run(site, year, args.overwrite)


if __name__ == "__main__":
    main()
