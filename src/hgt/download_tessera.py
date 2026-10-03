"""Stage 1d: TESSERA v1.1 annual embeddings on the 10 m site grid.

Source: TESSERA v1.1 global store on the AWS Registry of Open Data
(https://registry.opendata.aws/tessera/): Icechunk (Zarr v3) repository
s3://tessera-embeddings/v1.1/dclimate.icechunk, us-west-2, anonymous reads. Variant "dclimate"
("complete global run", checkpoint tessera_v1_1_aws_encoder.pt), the default v1.1 variant of the
geotessera package. Store docs: https://github.com/dClimate/tessera-embeddings/blob/main/docs/global-store.md
License CC-BY-SA (as stated by the AWS registry entry). Attribution / citation: "TESSERA (Temporal
Embeddings of Surface Spectra for Earth Representation and Analysis) was accessed on <DATE> from
https://registry.opendata.aws/tessera"; Feng et al. 2025, "TESSERA: Temporal Embeddings of Surface
Spectra for Earth Representation and Analysis", arXiv:2506.20380.

Layout: one Zarr group per UTM zone and hemisphere ("18N", "36S", ...), each on a single global
10 m grid in that zone's CRS (EPSG:326xx / 327xx), so no reprojection or resampling is needed:
the site grid is a plain array window. Arrays are (time, northing, easting, band): `embeddings`
int8 (9 years 2017-2025, 128 dims), `scales` float32 (one scale per pixel-year, NaN = not
embedded). Inner chunks 256 x 256 px, shards 2048 x 2048 px (= registry "tiles" chunk_{row}_{col});
zarr reads only the inner chunks a window overlaps (byte ranges), so nothing is cached on disk.

Output: data/raw/tessera/{site}_{year}.tif, float32, 128 bands tes_E000 .. tes_E127, de-quantised
per pixel as embedding = int8 * scale (that pixel-year's own scale), NaN where scale is not finite
(pixels with < 15 cloud-free optical observations in the year are left unembedded upstream).

Orientation: the group's `spatial:transform` is north-up (pixel height -10 m) and the `northing`
coordinate is descending; for every window we check that the northing/easting coordinate values
equal the site grid's pixel centres before reading, so row 0 = north and col 0 = west exactly.

Usage:
    uv run python -m hgt.download_tessera                 # all site-years
    uv run python -m hgt.download_tessera --site colombia --year 2021 [--overwrite]
"""
import argparse
import io
import os
import re
import time
import urllib.request
from functools import lru_cache

import numpy as np
import pandas as pd
import rasterio
from affine import Affine

from hgt.config import site_years
from hgt.grid import SiteGrid, get_grid
from hgt.io import check_on_grid, raw_path, results_dir, target_path, write_feature_raster

BUCKET, PREFIX, REGION = "tessera-embeddings", "v1.1/dclimate.icechunk", "us-west-2"
STORE_URL = f"s3://{BUCKET}/{PREFIX}"
HTTPS_ROOT = f"https://{BUCKET}.s3.{REGION}.amazonaws.com"
REGISTRY_PREFIX = "v1.1/dclimate.registry/parts"
N_DIMS = 128
BAND_NAMES = [f"tes_E{i:03d}" for i in range(N_DIMS)]
LICENSE = "CC-BY-SA (AWS Registry of Open Data entry 'tessera')"
ATTRIBUTION = ("TESSERA (Temporal Embeddings of Surface Spectra for Earth Representation and "
               "Analysis) was accessed from https://registry.opendata.aws/tessera. Feng et al. 2025, "
               "arXiv:2506.20380.")
RES = 10.0
SHARD = 2048            # px, = one registry tile
STRIP = 256             # px rows per read = one inner-chunk row (unit of retry)
N_RETRIES = 4


# --- store -------------------------------------------------------------------
@lru_cache
def open_root():
    """Read-only zarr root group of the Icechunk repository (branch main)."""
    import icechunk
    import zarr

    for attempt in range(N_RETRIES):
        try:
            storage = icechunk.s3_storage(bucket=BUCKET, prefix=PREFIX, region=REGION,
                                          anonymous=True)
            session = icechunk.Repository.open(storage).readonly_session(branch="main")
            return session.snapshot_id, zarr.open_group(session.store, mode="r")
        except Exception as e:  # network / storage errors from icechunk
            if attempt == N_RETRIES - 1:
                raise
            print(f"  opening store failed ({e}); retrying")
            time.sleep(10 * 2 ** attempt)


def zone_of(grid: SiteGrid) -> str:
    """Nominal 6-degree UTM zone group (e.g. '18N') of the grid centre; must match the grid CRS.

    Adjacent zone groups overlap by up to a shard, so data must come from the nominal zone.
    """
    w, s, e, n = grid.bbox_wgs84()
    lon, lat = (w + e) / 2, (s + n) / 2
    number = int((lon + 180) // 6) + 1
    hemi = "N" if lat >= 0 else "S"
    epsg = (32600 if hemi == "N" else 32700) + number
    assert grid.crs == f"EPSG:{epsg}", f"{grid.site}: grid CRS {grid.crs} is not the nominal zone"
    assert w >= (number - 1) * 6 - 180 and e <= number * 6 - 180, "grid straddles a zone edge"
    return f"{number:02d}{hemi}"


def window_of(group, grid: SiteGrid) -> tuple[int, int]:
    """(row0, col0) of the site grid's NW pixel in the zone array, with orientation checks."""
    assert group.attrs["crs"] == grid.crs, f"group CRS {group.attrs['crs']} != {grid.crs}"
    a, b, c, d, e, f = group.attrs["spatial:transform"]
    t = Affine(a, b, c, d, e, f)
    assert (t.a, t.b, t.d, t.e) == (RES, 0, 0, -RES), f"unexpected (not north-up 10 m) transform {t}"
    assert group.attrs.get("spatial:registration", "pixel") == "pixel"
    col0, row0 = (grid.minx - t.c) / RES, (t.f - grid.maxy) / RES
    assert col0 == int(col0) and row0 == int(row0), "site grid not aligned to the zone pixel grid"
    row0, col0 = int(row0), int(col0)
    h, w = grid.feature_shape
    # coordinate arrays must be the pixel centres of the site grid (row 0 = north, col 0 = west)
    northing = group["northing"][row0:row0 + h]
    easting = group["easting"][col0:col0 + w]
    assert np.array_equal(northing, grid.maxy - RES * (np.arange(h) + 0.5)), "northing mismatch"
    assert np.array_equal(easting, grid.minx + RES * (np.arange(w) + 0.5)), "easting mismatch"
    return row0, col0


def time_index(group, year: int) -> int:
    assert year in group.attrs["years_complete"], \
        f"{year} not in years_complete {group.attrs['years_complete']} of zone {group.basename}"
    years = group["time"][:].astype("datetime64[ns]").astype("datetime64[Y]").astype(int) + 1970
    hits = np.flatnonzero(years == year)
    assert len(hits) == 1, f"year {year} not in time axis {years}"
    return int(hits[0])


# --- tile registry (per-shard coverage metadata) -----------------------------
def registry_rows(zone: str, year: int, shards: list[tuple[int, int]]) -> pd.DataFrame:
    """Registry rows (latest run) for the given (shard_row, shard_col) tiles."""
    prefix = f"{REGISTRY_PREFIX}/zone={zone}/year={year}/"
    listing = http_get(f"{HTTPS_ROOT}/?list-type=2&prefix={prefix}").decode()
    keys = re.findall(r"<Key>([^<]+\.parquet)</Key>", listing)
    assert keys, f"no registry part for {zone} {year}"
    df = pd.concat([pd.read_parquet(io.BytesIO(http_get(f"{HTTPS_ROOT}/{k}"))) for k in keys])
    df = df.sort_values("assembled_at").drop_duplicates("tile", keep="last")   # latest run wins
    names = [f"chunk_{r}_{c}" for r, c in shards]
    out = df[df.tile.isin(names)].set_index("tile").reindex(names)
    return out


def http_get(url: str) -> bytes:
    for attempt in range(N_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except OSError as e:
            if attempt == N_RETRIES - 1:
                raise
            print(f"  GET {url} failed ({e}); retrying")
            time.sleep(5 * 2 ** attempt)


# --- reading -----------------------------------------------------------------
def read_strip(group, ti: int, r0: int, r1: int, c0: int, c1: int):
    """int8 (rows, cols, 128) and float32 (rows, cols) scales, with retries."""
    for attempt in range(N_RETRIES):
        try:
            q = group["embeddings"][ti, r0:r1, c0:c1, :]
            s = group["scales"][ti, r0:r1, c0:c1]
            return q, s
        except AssertionError:
            raise
        except Exception as e:  # transient object-store errors
            if attempt == N_RETRIES - 1:
                raise
            wait = 10 * 2 ** attempt
            print(f"  read failed ({e}); retry in {wait}s")
            time.sleep(wait)


def read_on_grid(group, ti: int, row0: int, col0: int, grid: SiteGrid) -> tuple[np.ndarray, dict]:
    """De-quantised float32 (128, H, W) on the site grid, NaN where not embedded.

    Reads in strips aligned to the 256 px inner chunks; each strip is de-quantised straight into
    the output, so peak RAM ~ output + one strip.
    """
    h, w = grid.feature_shape
    out = np.empty((N_DIMS, h, w), dtype="float32")
    n_zero_vec = 0
    r = 0
    while r < h:
        r_end = min(h, (((row0 + r) // STRIP) + 1) * STRIP - row0)    # next chunk boundary
        q, s = read_strip(group, ti, row0 + r, row0 + r_end, col0, col0 + w)
        assert q.shape == (r_end - r, w, N_DIMS) and q.dtype == np.int8, (q.shape, q.dtype)
        s = np.where(np.isfinite(s) & (s > 0), s, np.nan).astype("float32")
        n_zero_vec += int((np.isfinite(s) & ~q.any(axis=-1)).sum())
        for b in range(N_DIMS):
            np.multiply(q[:, :, b], s, out=out[b, r:r_end], dtype="float32")
        del q, s
        r = r_end
    return out, dict(n_zero_vec=n_zero_vec)


# --- main build --------------------------------------------------------------
def build(site: str, year: int, overwrite: bool = False) -> None:
    path = raw_path("tessera", site, year)
    if path.exists() and not overwrite:
        print(f"{site} {year}: exists, skipping ({path.name})")
        return
    grid = get_grid(site)
    snapshot, root = open_root()
    zone = zone_of(grid)
    group = root[zone]
    assert int(root.attrs["geoemb:dimensions"]) == N_DIMS
    assert root.attrs["geoemb:quantization"]["method"] == "per_pixel_scale"
    ti = time_index(group, year)
    row0, col0 = window_of(group, grid)
    h, w = grid.feature_shape
    shards = [(sr, sc) for sr in range(row0 // SHARD, (row0 + h - 1) // SHARD + 1)
              for sc in range(col0 // SHARD, (col0 + w - 1) // SHARD + 1)]
    reg = registry_rows(zone, year, shards)
    run_id = group.attrs.get("runs", {}).get(str(year), {}).get("run_id", "")
    tiles = [f"{zone}/{year}/chunk_{sr}_{sc}" for sr, sc in shards]
    print(f"{site} {year}: zone {zone}, snapshot {snapshot}, run {run_id}, window rows "
          f"{row0}:{row0 + h} cols {col0}:{col0 + w}; tiles {', '.join(tiles)}")
    for name, row in reg.iterrows():
        assert row.notna().any(), f"tile {name} missing from registry"
        assert bool(row.embedded), f"tile {name} not embedded for {year}"
        print(f"  {name}: embedded={row.embedded} refused_px={row.refused_px} "
              f"({row.refused_px / row.chunk_px:.1%} of tile; thin={row.refused_thin_px}, "
              f"no_optical={row.refused_no_optical_px})")

    data, info = read_on_grid(group, ti, row0, col0, grid)
    if info["n_zero_vec"]:
        print(f"  WARNING: {info['n_zero_vec']} embedded pixels with an all-zero int8 vector")
    tmp = path.with_name(path.stem + ".tmp.tif")
    # small cache + few compression threads while writing keeps peak RAM down
    with rasterio.Env(GDAL_CACHEMAX=64 * 2**20, GDAL_NUM_THREADS="2"):
        write_feature_raster(tmp, data, BAND_NAMES, grid)
    del data
    with rasterio.open(tmp, "r+") as dst:
        dst.update_tags(TESSERA_SOURCE=STORE_URL, TESSERA_SNAPSHOT=snapshot,
                        TESSERA_ZONE=zone, TESSERA_RUN_ID=run_id,
                        TESSERA_WINDOW=f"time={ti} rows={row0}:{row0 + h} cols={col0}:{col0 + w}",
                        TESSERA_TILES=";".join(tiles), TESSERA_LICENSE=LICENSE,
                        TESSERA_ATTRIBUTION=ATTRIBUTION,
                        TESSERA_DEQUANT="embedding = int8 * scales (per pixel); NaN where scale not finite")
    os.replace(tmp, path)
    check_on_grid(path, grid)
    print(f"  wrote {path}")


# --- QA ----------------------------------------------------------------------
def scan(path, grid: SiteGrid, rgb_bands=(0, 1, 2), strip_rows: int = 100) -> dict:
    """One pass over the raster in row strips (all bands at once: the GeoTIFF is pixel-interleaved,
    so band-by-band reads would decompress every tile 128 times).

    Returns per-pixel sum of squares and NaN-band count, value range, 3 RGB bands and the
    (cell_rows, cell_cols, bands) nan-mean of every 10 x 10 px cell.
    """
    px = grid.px_per_cell
    h, w = grid.feature_shape
    nr, nc = grid.cell_shape
    assert strip_rows % px == 0
    sumsq = np.empty((h, w), dtype="float64")
    n_nan = np.empty((h, w), dtype="uint8")
    rgb = np.empty((h, w, 3), dtype="float32")
    cells = None
    vmin, vmax = np.inf, -np.inf
    with rasterio.open(path) as src:
        for r in range(0, h, strip_rows):
            r1 = min(h, r + strip_rows)
            a = src.read(window=((r, r1), (0, w)))                    # (B, rows, W)
            sumsq[r:r1] = np.einsum("bij,bij->ij", a, a, dtype="float64")
            n_nan[r:r1] = np.isnan(a).sum(axis=0)
            if np.isfinite(a).any():
                vmin, vmax = min(vmin, np.nanmin(a)), max(vmax, np.nanmax(a))
            rgb[r:r1] = np.moveaxis(a[list(rgb_bands)], 0, -1)
            if cells is None:
                cells = np.empty((nr, nc, a.shape[0]), dtype="float32")
            blk = a.reshape(a.shape[0], (r1 - r) // px, px, nc, px)
            cnt = np.isfinite(blk).sum(axis=(2, 4))
            with np.errstate(invalid="ignore", divide="ignore"):
                m = np.nansum(blk, axis=(2, 4)) / cnt
            cells[r // px:r1 // px] = np.moveaxis(np.where(cnt > 0, m, np.nan), 0, -1)
            del a, blk
    return dict(sumsq=sumsq, n_nan=n_nan, vmin=float(vmin), vmax=float(vmax), rgb=rgb,
                cells=cells)


def alignment_r2(cells: np.ndarray, site: str, year: int) -> dict:
    """Ridge on 100 m cell means -> height, 5-fold CV grouped by block_id, under misalignments.

    Variants: as read, N-S flip, E-W flip, shifts of +-2 cells N/S/W/E. All variants are scored on
    the same cells (those with finite features in every variant).
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tg = pd.read_parquet(target_path(site, year))
    nr, nc = cells.shape[:2]

    def shift(a, dr, dc):
        """out[r, c] = a[r - dr, c - dc] (content moves dr rows south, dc cols east)."""
        o = np.full_like(a, np.nan)
        o[max(dr, 0):nr + min(dr, 0), max(dc, 0):nc + min(dc, 0)] = \
            a[max(-dr, 0):nr + min(-dr, 0), max(-dc, 0):nc + min(-dc, 0)]
        return o

    variants = {"as_read": cells, "flip_NS": cells[::-1], "flip_EW": cells[:, ::-1],
                "shift_N2": shift(cells, -2, 0), "shift_S2": shift(cells, 2, 0),
                "shift_W2": shift(cells, 0, -2), "shift_E2": shift(cells, 0, 2)}
    rows, cols = tg.row.to_numpy(), tg.col.to_numpy()
    Xs = {k: v[rows, cols] for k, v in variants.items()}
    ok = np.logical_and.reduce([np.isfinite(X).all(axis=1) for X in Xs.values()])
    y, g = tg.height.to_numpy()[ok], tg.block_id.to_numpy()[ok]
    res = {}
    for name, X in Xs.items():
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
        pred = cross_val_predict(model, X[ok], y, groups=g, cv=GroupKFold(5))
        res[name] = (float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)), int(ok.sum()))
    return res


def qa(site: str, year: int) -> dict:
    """Stats, alignment test and RGB quick-look (E000/E001/E002) for one output raster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = raw_path("tessera", site, year)
    grid = get_grid(site)
    check_on_grid(path, grid)
    with rasterio.open(path) as src:
        assert list(src.descriptions) == BAND_NAMES, "unexpected band names"
        assert src.count == N_DIMS and src.dtypes[0] == "float32"
        tags = src.tags()
    sc = scan(path, grid)

    valid = sc["n_nan"] == 0
    n_partial = int(((sc["n_nan"] > 0) & (sc["n_nan"] < N_DIMS)).sum())
    norm = np.sqrt(sc["sumsq"][valid])
    tg = pd.read_parquet(target_path(site, year))
    px = grid.px_per_cell
    cell_nan = (~valid).reshape(grid.cell_shape[0], px, grid.cell_shape[1], px).mean(axis=(1, 3))
    stats = dict(site=site, year=year, tiles=tags.get("TESSERA_TILES", "").split(";"),
                 pct_nan=100 * (1 - valid.mean()),
                 pct_nan_targets=100 * float(cell_nan[tg.row, tg.col].mean()),
                 pct_target_cells_all_nan=100 * float((cell_nan[tg.row, tg.col] == 1).mean()),
                 n_partial_nan=n_partial, norm_min=norm.min(), norm_median=np.median(norm),
                 norm_max=norm.max(), value_min=sc["vmin"], value_max=sc["vmax"],
                 alignment=alignment_r2(sc["cells"], site, year))

    rgb = sc["rgb"]
    for i in range(3):
        lo, hi = np.nanpercentile(rgb[..., i], [2, 98])
        rgb[..., i] = np.clip((rgb[..., i] - lo) / (hi - lo), 0, 1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    fig, ax = plt.subplots(figsize=(6, 6 * grid.feature_shape[0] / grid.feature_shape[1] + 0.5))
    ax.imshow(rgb, extent=(grid.minx, grid.maxx, grid.miny, grid.maxy))
    ax.set_title(f"TESSERA {site} {year}: E000/E001/E002 (2-98% stretch), north up\n"
                 "black = no embedding (NaN)", fontsize=9)
    ax.set_xlabel(f"easting (m, {grid.crs})")
    ax.set_ylabel("northing (m)")
    fig.tight_layout()
    out = results_dir() / "figures" / f"qa_tessera_{site}_{year}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--site")
    p.add_argument("--year", type=int)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-qa", action="store_true")
    args = p.parse_args()

    todo = [(s, y) for s, y in site_years()
            if (args.site is None or s == args.site) and (args.year is None or y == args.year)]
    assert todo, f"no site-year matches --site {args.site} --year {args.year}"
    for site, year in todo:
        build(site, year, overwrite=args.overwrite)
        if not args.no_qa:
            s = qa(site, year)
            print(f"  QA: tiles={s['tiles']} nan={s['pct_nan']:.2f}% "
                  f"nan in target cells={s['pct_nan_targets']:.2f}% "
                  f"(target cells fully NaN {s['pct_target_cells_all_nan']:.2f}%) "
                  f"partial-NaN px={s['n_partial_nan']} "
                  f"L2 norm min/median/max={s['norm_min']:.3f}/{s['norm_median']:.3f}/"
                  f"{s['norm_max']:.3f} values=[{s['value_min']:.3f}, {s['value_max']:.3f}]")
            print("  alignment R2 (ridge, 5-fold grouped by block): " + ", ".join(
                f"{k}={r2:.3f} (n={n})" for k, (r2, n) in s["alignment"].items()))


if __name__ == "__main__":
    main()
