"""Stage 1b: AlphaEarth Foundations annual embeddings on the 10 m site grid.

Source: Google AlphaEarth Foundations Satellite Embedding v1 (annual), mirrored on
Source Cooperative (s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual, anonymous).
License CC-BY-4.0. "The AlphaEarth Foundations Satellite Embedding dataset is produced by
Google and Google DeepMind."

Output: data/raw/aef/{site}_{year}.tif, float32, 64 bands aef_A00 .. aef_A63, de-quantised
with ((v / 127.5) ** 2) * sign(v), NaN where the source is -128.

Orientation: the source .tiff files are stored bottom-up (positive y pixel size). Each one has a
sidecar .vrt (a VRTWarpedDataset) that flips it to north-up; we read through the .vrt and check,
for every tile, that a small block read from the raw .tiff and flipped by hand matches the .vrt.

Usage:
    uv run python -m hgt.download_aef                 # all site-years
    uv run python -m hgt.download_aef --site colombia --year 2021 [--overwrite]
"""
import argparse
import os
import time
import urllib.request

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioError
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from hgt.config import ROOT, load_config, site_years
from hgt.grid import SiteGrid, get_grid
from hgt.io import check_on_grid, raw_path, results_dir, write_feature_raster

NODATA = -128
BAND_NAMES = [f"aef_A{i:02d}" for i in range(64)]
ATTRIBUTION = ("The AlphaEarth Foundations Satellite Embedding dataset is produced by "
               "Google and Google DeepMind.")

GDAL_ENV = dict(
    AWS_NO_SIGN_REQUEST="YES",          # the .vrt files point at /vsis3/ paths
    AWS_REGION="us-west-2",
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.vrt",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="2",
    GDAL_CACHEMAX=256 * 2**20,          # bytes, keep the block cache small
    GDAL_NUM_THREADS="ALL_CPUS",        # parallel block fetches: ~10x faster (latency ~1 s/request)
)
N_RETRIES = 4


# --- tile index --------------------------------------------------------------
def index_path():
    return ROOT / "data" / "raw" / "aef" / "aef_index.parquet"


def download_index() -> None:
    """Download the GeoParquet tile index once (~78 MB)."""
    path = index_path()
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    url = load_config()["aef"]["index"]
    tmp = path.with_suffix(".part")
    for attempt in range(N_RETRIES):
        try:
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
            return
        except OSError as e:
            print(f"  index download failed ({e}), retrying")
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"could not download {url}")


def find_tiles(site: str, year: int) -> list[dict]:
    """Index rows (as dicts) for the tiles of `year` that overlap the site grid."""
    download_index()
    grid = get_grid(site)
    zone = load_config()["sites"][site]["aef_zone"]
    cols = ["crs", "path", "year", "utm_zone", "utm_west", "utm_south", "utm_east", "utm_north"]
    # stream the index in batches: loading all ~300k rows at once costs a few hundred MB
    parts = []
    for batch in pq.ParquetFile(index_path()).iter_batches(batch_size=20_000, columns=cols):
        df = batch.to_pandas()
        parts.append(df[(df.year == year) & (df.utm_zone == zone)
                        & (df.utm_west < grid.maxx) & (df.utm_east > grid.minx)
                        & (df.utm_south < grid.maxy) & (df.utm_north > grid.miny)])
    df = pd.concat(parts)
    tiles = df.sort_values("path").to_dict("records")
    assert tiles, f"no AEF tiles for {site} {year}"
    for t in tiles:
        assert t["crs"] == grid.crs, f"tile CRS {t['crs']} != grid CRS {grid.crs}"
    return tiles


def vsi_path(tile: dict, ext: str) -> str:
    """s3://bucket/key.tiff -> /vsis3/bucket/key{ext}"""
    assert tile["path"].startswith("s3://") and tile["path"].endswith(".tiff")
    return "/vsis3/" + tile["path"][len("s3://"):-len(".tiff")] + ext


def tile_id(tile: dict) -> str:
    return tile["path"].rsplit("/", 1)[-1][:-len(".tiff")]


# --- reading -----------------------------------------------------------------
def check_orientation(src_vrt, tile: dict, grid: SiteGrid, n: int = 64) -> None:
    """Assert the .vrt is north-up and matches a hand-flipped read of the bottom-up .tiff.

    Uses band 1 over an n x n block at the north-west corner of the grid/tile overlap.
    """
    west, south, east, north = (tile["utm_west"], tile["utm_south"],
                                tile["utm_east"], tile["utm_north"])
    t = src_vrt.transform
    assert t.a == 10 and t.e == -10 and t.b == 0 and t.d == 0, f"unexpected VRT transform {t}"
    assert np.allclose(src_vrt.bounds, (west, south, east, north)), \
        f"VRT bounds {src_vrt.bounds} != index bounds {(west, south, east, north)}"

    x0 = max(grid.minx, west)
    y1 = min(grid.maxy, north)                 # top of the test block
    col = int(round((x0 - west) / 10))
    vrt_row = int(round((north - y1) / 10))    # north-up: row 0 = north
    vrt_block = src_vrt.read(1, window=Window(col, vrt_row, n, n))

    with rasterio.open(vsi_path(tile, ".tiff")) as src_tif:
        tt = src_tif.transform
        assert tt.e == 10 and np.isclose(tt.f, south), f"raw tiff not bottom-up as expected: {tt}"
        tif_row = int(round((y1 - n * 10 - south) / 10))   # bottom-up: row 0 = south
        tif_block = src_tif.read(1, window=Window(col, tif_row, n, n))[::-1]

    assert np.array_equal(vrt_block, tif_block), f"{tile_id(tile)}: VRT != flipped TIFF"
    assert not np.array_equal(vrt_block, vrt_block[::-1]), "test block is symmetric, inconclusive"


def read_tile_on_grid(tile: dict, grid: SiteGrid) -> np.ndarray:
    """Raw int8 (64, H, W) of one tile resampled (nearest) onto the 10 m grid; -128 outside."""
    h, w = grid.feature_shape
    with rasterio.open(vsi_path(tile, ".vrt")) as src:
        assert src.crs.to_string() == grid.crs, f"tile CRS {src.crs} != {grid.crs}"
        assert src.count == 64 and src.dtypes[0] == "int8"
        check_orientation(src, tile, grid)
        with WarpedVRT(src, crs=grid.crs, transform=grid.feature_transform, width=w, height=h,
                       resampling=Resampling.nearest, src_nodata=NODATA, nodata=NODATA) as vrt:
            return vrt.read()


def read_with_retry(tile: dict, grid: SiteGrid) -> np.ndarray:
    for attempt in range(N_RETRIES):
        try:
            return read_tile_on_grid(tile, grid)
        except RasterioError as e:
            if attempt == N_RETRIES - 1:
                raise
            wait = 10 * 2 ** attempt
            print(f"  read failed ({e}); retry in {wait}s")
            time.sleep(wait)


def dequantise(q: np.ndarray) -> np.ndarray:
    """int8 (B, H, W) -> float32, ((v/127.5)**2)*sign(v), NaN where v == -128. Band by band."""
    out = np.empty(q.shape, dtype="float32")
    for b in range(q.shape[0]):
        v = q[b].astype("float32") / np.float32(127.5)
        out[b] = v * v * np.sign(v)
        out[b][q[b] == NODATA] = np.nan
    return out


# --- main build --------------------------------------------------------------
def build(site: str, year: int, overwrite: bool = False) -> None:
    path = raw_path("aef", site, year)
    if path.exists() and not overwrite:
        print(f"{site} {year}: exists, skipping ({path.name})")
        return
    grid = get_grid(site)
    tiles = find_tiles(site, year)
    print(f"{site} {year}: {len(tiles)} tile(s): {', '.join(tile_id(t) for t in tiles)}")

    mosaic = np.full((64, *grid.feature_shape), NODATA, dtype="int8")
    for tile in tiles:
        data = read_with_retry(tile, grid)
        for b in range(64):                              # first valid wins
            np.copyto(mosaic[b], data[b], where=mosaic[b] == NODATA)
        del data
        print(f"  {tile_id(tile)}: done, {np.mean(mosaic[0] != NODATA):.1%} of grid filled")

    n_masked = np.zeros(grid.feature_shape, dtype="uint8")   # per pixel: bands == -128
    for b in range(64):
        n_masked += mosaic[b] == NODATA
    n_partial = int(((n_masked > 0) & (n_masked < 64)).sum())
    if n_partial:
        print(f"  WARNING: {n_partial} pixels masked in some bands but not all")
    del n_masked

    data = dequantise(mosaic)
    del mosaic
    tmp = path.with_name(path.stem + ".tmp.tif")
    # small cache + few compression threads while writing: keeps peak RAM ~0.4 GB lower
    with rasterio.Env(GDAL_CACHEMAX=64 * 2**20, GDAL_NUM_THREADS="2"):
        write_feature_raster(tmp, data, BAND_NAMES, grid)
    del data
    with rasterio.open(tmp, "r+") as dst:
        dst.update_tags(AEF_TILES=";".join(t["path"] for t in tiles), AEF_LICENSE="CC-BY-4.0",
                        AEF_ATTRIBUTION=ATTRIBUTION)
    os.replace(tmp, path)
    check_on_grid(path, grid)
    print(f"  wrote {path}")


# --- QA ----------------------------------------------------------------------
def qa(site: str, year: int) -> dict:
    """Stats + RGB quick-look (A01/A16/A09) for one output raster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = raw_path("aef", site, year)
    grid = get_grid(site)
    check_on_grid(path, grid)
    with rasterio.open(path) as src:
        assert list(src.descriptions) == BAND_NAMES, "unexpected band names"
        tiles = src.tags().get("AEF_TILES", "")
        sumsq = np.zeros(grid.feature_shape, dtype="float64")
        vmin, vmax, neigh_r = np.inf, -np.inf, []
        for b in range(1, src.count + 1):
            a = src.read(b)
            sumsq += a.astype("float64") ** 2
            vmin, vmax = min(vmin, np.nanmin(a)), max(vmax, np.nanmax(a))
            # correlation of each pixel with its east neighbour (shift / garbage check)
            x, y = a[:, :-1].ravel(), a[:, 1:].ravel()
            ok = np.isfinite(x) & np.isfinite(y)
            neigh_r.append(np.corrcoef(x[ok], y[ok])[0, 1])
        rgb = np.stack([src.read(i + 1) for i in (1, 16, 9)], axis=-1)

    valid = np.isfinite(sumsq)
    norm = np.sqrt(sumsq[valid])
    stats = dict(site=site, year=year, tiles=[p.rsplit("/", 1)[-1] for p in tiles.split(";")],
                 pct_nan=100 * (1 - valid.mean()), norm_min=norm.min(), norm_median=np.median(norm),
                 norm_max=norm.max(), value_min=float(vmin), value_max=float(vmax),
                 neighbour_r_median=float(np.median(neigh_r)))

    for i in range(3):
        lo, hi = np.nanpercentile(rgb[..., i], [2, 98])
        rgb[..., i] = np.clip((rgb[..., i] - lo) / (hi - lo), 0, 1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    fig, ax = plt.subplots(figsize=(6, 6 * grid.feature_shape[0] / grid.feature_shape[1] + 0.5))
    ax.imshow(rgb, extent=(grid.minx, grid.maxx, grid.miny, grid.maxy))
    ax.set_title(f"AEF {site} {year}: A01/A16/A09 (2-98% stretch), north up")
    ax.set_xlabel(f"easting (m, {grid.crs})")
    ax.set_ylabel("northing (m)")
    fig.tight_layout()
    out = results_dir() / "figures" / f"qa_aef_{site}_{year}.png"
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
    with rasterio.Env(**GDAL_ENV):
        for site, year in todo:
            build(site, year, overwrite=args.overwrite)
            if not args.no_qa:
                s = qa(site, year)
                print(f"  QA: tiles={s['tiles']} nan={s['pct_nan']:.2f}% "
                      f"L2 norm min/median/max={s['norm_min']:.3f}/{s['norm_median']:.3f}/"
                      f"{s['norm_max']:.3f} values=[{s['value_min']:.3f}, {s['value_max']:.3f}] "
                      f"east-neighbour r (median over bands)={s['neighbour_r_median']:.3f}")


if __name__ == "__main__":
    main()
