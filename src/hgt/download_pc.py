"""Stage 1a: annual Sentinel-2 / Sentinel-1 composites from Microsoft Planetary Computer.

For each site-year, writes on the exact 10 m site grid (see docs/CONTRACTS.md):
  data/raw/s2/{site}_{year}.tif + _qa.tif   band medians, index p10/p50/p90, s2_n_clear
  data/raw/s1/{site}_{year}.tif + _qa.tif   dB median/std/p10/p90 for vv, vh, vhvv, s1_n_obs

Memory strategy: scenes are loaded one at a time onto the site grid and written into an
on-disk scratch cube (numpy memmap in the system temp dir, outside OneDrive). Per-pixel
statistics are then computed in row strips from that cube, so peak RAM stays well under 1 GB.

Observations: items of the same UTC date (e.g. two overlapping MGRS tiles, or consecutive
S1 frames of one pass) are mosaicked into a single observation (first valid pixel wins),
so n_clear / n_obs count distinct acquisition dates per pixel.

Usage:
  uv run python -m hgt.download_pc                     # all site-years
  uv run python -m hgt.download_pc --site colombia --year 2021 [--overwrite] [--report-only]
"""
import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import odc.stac
import planetary_computer
import pystac_client
import rasterio
from odc.geo.geobox import GeoBox

from hgt.config import load_config, site_years
from hgt.grid import SiteGrid, get_grid
from hgt.io import check_on_grid, raw_path, results_dir, write_feature_raster

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_NODATA = 65535          # sentinel in the uint16 scratch cube (reflectance * 10000)
STRIP_BUDGET = 100e6       # bytes per (time, rows, W) float32 array when computing stats
N_TRIES = 4
LOAD_TIMEOUT = 300         # s; watchdog per item load (GDAL reads can hang without erroring)

log = logging.getLogger("download_pc")


# --- helpers ---------------------------------------------------------------
def geobox(grid: SiteGrid) -> GeoBox:
    return GeoBox(grid.feature_shape, grid.feature_transform, grid.crs)


def retry(fn, desc: str):
    """Call fn(); retry transient failures with backoff. Re-raises after N_TRIES."""
    for k in range(N_TRIES):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - network/GDAL errors come in many types
            if k == N_TRIES - 1:
                raise
            log.warning("  %s failed (%s: %s), retry %d", desc, type(e).__name__, str(e)[:200], k + 1)
            time.sleep(2 ** (k + 1))


def load_item(item, bands, gbox, resampling) -> dict[str, np.ndarray]:
    """Load bands of one STAC item onto the grid. Re-signs on every attempt (SAS tokens expire)."""
    def _load():
        ds = odc.stac.load([planetary_computer.sign(item)], bands=bands, geobox=gbox,
                           resampling=resampling, groupby="id", pool=4, fail_on_error=True)
        return {b: ds[b].values[0] for b in bands}

    def _load_with_watchdog():
        # A hung read is abandoned (its thread is left behind) and the load retried.
        ex = ThreadPoolExecutor(max_workers=1)
        try:
            return ex.submit(_load).result(timeout=LOAD_TIMEOUT)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    return retry(_load_with_watchdog, f"load {item.id} {bands}")


def nan_percentiles(a: np.ndarray, qs) -> list[np.ndarray]:
    """Per-pixel percentiles along axis 0 ignoring NaN (numpy 'linear' method).

    Sorts `a` in place (NaN sort last) and interpolates, which is much faster than
    np.nanpercentile along an axis.
    """
    a.sort(axis=0)
    n = (~np.isnan(a)).sum(axis=0)
    out = []
    for q in qs:
        pos = np.maximum(n - 1, 0) * (q / 100.0)
        lo = np.floor(pos).astype(np.intp)
        hi = np.ceil(pos).astype(np.intp)
        vlo = np.take_along_axis(a, lo[None], axis=0)[0]
        vhi = np.take_along_axis(a, hi[None], axis=0)[0]
        v = vlo + (vhi - vlo) * (pos - lo)
        v[n == 0] = np.nan
        out.append(v.astype("float32"))
    return out


def strips(n_t: int, grid: SiteGrid):
    """Row slices so that a (n_t, rows, W) float32 array stays within STRIP_BUDGET."""
    H, W = grid.feature_shape
    rows = int(np.clip(STRIP_BUDGET // max(n_t * W * 4, 1), 8, H))
    for r0 in range(0, H, rows):
        yield slice(r0, min(r0 + rows, H))


def by_date(items) -> list[list]:
    """Group items by UTC acquisition date, sorted by date."""
    groups = defaultdict(list)
    for it in items:
        groups[it.datetime.date()].append(it)
    return [groups[d] for d in sorted(groups)]


def scratch_cube(name: str, shape, dtype) -> tuple[np.memmap, Path]:
    d = Path(tempfile.mkdtemp(prefix="hgt_pc_"))
    path = d / f"{name}.dat"
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape), d


def write_with_tags(path: Path, data, names, grid, tags: dict) -> None:
    write_feature_raster(path, data, names, grid)
    with rasterio.open(path, "r+") as dst:
        dst.update_tags(**{k: json.dumps(v) for k, v in tags.items()})
    check_on_grid(path, grid)


def search(catalog, collection, grid, year):
    return list(catalog.search(collections=[collection], bbox=grid.bbox_wgs84(buffer_m=200),
                               datetime=f"{year}-01-01/{year}-12-31").items())


# --- Sentinel-2 ------------------------------------------------------------
def s2_band_names(cfg) -> list[str]:
    s2 = cfg["s2"]
    return ([f"s2_{b}_median" for b in s2["bands"]]
            + [f"s2_{i}_p{p}" for i in s2["indices"] for p in s2["index_percentiles"]])


def s2_items(catalog, grid, year, cfg) -> tuple[list, dict]:
    raw = search(catalog, cfg["s2"]["collection"], grid, year)
    cloud = [it for it in raw if it.properties.get("eo:cloud_cover", 100) < cfg["s2"]["max_cloud_cover"]]
    # De-duplicate reprocessed acquisitions: keep the highest processing baseline per (datetime, tile).
    best = {}
    for it in cloud:
        key = (it.datetime.replace(microsecond=0), it.properties["s2:mgrs_tile"])
        if key not in best or (float(it.properties["s2:processing_baseline"])
                               > float(best[key].properties["s2:processing_baseline"])):
            best[key] = it
    dedup = sorted(best.values(), key=lambda it: it.datetime)
    counts = dict(raw=len(raw), cloud_filtered=len(cloud), dedup=len(dedup),
                  dates=len(by_date(dedup)),
                  tiles=sorted({it.properties["s2:mgrs_tile"] for it in dedup}),
                  baselines=sorted({it.properties["s2:processing_baseline"] for it in dedup}))
    return dedup, counts


def index_series(name: str, r: dict) -> np.ndarray:
    """One spectral index per observation from reflectance arrays (NaN = no obs)."""
    def nd(a, b):
        den = a + b
        return np.where(den != 0, (a - b) / np.where(den != 0, den, 1), np.nan)

    if name == "NDVI":
        return nd(r["B08"], r["B04"])
    if name == "EVI":
        den = r["B08"] + 6 * r["B04"] - 7.5 * r["B02"] + 1
        evi = 2.5 * (r["B08"] - r["B04"]) / np.where(den != 0, den, np.nan)
        return np.clip(evi, -1.0, 1.5)
    if name == "NDMI":
        return nd(r["B08"], r["B11"])
    if name == "NBR":
        return nd(r["B08"], r["B12"])
    if name == "NDRE":
        return nd(r["B8A"], r["B05"])
    raise ValueError(name)


INDEX_BANDS = {"NDVI": ["B08", "B04"], "EVI": ["B08", "B04", "B02"], "NDMI": ["B08", "B11"],
               "NBR": ["B08", "B12"], "NDRE": ["B8A", "B05"]}


def run_s2(catalog, site, year, grid, cfg) -> None:
    s2 = cfg["s2"]
    bands = s2["bands"]
    gbox = geobox(grid)
    H, W = grid.feature_shape
    items, counts = s2_items(catalog, grid, year, cfg)
    groups = by_date(items)
    log.info("[S2 %s %d] items raw=%d -> cloud<%s: %d -> dedup: %d (%d dates, tiles %s, baselines %s)",
             site, year, counts["raw"], s2["max_cloud_cover"], counts["cloud_filtered"],
             counts["dedup"], counts["dates"], counts["tiles"], counts["baselines"])

    cube, tmpdir = scratch_cube("s2", (len(bands), len(groups), H, W), "uint16")
    keep = np.array(s2["scl_keep"])
    resampling = {b: "bilinear" for b in bands}
    n_t, failed, no_clear = 0, [], 0
    try:
        for gi, group in enumerate(groups):
            acc = np.full((len(bands), H, W), S2_NODATA, dtype="uint16")
            for it in group:
                try:
                    scl = load_item(it, ["SCL"], gbox, {"SCL": "nearest"})["SCL"]
                    clear = np.isin(scl, keep)
                    if not clear.any():
                        continue
                    dn = load_item(it, bands, gbox, resampling)
                except Exception as e:  # noqa: BLE001
                    log.error("  skip %s: %s", it.id, str(e)[:200])
                    failed.append(it.id)
                    continue
                # Per-item offset: baseline >= 04.00 adds a +1000 DN offset.
                offset = 1000 if float(it.properties["s2:processing_baseline"]) >= 4.0 else 0
                valid = clear & np.all([dn[b] > 0 for b in bands], axis=0) & (acc[0] == S2_NODATA)
                for k, b in enumerate(bands):
                    refl = np.clip((dn[b].astype("float32") - offset) / 10000.0, 0.0, 1.0)
                    acc[k][valid] = np.round(refl[valid] * 10000).astype("uint16")
            if (acc[0] != S2_NODATA).any():
                cube[:, n_t] = acc
                n_t += 1
            else:
                no_clear += 1
            if (gi + 1) % 10 == 0 or gi == len(groups) - 1:
                log.info("  S2 %s %d: %d/%d dates done, %d with clear pixels, %d failed items",
                         site, year, gi + 1, len(groups), n_t, len(failed))
        cube.flush()

        # Per-strip statistics from the scratch cube.
        names = s2_band_names(cfg)
        out = np.full((len(names), H, W), np.nan, dtype="float32")
        n_clear = np.zeros((H, W), dtype="float32")
        qs = s2["index_percentiles"]

        def refl_strip(b, rs):
            a = np.asarray(cube[bands.index(b), :n_t, rs, :]).astype("float32")
            a[a == S2_NODATA] = np.nan
            return a / 10000.0

        if n_t > 0:
            for rs in strips(n_t, grid):
                n_clear[rs] = (cube[0, :n_t, rs, :] != S2_NODATA).sum(axis=0)
                for k, b in enumerate(bands):
                    out[k, rs] = nan_percentiles(refl_strip(b, rs), [50])[0]
                k = len(bands)
                for idx in s2["indices"]:
                    r = {b: refl_strip(b, rs) for b in INDEX_BANDS[idx]}
                    series = index_series(idx, r).astype("float32")
                    del r
                    for v in nan_percentiles(series, qs):
                        out[k, rs] = v
                        k += 1
                    del series
                log.info("  S2 stats rows %d-%d done", rs.start, rs.stop)

        counts.update(dates_with_clear=n_t, dates_no_clear=no_clear, failed_items=failed)
        write_with_tags(raw_path("s2", site, year), out, names, grid, {"counts": counts})
        write_with_tags(raw_path("s2", site, year, "_qa"), n_clear, ["s2_n_clear"], grid,
                        {"counts": counts})
        log.info("[S2 %s %d] written (%d dates with clear obs, %d failed items)",
                 site, year, n_t, len(failed))
    finally:
        del cube
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- Sentinel-1 ------------------------------------------------------------
def s1_band_names() -> list[str]:
    return [f"s1_{p}_db_{s}" for p in ["vv", "vh", "vhvv"] for s in ["median", "std", "p10", "p90"]]


def run_s1(catalog, site, year, grid, cfg) -> None:
    s1 = cfg["s1"]
    gbox = geobox(grid)
    H, W = grid.feature_shape
    raw = search(catalog, s1["collection"], grid, year)
    # Descending + Sentinel-1A only, for all site-years (platform is "SENTINEL-1A" in the catalog).
    items = [it for it in raw
             if it.properties.get("sat:orbit_state") == s1["orbit_state"]
             and it.properties.get("platform", "").lower() == s1["platform"]]
    items.sort(key=lambda it: it.datetime)
    groups = by_date(items)
    counts = dict(raw=len(raw), filtered=len(items), dates=len(groups),
                  relative_orbits=sorted({it.properties.get("sat:relative_orbit") for it in items}))
    log.info("[S1 %s %d] scenes raw=%d -> descending+S1A: %d (%d dates, rel. orbits %s)",
             site, year, len(raw), len(items), len(groups), counts["relative_orbits"])

    # Bilinear resampling of linear gamma0 onto the grid (RTC is already ~10 m UTM; nodata is
    # excluded by the warper), then conversion to dB per scene.
    resampling = {"vv": "bilinear", "vh": "bilinear"}
    cube, tmpdir = scratch_cube("s1", (2, len(groups), H, W), "float32")
    n_t, failed = 0, []
    try:
        for gi, group in enumerate(groups):
            acc = np.full((2, H, W), np.nan, dtype="float32")
            for it in group:
                try:
                    d = load_item(it, ["vv", "vh"], gbox, resampling)
                except Exception as e:  # noqa: BLE001
                    log.error("  skip %s: %s", it.id, str(e)[:200])
                    failed.append(it.id)
                    continue
                vv, vh = d["vv"].astype("float32"), d["vh"].astype("float32")
                ok = (vv > 0) & (vh > 0) & np.isfinite(vv) & np.isfinite(vh) & np.isnan(acc[0])
                acc[0][ok] = 10 * np.log10(vv[ok])
                acc[1][ok] = 10 * np.log10(vh[ok])
            if np.isfinite(acc[0]).any():
                cube[:, n_t] = acc
                n_t += 1
            log.info("  S1 %s %d: %d/%d dates done", site, year, gi + 1, len(groups))
        cube.flush()

        names = s1_band_names()
        out = np.full((len(names), H, W), np.nan, dtype="float32")
        n_obs = np.zeros((H, W), dtype="float32")
        if n_t > 0:
            for rs in strips(n_t, grid):
                vv = np.array(cube[0, :n_t, rs, :])
                vh = np.array(cube[1, :n_t, rs, :])
                n_obs[rs] = np.isfinite(vv).sum(axis=0)
                for k, a in enumerate([vv, vh, vh - vv]):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        std = np.nanstd(a, axis=0)
                    med, p10, p90 = nan_percentiles(a, [50, 10, 90])
                    out[4 * k:4 * k + 4, rs] = [med, std, p10, p90]

        counts.update(dates_with_obs=n_t, failed_items=failed)
        write_with_tags(raw_path("s1", site, year), out, names, grid, {"counts": counts})
        write_with_tags(raw_path("s1", site, year, "_qa"), n_obs, ["s1_n_obs"], grid,
                        {"counts": counts})
        log.info("[S1 %s %d] written (%d dates, %d failed scenes)", site, year, n_t, len(failed))
    finally:
        del cube
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- QA report -------------------------------------------------------------
def report(site: str, year: int) -> None:
    """Print acceptance-check numbers and save a quick-look PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = get_grid(site)
    arrs = {}
    print(f"=== {site} {year} ===")
    for kind in ["s2", "s1"]:
        for suffix in ["", "_qa"]:
            path = raw_path(kind, site, year, suffix)
            check_on_grid(path, grid)
            with rasterio.open(path) as src:
                data = src.read()
                names = list(src.descriptions)
                counts = json.loads(src.tags().get("counts", "{}"))
            arrs.update({n: data[i] for i, n in enumerate(names)})
            print(f"  {path.name}: on grid OK, {len(names)} bands, NaN {100 * np.isnan(data).mean():.2f}%")
        print(f"  {kind} counts: {counts}")
    for qa in ["s2_n_clear", "s1_n_obs"]:
        a = arrs[qa]
        print(f"  {qa}: min {a.min():.0f}, median {np.median(a):.0f}, max {a.max():.0f}")
    ndvi, vv = arrs["s2_NDVI_p50"], arrs["s1_vv_db_median"]
    print(f"  median s2_NDVI_p50 = {np.nanmedian(ndvi):.3f}, median s1_vv_db_median = {np.nanmedian(vv):.2f} dB")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5.5), constrained_layout=True)
    ext = [grid.minx, grid.maxx, grid.miny, grid.maxy]
    for a, data, title, cmap, lim in [(ax[0], ndvi, "S2 NDVI p50", "YlGn", (0, 1)),
                                      (ax[1], vv, "S1 VV dB median", "gray", (-20, 0))]:
        im = a.imshow(data, extent=ext, cmap=cmap, vmin=lim[0], vmax=lim[1])
        a.set_title(f"{site} {year}: {title}")
        fig.colorbar(im, ax=a, shrink=0.8)
    out = results_dir() / "figures" / f"qa_pc_{site}_{year}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  quick-look: {out}")


# --- main ------------------------------------------------------------------
def outputs_exist(kind, site, year) -> bool:
    return all(raw_path(kind, site, year, s).exists() for s in ["", "_qa"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site")
    ap.add_argument("--year", type=int)
    ap.add_argument("--only", choices=["s1", "s2"], help="process only one sensor")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="only print QA numbers / quick-looks")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config()
    todo = [(s, y) for s, y in site_years()
            if (args.site is None or s == args.site) and (args.year is None or y == args.year)]
    if not todo:
        sys.exit(f"no site-year matches --site {args.site} --year {args.year}")

    if not args.report_only:
        odc.stac.configure_rio(cloud_defaults=True, GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2",
                               GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_CONNECTTIMEOUT="20")
        # Items stay unsigned here; load_item signs a fresh copy right before every read.
        catalog = pystac_client.Client.open(STAC_URL)
        for site, year in todo:
            grid = get_grid(site)
            for kind, fn in [("s2", run_s2), ("s1", run_s1)]:
                if args.only and kind != args.only:
                    continue
                if outputs_exist(kind, site, year) and not args.overwrite:
                    log.info("[%s %s %d] outputs exist, skipping (use --overwrite)", kind.upper(), site, year)
                    continue
                t0 = time.time()
                fn(catalog, site, year, grid, cfg)
                log.info("[%s %s %d] took %.1f min", kind.upper(), site, year, (time.time() - t0) / 60)

    for site, year in todo:
        if all(outputs_exist(k, site, year) for k in ["s2", "s1"]):
            report(site, year)


if __name__ == "__main__":
    main()
