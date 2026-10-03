"""Path conventions and raster read/write helpers that enforce the site grid."""
from pathlib import Path

import numpy as np
import rasterio

from hgt.config import ROOT, load_config
from hgt.grid import SiteGrid, get_grid

RAW_KINDS = ("s1", "s2", "aef", "tessera")


def chm_path(site: str, year: int) -> Path:
    return ROOT / load_config()["sites"][site]["years"][year]


def raw_path(kind: str, site: str, year: int, suffix: str = "") -> Path:
    """data/raw/{kind}/{site}_{year}{suffix}.tif  (suffix e.g. '_qa')."""
    assert kind in RAW_KINDS
    return ROOT / "data" / "raw" / kind / f"{site}_{year}{suffix}.tif"


def target_path(site: str, year: int) -> Path:
    return ROOT / "data" / "interim" / "targets" / f"{site}_{year}.parquet"


def table_path(site: str, year: int) -> Path:
    return ROOT / "data" / "tables" / f"{site}_{year}.parquet"


def map_table_path(site: str, year: int) -> Path:
    """Feature table for ALL 100 m cells of the site grid (inference / mapping)."""
    return ROOT / "data" / "tables" / f"map_{site}_{year}.parquet"


def results_dir() -> Path:
    return ROOT / "results"


def experiment_dir() -> Path:
    """Stage-3 outputs (metrics, predictions, figures, summary) of the configured learner:
    results/{model.learner}/."""
    return results_dir() / load_config()["model"]["learner"]


def write_feature_raster(path: Path, data: np.ndarray, band_names: list[str],
                         grid: SiteGrid, nodata: float = np.nan) -> None:
    """Write a (bands, H, W) float32 stack on the site's 10 m feature grid."""
    data = np.asarray(data, dtype="float32")
    if data.ndim == 2:
        data = data[None]
    assert data.shape[1:] == grid.feature_shape, (data.shape, grid.feature_shape)
    assert len(band_names) == data.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", dtype="float32", count=data.shape[0],
                   height=data.shape[1], width=data.shape[2], crs=grid.crs,
                   transform=grid.feature_transform, nodata=nodata,
                   compress="deflate", predictor=3, tiled=True,
                   blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)


def write_cell_raster(path: Path, data: np.ndarray, band_names: list[str],
                      grid: SiteGrid, nodata: float = np.nan) -> None:
    """Write a (bands, H, W) float32 stack on the site's 100 m cell grid (maps)."""
    data = np.asarray(data, dtype="float32")
    if data.ndim == 2:
        data = data[None]
    assert data.shape[1:] == grid.cell_shape, (data.shape, grid.cell_shape)
    assert len(band_names) == data.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", dtype="float32", count=data.shape[0],
                   height=data.shape[1], width=data.shape[2], crs=grid.crs,
                   transform=grid.cell_transform, nodata=nodata, compress="deflate", predictor=3)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)


def check_on_grid(path: Path, grid: SiteGrid) -> None:
    """Raise if a raster is not exactly on the site's 10 m feature grid."""
    with rasterio.open(path) as src:
        assert src.crs.to_string() == grid.crs, f"{path}: CRS {src.crs} != {grid.crs}"
        assert src.transform.almost_equals(grid.feature_transform), \
            f"{path}: transform {src.transform} != {grid.feature_transform}"
        assert (src.height, src.width) == grid.feature_shape, \
            f"{path}: shape {(src.height, src.width)} != {grid.feature_shape}"
        assert all(src.descriptions), f"{path}: missing band names"


def read_feature_raster(path: Path, site: str) -> tuple[np.ndarray, list[str]]:
    """Read a feature stack, validating it is on the site grid."""
    grid = get_grid(site)
    check_on_grid(path, grid)
    with rasterio.open(path) as src:
        return src.read().astype("float32"), list(src.descriptions)
