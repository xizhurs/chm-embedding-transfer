"""Site grids: the single source of truth for pixel/cell/block geometry.

Every raster and table in the project must be on these grids:
  - 10 m feature grid  (S1, S2, AEF rasters)
  - 100 m cell grid    (targets, modelling table)
  - 1 km blocks        (spatial CV groups) = 10 x 10 cells
All three share the same upper-left origin, so they nest exactly.

cell_id  = row * n_cell_cols + col          (row 0 = north)
block_id = (row // 10) * n_block_cols + (col // 10)
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from affine import Affine
from pyproj import Transformer

from hgt.config import load_config


@dataclass(frozen=True)
class SiteGrid:
    site: str
    crs: str
    minx: float
    miny: float
    maxx: float
    maxy: float
    feature_res: int = 10
    cell_res: int = 100
    block_size: int = 1000

    # --- transforms / shapes -------------------------------------------------
    def transform(self, res: float) -> Affine:
        return Affine(res, 0.0, self.minx, 0.0, -res, self.maxy)

    @property
    def feature_transform(self) -> Affine:
        return self.transform(self.feature_res)

    @property
    def cell_transform(self) -> Affine:
        return self.transform(self.cell_res)

    @property
    def feature_shape(self) -> tuple[int, int]:
        return (int((self.maxy - self.miny) // self.feature_res),
                int((self.maxx - self.minx) // self.feature_res))

    @property
    def cell_shape(self) -> tuple[int, int]:
        return (int((self.maxy - self.miny) // self.cell_res),
                int((self.maxx - self.minx) // self.cell_res))

    @property
    def px_per_cell(self) -> int:
        return self.cell_res // self.feature_res

    @property
    def cells_per_block(self) -> int:
        return self.block_size // self.cell_res

    @property
    def n_block_cols(self) -> int:
        return self.cell_shape[1] // self.cells_per_block

    # --- ids -----------------------------------------------------------------
    def cell_id(self, row, col):
        return np.asarray(row) * self.cell_shape[1] + np.asarray(col)

    def block_id(self, row, col):
        cpb = self.cells_per_block
        return (np.asarray(row) // cpb) * self.n_block_cols + (np.asarray(col) // cpb)

    def cell_table(self) -> pd.DataFrame:
        """One row per 100 m cell: cell_id, row, col, x, y (cell centre), block_id."""
        nr, nc = self.cell_shape
        row, col = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
        row, col = row.ravel(), col.ravel()
        return pd.DataFrame({
            "cell_id": self.cell_id(row, col),
            "row": row,
            "col": col,
            "x": self.minx + (col + 0.5) * self.cell_res,
            "y": self.maxy - (row + 0.5) * self.cell_res,
            "block_id": self.block_id(row, col),
        })

    # --- helpers -------------------------------------------------------------
    def bbox_wgs84(self, buffer_m: float = 0.0) -> tuple[float, float, float, float]:
        """(west, south, east, north) in EPSG:4326, optionally buffered."""
        t = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        xs = np.linspace(self.minx - buffer_m, self.maxx + buffer_m, 21)
        ys = np.linspace(self.miny - buffer_m, self.maxy + buffer_m, 21)
        X, Y = np.meshgrid(xs, ys)
        lon, lat = t.transform(X.ravel(), Y.ravel())
        return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.minx, self.miny, self.maxx, self.maxy)


@lru_cache
def get_grid(site: str) -> SiteGrid:
    cfg = load_config()
    s, g = cfg["sites"][site], cfg["grid"]
    minx, miny, maxx, maxy = s["grid_bounds"]
    grid = SiteGrid(site, s["crs"], minx, miny, maxx, maxy,
                    g["feature_res"], g["cell_res"], g["block_size"])
    for v in grid.bounds:
        assert v % grid.block_size == 0, f"{site}: grid bounds must be multiples of block_size"
    return grid
