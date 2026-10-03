"""Raster writers must put data exactly on the site grids."""
import numpy as np
import pandas as pd
import pytest
import rasterio

from hgt.grid import get_grid
from hgt.inference import to_grid
from hgt.io import check_on_grid, write_cell_raster, write_feature_raster


def test_write_cell_raster_roundtrip(tmp_path):
    g = get_grid("tanzania")
    data = np.random.default_rng(0).random((2, *g.cell_shape)).astype("float32")
    path = tmp_path / "map.tif"
    write_cell_raster(path, data, ["lidar_p95", "pred_HC"], g)
    with rasterio.open(path) as r:
        assert r.crs.to_string() == g.crs
        assert r.transform.almost_equals(g.cell_transform)
        assert (r.height, r.width) == g.cell_shape
        assert list(r.descriptions) == ["lidar_p95", "pred_HC"]
        np.testing.assert_array_equal(r.read(), data)


def test_write_cell_raster_rejects_wrong_shape(tmp_path):
    g = get_grid("tanzania")
    with pytest.raises(AssertionError):
        write_cell_raster(tmp_path / "bad.tif", np.zeros((1, 3, 3)), ["x"], g)


def test_feature_raster_passes_grid_check_and_cell_raster_does_not(tmp_path):
    g = get_grid("tanzania")
    fpath, cpath = tmp_path / "feat.tif", tmp_path / "cell.tif"
    write_feature_raster(fpath, np.zeros((1, *g.feature_shape)), ["s2_B02_median"], g)
    check_on_grid(fpath, g)
    write_cell_raster(cpath, np.zeros((1, *g.cell_shape)), ["pred_HC"], g)
    with pytest.raises(AssertionError):
        check_on_grid(cpath, g)


def test_to_grid_places_values():
    g = get_grid("tanzania")
    cells = pd.DataFrame({"row": [0, 5], "col": [1, 2]})
    img = to_grid(g, cells, np.array([3.0, 7.0]))
    assert img.shape == g.cell_shape
    assert img[0, 1] == 3.0 and img[5, 2] == 7.0 and np.isnan(img).sum() == img.size - 2
