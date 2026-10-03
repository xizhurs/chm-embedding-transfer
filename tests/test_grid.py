"""Site grid geometry: 10 m pixels, 100 m cells and 1 km blocks must nest exactly."""
import numpy as np
import pytest

from hgt.config import site_years
from hgt.grid import get_grid

SITES = sorted({s for s, _ in site_years()})


@pytest.mark.parametrize("site", SITES)
def test_grid_nesting(site):
    g = get_grid(site)
    for v in g.bounds:
        assert v % g.block_size == 0
    assert g.px_per_cell == 10 and g.cells_per_block == 10
    assert g.feature_shape == (g.cell_shape[0] * 10, g.cell_shape[1] * 10)
    assert g.cell_shape[0] % 10 == 0 and g.cell_shape[1] % 10 == 0


@pytest.mark.parametrize("site", SITES)
def test_cell_table_ids_and_centres(site):
    g = get_grid(site)
    t = g.cell_table()
    nr, nc = g.cell_shape
    assert len(t) == nr * nc
    assert t["cell_id"].is_unique
    assert t["block_id"].nunique() == (nr // 10) * (nc // 10)
    # every block holds exactly 10 x 10 cells
    assert (t.groupby("block_id").size() == 100).all()
    # cell centres agree with the affine transform and lie inside the grid
    x, y = g.cell_transform @ (t["col"].to_numpy() + 0.5, t["row"].to_numpy() + 0.5)
    np.testing.assert_allclose(x, t["x"])
    np.testing.assert_allclose(y, t["y"])
    assert t["x"].between(g.minx, g.maxx).all() and t["y"].between(g.miny, g.maxy).all()


@pytest.mark.parametrize("site", SITES)
def test_block_id_matches_cell_table(site):
    g = get_grid(site)
    t = g.cell_table()
    np.testing.assert_array_equal(g.block_id(t["row"], t["col"]), t["block_id"])
    np.testing.assert_array_equal(g.cell_id(t["row"], t["col"]), t["cell_id"])


@pytest.mark.parametrize("site", SITES)
def test_bbox_wgs84_is_plausible(site):
    w, s, e, n = get_grid(site).bbox_wgs84()
    assert -180 < w < e < 180 and -90 < s < n < 90
    assert e - w < 0.5 and n - s < 0.5          # a site is a few km across
    w2, s2, e2, n2 = get_grid(site).bbox_wgs84(buffer_m=200)
    assert w2 < w and s2 < s and e2 > e and n2 > n
