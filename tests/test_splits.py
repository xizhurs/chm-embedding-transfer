"""Spatial block splits, buffer and few-shot sampling (leakage-critical code)."""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from hgt.grid import get_grid
from hgt.splits import (META_COLS, assign_block_folds, buffer_mask, e3_shot_blocks, e4_partition,
                        feature_columns, fold_blocks, fold_masks)


def site_cells(site="tanzania"):
    """All cells of a real site grid (geometry only, no data)."""
    return get_grid(site).cell_table()


def test_assign_block_folds_deterministic_and_balanced():
    rng = np.random.default_rng(0)
    ids = np.arange(40)
    counts = rng.integers(1, 101, size=40)
    a = assign_block_folds(ids, 5, 7, counts)
    assert a == assign_block_folds(ids, 5, 7, counts)
    assert set(a) == set(ids) and set(a.values()) == set(range(5))
    sizes = np.bincount([a[b] for b in ids], weights=counts, minlength=5)
    assert sizes.max() - sizes.min() <= counts.max()     # greedy balance bound
    assert a != assign_block_folds(ids, 5, 8, counts)    # another seed, another partition


def test_assign_block_folds_needs_enough_blocks():
    with pytest.raises(AssertionError):
        assign_block_folds(np.arange(3), 5, 0, np.ones(3))


def test_fold_masks_partition_rows():
    df = site_cells()
    folds = assign_block_folds(np.unique(df["block_id"]), 5, 1, df.groupby("block_id").size().to_numpy())
    seen = np.zeros(len(df), dtype=int)
    for f in range(5):
        tr, te = fold_masks(df["block_id"], folds, f)
        assert not (tr & te).any() and (tr | te).all()
        assert not set(df["block_id"][tr]) & set(df["block_id"][te])
        seen += te
    assert (seen == 1).all()                               # every row tested exactly once


def test_feature_columns_select_by_prefix():
    df = pd.DataFrame(columns=list(META_COLS) + ["hc_a", "hc_b", "aef_A00_mean", "tes_E000_mean", "other"])
    assert feature_columns(df, "HC") == ["hc_a", "hc_b"]
    assert feature_columns(df, "AEF") == ["aef_A00_mean"]
    assert feature_columns(df, "TES") == ["tes_E000_mean"]
    assert feature_columns(df, "HC+TES") == ["hc_a", "hc_b", "tes_E000_mean"]
    for fs in ("HC", "AEF", "TES", "HC+AEF", "HC+TES"):
        assert not set(feature_columns(df, fs)) & set(META_COLS)


@pytest.mark.parametrize("n", [1, 2])
def test_buffer_mask_keeps_training_cells_away_from_test(n):
    site = "tanzania"
    df = site_cells(site)
    folds = assign_block_folds(np.unique(df["block_id"]), 5, 3, df.groupby("block_id").size().to_numpy())
    for f in range(5):
        tr, te = fold_masks(df["block_id"], folds, f)
        buf = buffer_mask(df, folds, f, site, n=n)
        assert not (buf & te).any()                        # buffer never touches test cells
        train = tr & ~buf
        xy_tr = df.loc[train, ["row", "col"]].to_numpy()
        xy_te = df.loc[te, ["row", "col"]].to_numpy()
        dist, _ = cKDTree(xy_te).query(xy_tr, k=1, p=np.inf)
        assert dist.min() > n                              # Chebyshev distance > buffer
        # the buffer is exactly the ring of cells within n of the test blocks
        d_buf, _ = cKDTree(xy_te).query(df.loc[buf, ["row", "col"]].to_numpy(), k=1, p=np.inf)
        assert (d_buf <= n).all()


def test_buffer_mask_zero_is_empty():
    df = site_cells()
    folds = assign_block_folds(np.unique(df["block_id"]), 5, 3, df.groupby("block_id").size().to_numpy())
    assert not buffer_mask(df, folds, 0, "tanzania", n=0).any()


def test_e3_shot_blocks_come_from_training_blocks():
    df = site_cells()
    folds = assign_block_folds(np.unique(df["block_id"]), 5, 3, df.groupby("block_id").size().to_numpy())
    for f in range(5):
        train_blocks, test_blocks = fold_blocks(folds, f)
        for k in (1, 5, 20):
            shots = e3_shot_blocks(train_blocks, f, k, 0)
            assert shots == e3_shot_blocks(train_blocks, f, k, 0)
            assert len(set(shots)) == k and set(shots) <= train_blocks and not set(shots) & test_blocks


def test_e4_partition_independent_of_year_order():
    df = site_cells()
    a = df.sample(frac=0.6, random_state=1)[["cell_id", "block_id"]]
    b = df.sample(frac=0.6, random_state=2)[["cell_id", "block_id"]]
    p = e4_partition(a, b, 0)
    assert p == e4_partition(b, a, 0)
    assert set(p) == set(a["block_id"]) | set(b["block_id"])
