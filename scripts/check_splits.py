"""Assert-based leakage checks over every split configuration used by hgt.experiments.

Checks, for the full config (all repeats, folds, k and draws):
  - E1: train/test blocks (and cells) disjoint for every (site-year, repeat, fold);
        every block in exactly one test fold per repeat
  - E2: Colombia 2022 test folds == E1 colombia 2022 test folds, per repeat
  - E3: shot blocks are a subset of the fold's training blocks and never test blocks
  - E4: one partition per site and repeat, identical for both years (and independent of year order);
        train/test blocks disjoint across years; every union block in exactly one test fold
  - buffer: after hgt.splits.buffer_mask, every training cell (E1, E3 shot cells, E4 cross-year) is
        more than `pipeline.buffer_cells` cells (Chebyshev distance, independent KD-tree check) from
        every test cell

Usage:
  uv run python scripts/check_splits.py
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from hgt.config import load_config, site_years
from hgt.io import experiment_dir
from hgt.experiments import E2_TEST, E4_SITES, load_table
from hgt.splits import (buffer_mask, e1_partition, e2_partition, e3_shot_blocks, e4_partition,
                        fold_blocks, fold_masks)

cfg = load_config()
N_FOLDS = cfg["model"]["n_folds"]
K_BLOCKS = cfg["fewshot"]["k_blocks"]
# Cover every repeat / draw in the config AND in results/metrics.csv (runs may use CLI overrides).
N_REPEATS, N_DRAWS = cfg["model"]["n_repeats"], cfg["fewshot"]["n_draws"]
_metrics = experiment_dir() / "metrics.csv"
if _metrics.exists():
    _m = pd.read_csv(_metrics, usecols=["repeat", "draw"])
    N_REPEATS = max(N_REPEATS, int(_m["repeat"].max()) + 1)
    N_DRAWS = max(N_DRAWS, int(_m["draw"].max()) + 1)


def check_partition(df, folds, label):
    """Disjoint train/test per fold; every block of df in exactly one test fold."""
    blocks = set(df["block_id"].unique())
    assert blocks <= set(folds), f"{label}: blocks without a fold"
    assert set(folds.values()) == set(range(N_FOLDS)), f"{label}: empty fold"
    seen = []
    for f in range(N_FOLDS):
        train_blocks, test_blocks = fold_blocks(folds, f)
        assert not train_blocks & test_blocks, f"{label} fold {f}: train/test blocks overlap"
        tr, te = fold_masks(df["block_id"], folds, f)
        assert not (tr & te).any() and (tr | te).all(), f"{label} fold {f}: bad row masks"
        assert not set(df["block_id"][tr]) & set(df["block_id"][te]), f"{label} fold {f}: block leak"
        assert not set(df["cell_id"][tr]) & set(df["cell_id"][te]), f"{label} fold {f}: cell leak"
        seen.extend(test_blocks)
    assert len(seen) == len(set(seen)) == len(folds), f"{label}: block not in exactly one test fold"


BUFFER = int(cfg["pipeline"]["buffer_cells"])


def check_buffer(d_tr, tr, d_te, te, label):
    """Min Chebyshev distance (in cells) between training and test cells must exceed BUFFER."""
    if BUFFER <= 0 or not tr.any() or not te.any():
        return
    xy_tr = d_tr.loc[tr, ["row", "col"]].to_numpy()
    xy_te = d_te.loc[te, ["row", "col"]].to_numpy()
    dist, _ = cKDTree(xy_te).query(xy_tr, k=1, p=np.inf)
    assert dist.min() > BUFFER, f"{label}: training cell {dist.min():.0f} cell(s) from a test cell"


def fold_sizes(df, folds):
    return [int(fold_masks(df["block_id"], folds, f)[1].sum()) for f in range(N_FOLDS)]


def main():
    tables = {sy: load_table(*sy) for sy in site_years()}

    # E1
    for (site, year), df in tables.items():
        for r in range(N_REPEATS):
            folds = e1_partition(df, r)
            assert folds == e1_partition(df, r), "E1 partition not deterministic"
            check_partition(df, folds, f"E1 {site} {year} r{r}")
            for f in range(N_FOLDS):
                tr, te = fold_masks(df["block_id"], folds, f)
                check_buffer(df, tr & ~buffer_mask(df, folds, f, site), df, te, f"E1 {site} {year} r{r} f{f}")
        print(f"E1 {site} {year}: {df['block_id'].nunique()} blocks, "
              f"r0 test-fold cells {fold_sizes(df, e1_partition(df, 0))}")

    # E2: same Colombia 2022 test folds as E1
    co = tables[E2_TEST]
    for r in range(N_REPEATS):
        e1, e2 = e1_partition(co, r), e2_partition(co, r)
        assert e1 == e2, f"E2 r{r}: folds differ from E1"
        for f in range(N_FOLDS):
            assert fold_blocks(e1, f)[1] == fold_blocks(e2, f)[1]
            assert (fold_masks(co["block_id"], e1, f)[1] == fold_masks(co["block_id"], e2, f)[1]).all()
    print("E2: Colombia 2022 test folds == E1 folds for all repeats")

    # E3: repeat 0, every (fold, k, draw)
    folds = e1_partition(co, 0)
    n_shot_cells = {}
    for f in range(N_FOLDS):
        train_blocks, test_blocks = fold_blocks(folds, f)
        _, te = fold_masks(co["block_id"], folds, f)
        for k in K_BLOCKS:
            for d in range(N_DRAWS):
                shots = e3_shot_blocks(train_blocks, f, k, d)
                assert shots == e3_shot_blocks(train_blocks, f, k, d), "E3 sampling not deterministic"
                assert len(shots) == len(set(shots)) == k, f"E3 f{f} k{k} d{d}: wrong number of blocks"
                assert set(shots) <= train_blocks, f"E3 f{f} k{k} d{d}: shot block not in training blocks"
                assert not set(shots) & test_blocks, f"E3 f{f} k{k} d{d}: shot block in test blocks"
                shot = co["block_id"].isin(shots).to_numpy() & ~buffer_mask(co, folds, f, E2_TEST[0])
                assert not (shot & te).any(), f"E3 f{f} k{k} d{d}: shot cell in test cells"
                check_buffer(co, shot, co, te, f"E3 f{f} k{k} d{d}")
                n_shot_cells.setdefault(k, []).append(int(shot.sum()))
    for k, n in n_shot_cells.items():
        print(f"E3 k={k}: shot cells min/median/max = {min(n)}/{int(np.median(n))}/{max(n)}")

    # E4: one partition over the union of both years' blocks
    for site, (ya, yb) in E4_SITES.items():
        dfa, dfb = tables[(site, ya)], tables[(site, yb)]
        union = set(dfa["block_id"]) | set(dfb["block_id"])
        for r in range(N_REPEATS):
            folds = e4_partition(dfa, dfb, r)
            assert folds == e4_partition(dfb, dfa, r), f"E4 {site} r{r}: partition depends on year order"
            assert set(folds) == union, f"E4 {site} r{r}: partition is not over the union of blocks"
            for df, yr in ((dfa, ya), (dfb, yb)):
                check_partition(df, folds, f"E4 {site} {yr} r{r}")
            seen = []
            for f in range(N_FOLDS):
                train_blocks, test_blocks = fold_blocks(folds, f)
                assert not train_blocks & test_blocks
                seen.extend(test_blocks)
                # cross-year: A's training blocks never contain B's test cells
                for d_tr, d_te in ((dfa, dfb), (dfb, dfa)):
                    tr = fold_masks(d_tr["block_id"], folds, f)[0]
                    te = fold_masks(d_te["block_id"], folds, f)[1]
                    assert not set(d_tr["block_id"][tr]) & set(d_te["block_id"][te])
                    assert not set(d_tr["cell_id"][tr]) & set(d_te["cell_id"][te])
                    check_buffer(d_tr, tr & ~buffer_mask(d_tr, folds, f, site), d_te, te, f"E4 {site} r{r} f{f}")
            assert len(seen) == len(set(seen)) == len(union), f"E4 {site} r{r}: test folds"
        for r in range(N_REPEATS):
            folds = e4_partition(dfa, dfb, r)
            print(f"E4 {site} r{r}: {len(union)} union blocks, test-fold cells "
                  f"{ya}={fold_sizes(dfa, folds)} {yb}={fold_sizes(dfb, folds)}")

    print(f"buffer: all training cells > {BUFFER} cell(s) from test cells (E1, E3, E4)")
    print("ALL SPLIT CHECKS PASSED")


if __name__ == "__main__":
    main()
