"""Spatial block splits shared by every experiment (stage 3).

All train/test splits are made at the level of 1 km blocks (`block_id`), never cells:
  - assign_block_folds: random block -> fold assignment, fold sizes balanced by cell count
  - e1_partition:       per site-year partition for repeat r (seed = cfg seed + r)
  - e2_partition:       the Colombia 2022 test folds of E2 = the E1 colombia-2022 partition
  - e4_partition:       one partition over the UNION of a site's two years' blocks
  - e3_shot_blocks:     few-shot blocks sampled from a fold's TRAINING blocks
  - fold_masks:         train/test row masks for one fold
  - buffer_mask:        rows within `pipeline.buffer_cells` cells of a fold's test blocks
                        (dropped from training so test cells have no adjacent training cells)

The experiments and scripts/check_splits.py both call these functions, so the checks
cover exactly the splits that are used.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from hgt.config import load_config
from hgt.grid import get_grid

META_COLS = ("cell_id", "row", "col", "x", "y", "block_id", "height", "valid_frac")
FEATURE_SETS = ("HC", "AEF", "HC+AEF", "TES", "HC+TES")


def feature_columns(df: pd.DataFrame, fs: str) -> list[str]:
    """Feature columns of a feature set: HC = hc_*, AEF = aef_* (AlphaEarth), TES = tes_* (Tessera),
    and the combinations HC+AEF, HC+TES."""
    groups = {"HC": [c for c in df.columns if c.startswith("hc_")],
              "AEF": [c for c in df.columns if c.startswith("aef_")],
              "TES": [c for c in df.columns if c.startswith("tes_")]}
    cols = [c for part in fs.split("+") for c in groups[part]]
    assert cols and not set(cols) & set(META_COLS)
    return cols


def derive_seed(*parts: int) -> int:
    """Deterministic 32-bit seed from a tuple of non-negative ints."""
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


# --- block -> fold assignment --------------------------------------------------
def block_cell_counts(block_col) -> tuple[np.ndarray, np.ndarray]:
    """(unique block ids, number of rows per block) from a per-row block_id column."""
    ids, counts = np.unique(np.asarray(block_col), return_counts=True)
    return ids, counts


def assign_block_folds(block_ids, n_folds: int, seed: int, cell_counts) -> dict[int, int]:
    """Random block -> fold assignment with folds balanced by cell count.

    The blocks are shuffled with `seed`, then visited in descending size order (stable sort,
    so ties keep the shuffled order) and each is greedily given to the fold with the fewest
    cells so far (ties -> lowest fold index)."""
    block_ids = np.asarray(block_ids)
    cell_counts = np.asarray(cell_counts)
    assert len(block_ids) == len(cell_counts) == len(np.unique(block_ids))
    assert len(block_ids) >= n_folds, "fewer blocks than folds"
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(block_ids))
    order = sorted(order, key=lambda i: -cell_counts[i])     # stable: ties keep shuffle order
    fold_cells = np.zeros(n_folds, dtype=np.int64)
    folds = {}
    for i in order:
        f = int(np.argmin(fold_cells))
        folds[int(block_ids[i])] = f
        fold_cells[f] += cell_counts[i]
    return folds


def fold_masks(block_col, folds: dict[int, int], fold: int) -> tuple[np.ndarray, np.ndarray]:
    """(train_mask, test_mask) row masks: test = rows whose block is in `fold`."""
    fold_of_row = pd.Series(np.asarray(block_col)).map(folds)
    assert not fold_of_row.isna().any(), "block without a fold"
    test = (fold_of_row == fold).to_numpy()
    return ~test, test


def buffer_mask(df: pd.DataFrame, folds: dict[int, int], fold: int, site: str,
                n: int | None = None) -> np.ndarray:
    """Rows of df whose 100 m cell lies within Chebyshev distance n (cells) of any cell of the
    fold's TEST BLOCKS but outside them. Training rows are `train & ~buffer_mask`.
    Defined on block geometry (not on which cells exist), so it is the same for every year."""
    if n is None:
        n = int(load_config()["pipeline"]["buffer_cells"])
    if n <= 0:
        return np.zeros(len(df), dtype=bool)
    grid = get_grid(site)
    nr, nc = grid.cell_shape
    rr, cc = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
    test_blocks = [b for b, f in folds.items() if f == fold]
    in_test = np.isin(grid.block_id(rr, cc), test_blocks)
    near = binary_dilation(in_test, structure=np.ones((3, 3), dtype=bool), iterations=n) & ~in_test
    return near[df["row"].to_numpy(), df["col"].to_numpy()]


def fold_blocks(folds: dict[int, int], fold: int) -> tuple[set[int], set[int]]:
    """(train_blocks, test_blocks) of one fold."""
    test = {b for b, f in folds.items() if f == fold}
    train = {b for b, f in folds.items() if f != fold}
    return train, test


# --- experiment partitions -----------------------------------------------------
def e1_partition(df: pd.DataFrame, repeat: int) -> dict[int, int]:
    """E1 partition of one site-year table for a repeat (seed = cfg seed + repeat)."""
    m = load_config()["model"]
    ids, counts = block_cell_counts(df["block_id"])
    return assign_block_folds(ids, m["n_folds"], m["seed"] + repeat, counts)


def e2_partition(df_colombia_2022: pd.DataFrame, repeat: int) -> dict[int, int]:
    """E2 Colombia 2022 test folds: the same partition as E1 colombia 2022."""
    return e1_partition(df_colombia_2022, repeat)


def e4_partition(df_a: pd.DataFrame, df_b: pd.DataFrame, repeat: int) -> dict[int, int]:
    """E4 partition over the union of two years' blocks of one site.

    Block size = number of distinct cell_ids in the block over both years. The result
    does not depend on the order of the two tables."""
    m = load_config()["model"]
    cells = pd.concat([df_a[["cell_id", "block_id"]], df_b[["cell_id", "block_id"]]])
    cells = cells.drop_duplicates("cell_id")
    assert not cells["cell_id"].duplicated().any()
    ids, counts = block_cell_counts(cells["block_id"])
    return assign_block_folds(ids, m["n_folds"], m["seed"] + repeat, counts)


# --- few-shot sampling ---------------------------------------------------------
def sample_shot_blocks(train_blocks, k: int, rng: np.random.Generator) -> list[int]:
    """k distinct blocks drawn uniformly (without replacement) from the training blocks."""
    pool = np.array(sorted(train_blocks))
    assert k <= len(pool), f"k={k} > {len(pool)} training blocks"
    return [int(b) for b in rng.choice(pool, size=k, replace=False)]


def e3_shot_blocks(train_blocks, fold: int, k: int, draw: int) -> list[int]:
    """E3 shot blocks for (fold, k, draw); rng seeded by (cfg seed, fold, k, draw)."""
    seed = load_config()["model"]["seed"]
    rng = np.random.default_rng([seed, fold, k, draw])
    return sample_shot_blocks(train_blocks, k, rng)
