"""E4 temporal transfer (AEF) with nuisance-shift corrections estimated on height-stable training cells.

Per fold and direction a -> b (model trained on year a training blocks, tested on year b test blocks):
  - stable cells = cells present in both years, inside the fold's training blocks (same buffer as E4),
    with |height_b - height_a| <= stable_m
  - shift d = aef_mean(b) - aef_mean(a) on those cells; mu = mean(d); dims ranked by mu^2
  variants:
    baseline  no correction (reproduces hgt.experiments E4)
    offset    subtract mu from the year-b aef_*_mean columns before predicting
    offset_all  same, but mu is the mean shift over ALL paired training-block cells (no heights used,
              so it needs no target-year lidar)
    drop-k    drop the k top-ranked dims (aef_Axx_mean AND aef_Axx_std) from train and test
    both-k    drop-k, then subtract mu on the remaining dims
Only training-block cells are used to estimate mu and the ranking; test blocks are never touched.
Same model, partition, buffer and seeds as hgt.experiments.run_e4 (AEF, config learner).

Usage:
  uv run python scripts/e4_shift_correction.py [--site tanzania] [--stable-m 1.0] [--ks 8 20 32]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from hgt.config import load_config
from hgt.experiments import E4_SITES, cell_subset, load_table, log, make_model
from hgt.io import experiment_dir
from hgt.splits import buffer_mask, e4_partition, feature_columns, fold_masks

FS = "AEF"


def estimate_shift(df_a, df_b, tr_a, tr_b, mean_cols, stable_m):
    """mu = mean(e_b - e_a) over cells in both years' training sets, restricted to |dh| <= stable_m
    (stable_m=None: all paired cells, no heights used)."""
    a = df_a.loc[tr_a, ["cell_id", "height"] + mean_cols].set_index("cell_id")
    b = df_b.loc[tr_b, ["cell_id", "height"] + mean_cols].set_index("cell_id")
    ids = a.index.intersection(b.index)
    a, b = a.loc[ids], b.loc[ids]
    stable = (np.ones(len(ids), bool) if stable_m is None
              else (b["height"] - a["height"]).abs().to_numpy() <= stable_m)
    d = b[mean_cols].to_numpy()[stable] - a[mean_cols].to_numpy()[stable]
    return d.mean(0), int(stable.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", default="tanzania")
    p.add_argument("--stable-m", type=float, default=1.0)
    p.add_argument("--ks", type=int, nargs="+", default=[8, 20, 32])
    args = p.parse_args()

    cfg = load_config()["model"]
    n_folds, seed, n_est = cfg["n_folds"], cfg["seed"], cfg["n_estimators"]
    site = args.site
    years = E4_SITES[site]
    dfs_full = {y: load_table(site, y) for y in years}
    dfs = {y: cell_subset(d) for y, d in dfs_full.items()}
    cols = feature_columns(dfs[years[0]], FS)
    mean_cols = [c for c in cols if c.endswith("_mean")]
    dim_of = lambda c: c.split("_")[1]
    dims = [dim_of(c) for c in mean_cols]
    mean_idx = {c: cols.index(c) for c in mean_cols}
    folds = e4_partition(dfs_full[years[0]], dfs_full[years[1]], 0)

    X = {y: dfs[y][cols].to_numpy(np.float32) for y in years}
    y_ = {y: dfs[y]["height"].to_numpy(np.float64) for y in years}
    variants = ["baseline", "offset", "offset_all"] + [f"drop{k}" for k in args.ks] + [f"both{k}" for k in args.ks]
    pooled = {}   # (train_year, test_year, variant) -> ([obs], [pred])
    n_stable, n_paired = [], []

    for f in range(n_folds):
        masks = {}
        for y in years:
            tr, te = fold_masks(dfs[y]["block_id"], folds, f)
            masks[y] = (tr & ~buffer_mask(dfs[y], folds, f, site), te)
        for a in years:
            for b in years:
                if a == b:
                    continue
                mu, n_s = estimate_shift(dfs[a], dfs[b], masks[a][0], masks[b][0], mean_cols, args.stable_m)
                n_stable.append(n_s)
                mu_all, n_all = estimate_shift(dfs[a], dfs[b], masks[a][0], masks[b][0], mean_cols, None)
                n_paired.append(n_all)
                rank = np.argsort(-mu ** 2)
                tr_a, te_a, te_b = masks[a][0], masks[a][1], masks[b][1]
                models = {}   # dropped dims (frozenset) -> (fitted model, kept column indices)
                for v in variants:
                    k = int(v.replace("drop", "").replace("both", "")) if v[:4] in ("drop", "both") else 0
                    drop = frozenset(dims[i] for i in rank[:k])
                    if drop not in models:
                        keep = [i for i, c in enumerate(cols) if dim_of(c) not in drop]
                        models[drop] = (make_model(n_est, seed).fit(X[a][tr_a][:, keep], y_[a][tr_a]), keep)
                    model, keep = models[drop]
                    xb = X[b][te_b].copy()
                    if v in ("offset", "offset_all") or v.startswith("both"):
                        shift = mu_all if v == "offset_all" else mu
                        for j, c in enumerate(mean_cols):
                            xb[:, mean_idx[c]] -= shift[j]
                    for ta, tb, x_te, y_te in ((a, a, X[a][te_a], y_[a][te_a]), (a, b, xb, y_[b][te_b])):
                        o, q = pooled.setdefault((ta, tb, v), ([], []))
                        o.append(y_te)
                        q.append(model.predict(x_te[:, keep]))
        log(f"fold {f} done")

    print(f"\n{site} stable cells per (fold, direction): min {min(n_stable)}, median {int(np.median(n_stable))}; "
          f"all paired: min {min(n_paired)}, median {int(np.median(n_paired))}")
    rows = []
    for (a, b, v), (o, q) in pooled.items():
        rows.append(dict(train=a, test=b, variant=v, r2=r2_score(np.concatenate(o), np.concatenate(q))))
    df = pd.DataFrame(rows)
    out = df.pivot_table(index="variant", columns=["train", "test"], values="r2").reindex(variants)
    print(out.round(3).to_string())
    out_path = experiment_dir() / f"e4_shift_correction_{site}.csv"
    df.to_csv(out_path, index=False)
    print("saved", out_path)


if __name__ == "__main__":
    main()
