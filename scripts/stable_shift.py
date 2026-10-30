"""Diagnostic 1: how do AEF embeddings shift between years on cells where canopy height did not change?

On height-stable cells any change in the embedding is nuisance (phenology, weather, composite, ...).
Reports, per site:
  - height-change distribution and the stable / changed cell counts
  - size of the shift relative to the within-year spread of the embeddings
  - share of shift energy in the global mean offset, and in the top PCA directions of the centred shift
  - how much of the (centred) shift is explained by the starting embedding and height (block-grouped CV ridge)
  - the same mean offset on changed cells, projected on / outside the stable-cell nuisance subspace

Usage:
  uv run python scripts/stable_shift.py [--site tanzania] [--stable-m 1.0] [--changed-m 3.0]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict

from hgt.config import site_years
from hgt.io import table_path


def load_pair(site, stable_m, changed_m):
    ya, yb = sorted(y for s_, y in site_years() if s_ == site)[:2]
    a, b = (pd.read_parquet(table_path(site, y)) for y in (ya, yb))
    cols = [c for c in a.columns if c.startswith("aef_") and c.endswith("_mean")]
    m = a.merge(b, on="cell_id", suffixes=("_a", "_b"))
    ea, eb = m[[c + "_a" for c in cols]].to_numpy(), m[[c + "_b" for c in cols]].to_numpy()
    ok = np.isfinite(ea).all(1) & np.isfinite(eb).all(1)
    m, ea, eb = m[ok], ea[ok], eb[ok]
    dh = (m.height_b - m.height_a).to_numpy()
    return (ya, yb), m, ea, eb, dh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", default="tanzania")
    p.add_argument("--stable-m", type=float, default=1.0, help="|dh| <= this is stable")
    p.add_argument("--changed-m", type=float, default=3.0, help="|dh| >= this is changed")
    args = p.parse_args()

    (ya, yb), m, ea, eb, dh = load_pair(args.site, args.stable_m, args.changed_m)
    d = eb - ea
    stable, changed = np.abs(dh) <= args.stable_m, np.abs(dh) >= args.changed_m
    print(f"{args.site} {ya}->{yb}: {len(m)} paired cells; stable (|dh|<={args.stable_m} m) {stable.sum()}, "
          f"changed (|dh|>={args.changed_m} m) {changed.sum()}")
    print("dh quantiles (5/25/50/75/95):", np.round(np.quantile(dh, [.05, .25, .5, .75, .95]), 2))

    ds = d[stable]
    mu = ds.mean(0)
    dc = ds - mu
    shift_e = (ds ** 2).sum(1).mean()
    spread_e = ((ea[stable] - ea[stable].mean(0)) ** 2).sum(1).mean()
    print(f"\nstable cells: mean |shift|^2 = {shift_e:.4f}; within-year embedding variance = {spread_e:.4f}")
    print(f"  mean offset |mu|^2 / mean |shift|^2 = {(mu ** 2).sum() / shift_e:.2f}  (share of shift that is one constant vector)")
    cos = (ea[stable] * eb[stable]).sum(1) / np.linalg.norm(ea[stable], axis=1) / np.linalg.norm(eb[stable], axis=1)
    print(f"  cosine(e_a, e_b): median {np.median(cos):.3f}, 5th pct {np.quantile(cos, .05):.3f}")

    # Nuisance subspace of the centred shift
    u, s, vt = np.linalg.svd(dc, full_matrices=False)
    ev = s ** 2 / (s ** 2).sum()
    print("  centred-shift PCA, explained variance of PC1..5:", np.round(ev[:5], 3), " cum5:", round(ev[:5].sum(), 3))

    # Is the centred shift predictable from where the cell starts (embedding / height)?
    groups = m.loc[stable, "block_id_a"].to_numpy()
    cv = GroupKFold(n_splits=5)
    for name, X in [("e_a", ea[stable]), ("height_a", m.loc[stable, "height_a"].to_numpy()[:, None]),
                    ("e_a+height_a", np.c_[ea[stable], m.loc[stable, "height_a"].to_numpy()])]:
        pred = cross_val_predict(Ridge(alpha=1.0), X, dc, groups=groups, cv=cv)
        r2 = 1 - ((dc - pred) ** 2).sum() / (dc ** 2).sum()
        print(f"  block-CV R2 of centred shift from {name:<13}: {r2:.3f}")

    # Changed cells: how much of their shift lies in the stable-cell nuisance subspace?
    dch = d[changed]
    if len(dch) >= 5:
        k = 5
        basis = vt[:k].T                      # 64 x k
        res = dch - mu                        # remove stable mean offset
        proj = res @ basis @ basis.T
        print(f"\nchanged cells ({len(dch)}): mean dh {dh[changed].mean():+.2f} m; |shift|^2 {(dch ** 2).sum(1).mean():.4f}")
        print(f"  after removing stable mean offset: {(res ** 2).sum(1).mean():.4f}; "
              f"share in stable top-{k} PCs: {(proj ** 2).sum() / (res ** 2).sum():.2f}")
        sgn = np.sign(dh[changed])
        print("  mean shift of height gainers vs losers, cosine:",
              round(float(np.dot(res[sgn > 0].mean(0), res[sgn < 0].mean(0)) /
                          (np.linalg.norm(res[sgn > 0].mean(0)) * np.linalg.norm(res[sgn < 0].mean(0)))), 3)
              if (sgn > 0).sum() >= 3 and (sgn < 0).sum() >= 3 else "n/a (too few gainers)")


if __name__ == "__main__":
    main()
