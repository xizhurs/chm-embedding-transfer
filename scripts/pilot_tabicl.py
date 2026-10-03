"""Timing / memory / accuracy pilot: TabICL vs XGBoost on one E1 fold, before any full TabICL run.

Uses exactly the experiment pipeline (covered cells, E1 partition repeat 0, 1-cell buffer,
in-fold mean-SHAP selection with an XGBoost selector) and only swaps the final learner.
TabICL uses the memory-saving settings of config `tabicl` (batch_size, disk offload).
Prints fit+predict seconds, peak process memory and R2 per (site-year, feature set, learner),
and an estimate of the full TabICL runtime from the number of fits per experiment.
A TabICL fit is skipped if less than MIN_FREE_GB of RAM is available.

Usage:
  uv run python scripts/pilot_tabicl.py
"""
import threading
import time

import numpy as np
import psutil
from sklearn.metrics import r2_score

from hgt.config import load_config
from hgt.experiments import ShapSelectModel, cell_subset, load_table, make_tabicl, make_xgb, xy
from hgt.splits import buffer_mask, e1_partition, fold_masks

CASES = [("tanzania", 2022), ("colombia", 2022)]   # smaller case first
FEATURE_SETS = ["AEF", "HC+TES"]                    # fewest / most selected features
TABICL_SIZES = [4, 8]                               # TabICL n_estimators to try (cheaper first)
MIN_FREE_GB = 3.0
# fits per experiment for 5 FS, 1 repeat, 3 draws (E3 = 5 folds x 5 k x 3 draws x 2 variants)
N_FITS = {"E1": 4 * 5 * 5, "E2": 5, "E4": 2 * 5 * 2 * 5, "E4matched": 5 * 2 * 5, "E3": 5 * 5 * 3 * 2 * 5}


class PeakRSS:
    """Samples this process's resident memory every 0.2 s; .peak_gb after stop()."""

    def __init__(self):
        self.proc, self.peak, self._run = psutil.Process(), 0, True
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while self._run:
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(0.2)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._run = False
        self._t.join()

    @property
    def peak_gb(self) -> float:
        return self.peak / 1024 ** 3


def main() -> None:
    cfg = load_config()
    sel = cfg["pipeline"]["feature_selection"]
    n_trees, seed = cfg["model"]["n_estimators"], cfg["model"]["seed"]
    print(f"TabICL settings: {cfg['tabicl']}", flush=True)
    times = {}
    for site, year in CASES:
        full = load_table(site, year)
        df = cell_subset(full)
        folds = e1_partition(full, 0)
        tr, te = fold_masks(df["block_id"], folds, 0)
        tr &= ~buffer_mask(df, folds, 0, site)
        for fs in FEATURE_SETS:
            X, y = xy(df, fs)
            learners = {"xgboost": lambda: make_xgb(n_trees, seed)}
            learners.update({f"tabicl-{n}": (lambda n=n: make_tabicl(n, seed)) for n in TABICL_SIZES})
            for name, final in learners.items():
                free_gb = psutil.virtual_memory().available / 1024 ** 3
                if name.startswith("tabicl") and free_gb < MIN_FREE_GB:
                    print(f"{site} {year} {fs:7s} {name:9s}: SKIPPED, only {free_gb:.1f} GB RAM free", flush=True)
                    continue
                model = ShapSelectModel(n_trees, seed, "mean", int(sel["k"]), int(sel["max_rows"]), final)
                with PeakRSS() as mem:
                    t0 = time.time()
                    model.fit(X[tr], y[tr])
                    t_fit = time.time() - t0
                    pred = model.predict(X[te])
                    t_all = time.time() - t0
                times.setdefault(name, []).append(t_all)
                print(f"{site} {year} {fs:7s} {name:9s}: n_train {int(tr.sum())}, n_test {int(te.sum())}, "
                      f"features {len(model.selected_)}/{X.shape[1]}, fit {t_fit:6.1f} s, "
                      f"fit+predict {t_all:6.1f} s, peak RSS {mem.peak_gb:4.1f} GB, "
                      f"free before {free_gb:4.1f} GB, R2 {r2_score(y[te], pred):.3f}", flush=True)

    print("\nEstimated full-run time per learner (mean fit+predict time x number of fits):")
    for name, ts in times.items():
        per_fit = float(np.mean(ts))
        est = {exp: per_fit * n / 3600 for exp, n in N_FITS.items()}
        print(f"  {name:9s}: {per_fit:5.1f} s/fit -> " +
              ", ".join(f"{exp} {h:.1f} h" for exp, h in est.items()) + f", total {sum(est.values()):.1f} h")


if __name__ == "__main__":
    main()
