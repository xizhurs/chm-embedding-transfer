"""Stage 3: experiments (XGBoost or TabICL, config model.learner), handcrafted S1/S2 features (HC) vs AlphaEarth embeddings (AEF).

  E1  in-domain blocked CV, per site-year
  E2  cross-country: train all Tanzania 2022 -> test all Colombia 2022
  E2rev  the reverse: train all Colombia 2022 -> test all Tanzania 2022 (not part of `--exp all`)
  E3  few-shot recovery in Colombia 2022: Tanzania 2022 + k Colombia blocks (tz+k) and k blocks only (k-only)
  E4  cross-year within a site, one block partition shared by both years
      (E4matched: Colombia only, both years trained/tested on the same shared cells)

All splits come from hgt.splits (checked by scripts/check_splits.py). The model is the same
everywhere: XGBRegressor(hist, hyper-parameters from config `model`, n_jobs=-1), no tuning.

Outputs (results/):
  metrics.csv                 long format, one row per metric; rerunning an experiment replaces its rows
  predictions_E1.parquet      OOF predictions (cell_id, site, year, obs, pred, fs, repeat, fold)
  predictions_E2.parquet      Colombia 2022 predictions of the Tanzania model
  importances_E1.csv          top-20 gain importances per site-year / FS (repeat 0, mean over folds)

Usage:
  uv run python -m hgt.experiments --exp E1|E2|E2rev|E3|E4|E4matched|all [--quick] [--n-draws N] [--repeats N]
"""
import argparse
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import DMatrix, XGBRegressor

from hgt.config import load_config, site_years
from hgt.io import experiment_dir, table_path
from hgt.splits import (FEATURE_SETS, buffer_mask, derive_seed, e1_partition, e2_partition,
                        e3_shot_blocks, e4_partition, feature_columns, fold_blocks, fold_masks)

E2_TRAIN = ("tanzania", 2022)
E2_TEST = ("colombia", 2022)
E4_SITES = {"colombia": (2021, 2022), "tanzania": (2022, 2024)}
E3_MAX_HOURS = 2.5          # reduce draws to 5 if the estimated E3 runtime exceeds this
ACTIVE_FS = list(FEATURE_SETS)   # feature sets run in this call (--fs); results of the others are kept
E3_VARIANTS = ["tz+k", "k-only"]  # E3 training variants run in this call (--e3-variants)
E3_K = None                       # E3 k values (--e3-k); None = config fewshot.k_blocks
METRIC_COLS = ["experiment", "train", "test", "feature_set", "repeat", "fold", "k", "draw",
               "variant", "metric", "value"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_table(site: str, year: int) -> pd.DataFrame:
    """Full feature table (all cells). Block partitions are always built from this table."""
    return pd.read_parquet(table_path(site, year))


def cell_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Cells used for training/evaluation (config pipeline.cells):
    'covered' = only cells with Tessera data, so every feature set is scored on identical cells;
    'all' = every cell. Applied AFTER the partition is built, so folds match the all-cell runs."""
    if load_config()["pipeline"]["cells"] == "all":
        return df
    tes = [c for c in df.columns if c.startswith("tes_")]
    return df[np.isfinite(df[tes].to_numpy()).all(axis=1)].reset_index(drop=True)


def xy(df: pd.DataFrame, fs: str) -> tuple[np.ndarray, np.ndarray]:
    return df[feature_columns(df, fs)].to_numpy(np.float32), df["height"].to_numpy(np.float64)


def make_xgb(n_estimators: int, seed: int) -> XGBRegressor:
    m = load_config()["model"]
    return XGBRegressor(n_estimators=n_estimators, learning_rate=m["learning_rate"],
                        max_depth=m["max_depth"], subsample=m["subsample"],
                        colsample_bytree=m["colsample_bytree"], tree_method="hist",
                        n_jobs=-1, random_state=seed)


class ShapSelectModel:
    """Feature selection + XGBoost, fitted on the training data only (so it runs inside every fold).

    1. fit a selector XGBoost (same hyper-parameters and seed) on the training data
    2. score features by mean |SHAP| (XGBoost's native TreeSHAP, pred_contribs) on at most
       `max_rows` training rows
    3. keep features with a score above the mean score (rule "mean", like SelectFromModel
       threshold="mean"), or the top k (rule "topk"), and fit the final model (XGBoost or
       TabICL, from `final`) on them. The selector is always XGBoost, so every learner sees
       the same features.
    feature_importances_ (full input length, 0 for unselected) exists only if the final model has it."""

    def __init__(self, n_estimators: int, seed: int, rule: str, k: int, max_rows: int, final):
        self.n_estimators, self.seed, self.rule, self.k, self.max_rows = n_estimators, seed, rule, k, max_rows
        self.final = final

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ShapTopKXGB":
        selector = make_xgb(self.n_estimators, self.seed).fit(X, y)
        rows = np.random.default_rng(self.seed).permutation(len(X))[:self.max_rows]
        contribs = selector.get_booster().predict(DMatrix(X[rows]), pred_contribs=True)[:, :-1]  # drop bias
        shap = np.abs(contribs).mean(axis=0)
        if self.rule == "mean":
            self.selected_ = np.flatnonzero(shap > shap.mean())
        else:
            self.selected_ = np.sort(np.argsort(-shap, kind="stable")[:min(self.k, X.shape[1])])
        self.shap_ = shap
        self.model_ = self.final().fit(X[:, self.selected_], y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X[:, self.selected_])

    def __getattr__(self, attr):
        # feature_importances_ only when the final model provides it (XGBoost yes, TabICL no)
        if attr == "feature_importances_" and hasattr(self.__dict__.get("model_"), "feature_importances_"):
            imp = np.zeros(len(self.shap_), dtype="float32")
            imp[self.selected_] = self.model_.feature_importances_
            return imp
        raise AttributeError(attr)


class PLSR:
    """PLS regression (sklearn) with a 1-D predict, as a linear baseline learner.
    Unlike tree ensembles it can extrapolate beyond the training target range."""

    def __init__(self, n_components: int, scale: bool):
        from sklearn.cross_decomposition import PLSRegression
        self.n_components = n_components
        self.model_ = PLSRegression(n_components=n_components, scale=scale)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSR":
        # n_components cannot exceed the number of features or samples
        self.model_.n_components = min(self.n_components, X.shape[1], len(X) - 1)
        self.model_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(np.ravel(self.model_.predict(X)), 0.0)   # heights cannot be negative


def make_tabicl(n_estimators: int, seed: int):
    """TabICLRegressor on CPU with the memory-saving settings of config `tabicl`:
    batch_size (ensemble members processed at once) and offload_mode / disk_offload_dir
    (column-embedding tensor memory-mapped to disk instead of held in RAM)."""
    import os
    from tabicl import TabICLRegressor
    t = load_config()["tabicl"]
    offload_dir = os.path.expandvars(t["disk_offload_dir"]) if t.get("offload_mode") == "disk" else None
    if offload_dir:
        os.makedirs(offload_dir, exist_ok=True)
    return TabICLRegressor(n_estimators=n_estimators, batch_size=int(t["batch_size"]),
                           offload_mode=t["offload_mode"], disk_offload_dir=offload_dir,
                           device="cpu", random_state=seed)


def make_final(n_estimators: int, seed: int):
    """Final regressor of the configured learner (config model.learner)."""
    cfg = load_config()
    learner = cfg["model"]["learner"]
    if learner == "xgboost":
        return make_xgb(n_estimators, seed)
    if learner == "plsr":
        return PLSR(int(cfg["plsr"]["n_components"]), bool(cfg["plsr"]["scale"]))
    if learner == "tabicl":
        return make_tabicl(int(cfg["tabicl"]["n_estimators"]), seed)
    raise ValueError(f"unknown learner {learner}")


def make_model(n_estimators: int, seed: int):
    """The model used everywhere: the configured learner, optionally behind in-fold SHAP feature
    selection (config pipeline.feature_selection; the selector is always XGBoost)."""
    sel = load_config()["pipeline"]["feature_selection"]
    if sel["method"] == "none":
        return make_final(n_estimators, seed)
    rule = {"shap_mean": "mean", "shap_topk": "topk"}[sel["method"]]
    return ShapSelectModel(n_estimators, seed, rule, int(sel["k"]), int(sel["max_rows"]),
                           final=lambda: make_final(n_estimators, seed))


def metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(obs, pred)))
    return {
        "r2": float(r2_score(obs, pred)),
        "rmse": rmse,
        "rrmse": rmse / float(np.mean(obs)),
        "mae": float(mean_absolute_error(obs, pred)),
        "bias": float(np.mean(pred - obs)),
        "n": float(len(obs)),
    }


def add_metrics(rows: list, obs, pred, **keys) -> None:
    """Append one long-format row per metric; keys = experiment, train, test, feature_set, ..."""
    base = {"repeat": -1, "fold": -1, "k": -1, "draw": -1, "variant": "main", **keys}
    for name, value in metrics(np.asarray(obs), np.asarray(pred)).items():
        rows.append({**base, "metric": name, "value": value})


def write_metrics(exp: str, rows: list, variants: list[str] | None = None) -> None:
    """Replace experiment `exp`'s rows of the active feature sets in results/metrics.csv
    (only `variants` of it, if given); rows of other feature sets are kept."""
    path = experiment_dir() / "metrics.csv"
    new = pd.DataFrame(rows, columns=METRIC_COLS)
    if path.exists():
        old = pd.read_csv(path)
        drop = (old["experiment"] == exp) & old["feature_set"].isin(ACTIVE_FS)
        if variants is not None:
            drop &= old["variant"].isin(variants)
        new = pd.concat([old[~drop], new], ignore_index=True)
    new.to_csv(path, index=False)
    log(f"{exp}: wrote {len(rows)} metric rows -> {path}")


def write_merged(df: pd.DataFrame, path, fs_col: str) -> None:
    """Write a predictions/importances table, keeping existing rows of non-active feature sets."""
    if path.exists():
        old = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        df = pd.concat([old[~old[fs_col].isin(ACTIVE_FS)], df], ignore_index=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def selection_label() -> str:
    sel = load_config()["pipeline"]["feature_selection"]
    return {"none": "none", "shap_mean": "shap_mean"}.get(sel["method"], f'{sel["method"]}(k={sel["k"]})')


def log_run(exp: str, **params) -> None:
    """Append the settings of a run to results/run_log.csv (provenance for the report)."""
    path = experiment_dir() / "run_log.csv"
    row = pd.DataFrame([{"time": time.strftime("%Y-%m-%d %H:%M:%S"), "experiment": exp, **params}])
    if path.exists():
        row = pd.concat([pd.read_csv(path), row], ignore_index=True)
    row.to_csv(path, index=False)


def name(site: str, year: int) -> str:
    return f"{site}_{year}"


# --- E1 ------------------------------------------------------------------------
def run_e1(n_estimators: int, n_repeats: int) -> None:
    n_folds, seed = load_config()["model"]["n_folds"], load_config()["model"]["seed"]
    rows, preds, imps = [], [], []
    for site, year in site_years():
        df_full = load_table(site, year)
        df = cell_subset(df_full)
        sy = name(site, year)
        for r in range(n_repeats):
            folds = e1_partition(df_full, r)
            for fs in ACTIVE_FS:
                t0 = time.time()
                X, y = xy(df, fs)
                oof = np.full(len(df), np.nan)
                fold_of = np.full(len(df), -1)
                fold_imps = []
                for f in range(n_folds):
                    tr, te = fold_masks(df["block_id"], folds, f)
                    buf = tr & buffer_mask(df, folds, f, site)
                    tr = tr & ~buf
                    model = make_model(n_estimators, seed + r).fit(X[tr], y[tr])
                    oof[te] = model.predict(X[te])
                    fold_of[te] = f
                    add_metrics(rows, y[te], oof[te], experiment="E1", train=sy, test=sy,
                                feature_set=fs, repeat=r, fold=f)
                    keys = {"experiment": "E1", "train": sy, "test": sy, "feature_set": fs, "repeat": r,
                            "fold": f, "k": -1, "draw": -1, "variant": "main"}
                    rows.append({**keys, "metric": "n_buffer_removed", "value": float(buf.sum())})
                    rows.append({**keys, "metric": "n_train", "value": float(tr.sum())})
                    if hasattr(model, "selected_"):     # in-fold feature selection: record how many were kept
                        rows.append({**keys, "metric": "n_selected", "value": float(len(model.selected_))})
                    if r == 0 and hasattr(model, "feature_importances_"):
                        fold_imps.append(model.feature_importances_)
                    del model
                assert not np.isnan(oof).any()
                add_metrics(rows, y, oof, experiment="E1", train=sy, test=sy,
                            feature_set=fs, repeat=r, fold=-1)
                preds.append(pd.DataFrame({
                    "cell_id": df["cell_id"].to_numpy(), "site": site, "year": year,
                    "obs": y, "pred": oof, "fs": fs, "repeat": r, "fold": fold_of}))
                if r == 0 and fold_imps:
                    imp = pd.Series(np.mean(fold_imps, axis=0), index=feature_columns(df, fs))
                    top = imp.sort_values(ascending=False).head(20)
                    imps.append(pd.DataFrame({"site_year": sy, "feature_set": fs,
                                              "rank": np.arange(1, len(top) + 1),
                                              "feature": top.index, "importance": top.to_numpy()}))
                log(f"E1 {sy} r{r} {fs}: pooled R2={r2_score(y, oof):.3f} ({time.time() - t0:.0f}s)")
    write_metrics("E1", rows)
    write_merged(pd.concat(preds), experiment_dir() / "predictions_E1.parquet", "fs")
    if imps:
        write_merged(pd.concat(imps), experiment_dir() / "importances_E1.csv", "feature_set")


# --- E2 ------------------------------------------------------------------------
def run_e2(n_estimators: int, n_repeats: int) -> None:
    n_folds, seed = load_config()["model"]["n_folds"], load_config()["model"]["seed"]
    co_full = load_table(*E2_TEST)
    tz, co = cell_subset(load_table(*E2_TRAIN)), cell_subset(co_full)
    tr_name, te_name = name(*E2_TRAIN), name(*E2_TEST)
    rows, preds = [], []
    for r in range(n_repeats):
        folds = e2_partition(co_full, r)
        for fs in ACTIVE_FS:
            X_tz, y_tz = xy(tz, fs)
            X_co, y_co = xy(co, fs)
            model = make_model(n_estimators, seed + r).fit(X_tz, y_tz)
            pred = model.predict(X_co)
            del model
            fold_of = np.full(len(co), -1)
            add_metrics(rows, y_co, pred, experiment="E2", train=tr_name, test=te_name,
                        feature_set=fs, repeat=r, fold=-1)
            for f in range(n_folds):
                _, te = fold_masks(co["block_id"], folds, f)
                fold_of[te] = f
                add_metrics(rows, y_co[te], pred[te], experiment="E2", train=tr_name,
                            test=te_name, feature_set=fs, repeat=r, fold=f)
            preds.append(pd.DataFrame({
                "cell_id": co["cell_id"].to_numpy(), "site": E2_TEST[0], "year": E2_TEST[1],
                "obs": y_co, "pred": pred, "fs": fs, "repeat": r, "fold": fold_of}))
            log(f"E2 r{r} {fs}: R2={r2_score(y_co, pred):.3f}")
    write_metrics("E2", rows)
    write_merged(pd.concat(preds), experiment_dir() / "predictions_E2.parquet", "fs")


# --- E2rev ---------------------------------------------------------------------
def run_e2rev(n_estimators: int, n_repeats: int) -> None:
    """E2 in the other direction: train all Colombia 2022 -> test all Tanzania 2022. Pooled only."""
    seed = load_config()["model"]["seed"]
    co, tz = cell_subset(load_table(*E2_TEST)), cell_subset(load_table(*E2_TRAIN))
    tr_name, te_name = name(*E2_TEST), name(*E2_TRAIN)
    rows, preds = [], []
    for r in range(n_repeats):
        for fs in ACTIVE_FS:
            X_co, y_co = xy(co, fs)
            X_tz, y_tz = xy(tz, fs)
            pred = make_model(n_estimators, seed + r).fit(X_co, y_co).predict(X_tz)
            add_metrics(rows, y_tz, pred, experiment="E2rev", train=tr_name, test=te_name,
                        feature_set=fs, repeat=r, fold=-1)
            preds.append(pd.DataFrame({
                "cell_id": tz["cell_id"].to_numpy(), "site": E2_TRAIN[0], "year": E2_TRAIN[1],
                "obs": y_tz, "pred": pred, "fs": fs, "repeat": r, "fold": -1}))
            log(f"E2rev r{r} {fs}: R2={r2_score(y_tz, pred):.3f} bias={np.mean(pred - y_tz):.2f} m")
    write_metrics("E2rev", rows)
    write_merged(pd.concat(preds), experiment_dir() / "predictions_E2rev.parquet", "fs")


# --- E3 ------------------------------------------------------------------------
def e3_fit(X_tz, y_tz, X_co, y_co, train_mask, test_mask, variant, n_estimators, rf_seed):
    """Fit one E3 model and return predictions on the test rows."""
    if variant == "tz+k":
        X = np.vstack([X_tz, X_co[train_mask]])
        y = np.concatenate([y_tz, y_co[train_mask]])
    else:
        X, y = X_co[train_mask], y_co[train_mask]
    model = make_model(n_estimators, rf_seed).fit(X, y)
    return model.predict(X_co[test_mask])


def estimate_e3_hours(tz, co, folds, n_estimators, n_draws) -> float:
    """Time one tz+k and one k-only fit (fold 0, k = 10) per FS and extrapolate."""
    k_blocks = load_config()["fewshot"]["k_blocks"]
    train_blocks, _ = fold_blocks(folds, 0)
    shots = e3_shot_blocks(train_blocks, 0, 10, 0)
    shot = co["block_id"].isin(shots).to_numpy() & ~buffer_mask(co, folds, 0, E2_TEST[0])
    _, te = fold_masks(co["block_id"], folds, 0)
    t0 = time.time()
    for fs in ACTIVE_FS:
        X_tz, y_tz = xy(tz, fs)
        X_co, y_co = xy(co, fs)
        for variant in E3_VARIANTS:
            e3_fit(X_tz, y_tz, X_co, y_co, shot, te, variant, n_estimators, 0)
    per_setting = time.time() - t0      # all FS x variants for one (fold, k, draw)
    n_settings = load_config()["model"]["n_folds"] * len(k_blocks) * n_draws
    return per_setting * n_settings / 3600


def run_e3(n_estimators: int, n_draws: int, check_runtime: bool) -> None:
    """Repeat 0 only: Colombia 2022 E1 folds, k shot blocks drawn from each fold's training blocks."""
    cfg = load_config()
    n_folds, seed = cfg["model"]["n_folds"], cfg["model"]["seed"]
    k_blocks = E3_K or cfg["fewshot"]["k_blocks"]
    co_full = load_table(*E2_TEST)
    tz, co = cell_subset(load_table(*E2_TRAIN)), cell_subset(co_full)
    tz_name, co_name = name(*E2_TRAIN), name(*E2_TEST)
    repeat = 0
    folds = e1_partition(co_full, repeat)

    if check_runtime:
        hours = estimate_e3_hours(tz, co, folds, n_estimators, n_draws)
        log(f"E3: estimated runtime with {n_draws} draws = {hours:.2f} h")
        if hours > E3_MAX_HOURS and n_draws > 5:
            n_draws = 5
            log(f"E3: estimate exceeds {E3_MAX_HOURS} h -> reducing draws to {n_draws}")

    data = {fs: (*xy(tz, fs), *xy(co, fs)) for fs in ACTIVE_FS}
    y_co = co["height"].to_numpy(np.float64)
    oof = {}            # (fs, variant, k, draw) -> OOF predictions over all Colombia 2022 cells
    shot_cells = {}     # (k, draw) -> list of shot cell counts per fold
    rows = []
    done_folds = []

    # per-fold checkpoint: an interrupted run resumes after the last completed fold, but only if
    # every setting that affects the results is unchanged
    ckpt = experiment_dir() / "e3_checkpoint.pkl"
    signature = dict(n_estimators=n_estimators, n_draws=n_draws, fs=list(ACTIVE_FS), k_blocks=list(k_blocks),
                     variants=list(E3_VARIANTS),
                     n_folds=n_folds, seed=seed, model=cfg["model"], pipeline=cfg["pipeline"],
                     tabicl=cfg.get("tabicl"), n_co=len(co), n_tz=len(tz))
    if ckpt.exists():
        with open(ckpt, "rb") as fh:
            state = pickle.load(fh)
        if state["signature"] == signature:
            rows, oof, shot_cells, done_folds = state["rows"], state["oof"], state["shot_cells"], state["done_folds"]
            log(f"E3: resuming from checkpoint, folds {done_folds} already done")
        else:
            log("E3: checkpoint settings differ from this run -> ignoring it")

    t_start, n_done = time.time(), 0
    n_total = (n_folds - len(done_folds)) * len(k_blocks) * n_draws
    for f in range(n_folds):
        if f in done_folds:
            continue
        train_blocks, test_blocks = fold_blocks(folds, f)
        _, te = fold_masks(co["block_id"], folds, f)
        buf = buffer_mask(co, folds, f, E2_TEST[0])
        for k in k_blocks:
            for d in range(n_draws):
                shots = e3_shot_blocks(train_blocks, f, k, d)
                assert set(shots) <= train_blocks and not set(shots) & test_blocks
                shot = co["block_id"].isin(shots).to_numpy() & ~buf     # shot cells next to test blocks dropped
                assert not (shot & te).any()
                n_shot = int(shot.sum())
                shot_cells.setdefault((k, d), []).append(n_shot)
                rf_seed = derive_seed(seed, repeat, f, k, d)
                for fs in ACTIVE_FS:
                    X_tz, y_tz, X_co, _ = data[fs]
                    for variant in E3_VARIANTS:
                        pred = e3_fit(X_tz, y_tz, X_co, y_co, shot, te, variant, n_estimators, rf_seed)
                        key = (fs, variant, k, d)
                        oof.setdefault(key, np.full(len(co), np.nan))[te] = pred
                        train = f"{tz_name}+{co_name}" if variant == "tz+k" else co_name
                        keys = dict(experiment="E3", train=train, test=co_name, feature_set=fs,
                                    repeat=repeat, fold=f, k=k, draw=d, variant=variant)
                        add_metrics(rows, y_co[te], pred, **keys)
                        rows.append({**keys, "metric": "n_shot_cells", "value": float(n_shot)})
                n_done += 1
                eta = (time.time() - t_start) / n_done * (n_total - n_done) / 60
                log(f"E3 fold {f} k={k} draw {d}: {n_shot} shot cells ({n_done}/{n_total}, ETA {eta:.0f} min)")
        done_folds.append(f)
        tmp = ckpt.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(dict(signature=signature, rows=rows, oof=oof, shot_cells=shot_cells,
                             done_folds=done_folds), fh)
        tmp.replace(ckpt)
        log(f"E3: checkpoint saved after fold {f}")

    for (fs, variant, k, d), pred in oof.items():
        assert not np.isnan(pred).any()
        train = f"{tz_name}+{co_name}" if variant == "tz+k" else co_name
        keys = dict(experiment="E3", train=train, test=co_name, feature_set=fs,
                    repeat=repeat, fold=-1, k=k, draw=d, variant=variant)
        add_metrics(rows, y_co, pred, **keys)
        rows.append({**keys, "metric": "n_shot_cells", "value": float(np.mean(shot_cells[(k, d)]))})
    write_metrics("E3", rows)
    ckpt.unlink(missing_ok=True)


# --- E4 ------------------------------------------------------------------------
def e4_test_variants(site: str, test_year: int, test_df: pd.DataFrame, shared: set) -> dict:
    """Test-cell subsets for a test year: Colombia 2022 -> overlap (cells also in 2021) + full."""
    if site == "colombia" and test_year == 2022:
        return {"overlap": test_df["cell_id"].isin(shared).to_numpy(),
                "full": np.ones(len(test_df), dtype=bool)}
    return {"main": np.ones(len(test_df), dtype=bool)}


def run_e4(n_estimators: int, n_repeats: int) -> None:
    """Per site and fold: one model per year (that year's training-block cells), each predicting
    both years' test-block cells. A->B = cross-year, B->B = matched in-year baseline."""
    n_folds, seed = load_config()["model"]["n_folds"], load_config()["model"]["seed"]
    rows, skipped = [], []
    for site, years in E4_SITES.items():
        dfs_full = {yr: load_table(site, yr) for yr in years}
        dfs = {yr: cell_subset(d) for yr, d in dfs_full.items()}
        shared = set(dfs[years[0]]["cell_id"]) & set(dfs[years[1]]["cell_id"])
        log(f"E4 {site}: {len(shared)} cells present in both years")
        for r in range(n_repeats):
            folds = e4_partition(dfs_full[years[0]], dfs_full[years[1]], r)
            for fs in ACTIVE_FS:
                data = {yr: xy(dfs[yr], fs) for yr in years}
                pooled = {}         # (train_year, test_year, variant) -> ([obs], [pred])
                for f in range(n_folds):
                    masks = {}
                    for yr in years:
                        tr_yr, te_yr = fold_masks(dfs[yr]["block_id"], folds, f)
                        masks[yr] = (tr_yr & ~buffer_mask(dfs[yr], folds, f, site), te_yr)
                    preds = {}      # (train_year, test_year) -> predictions on test_year's test cells
                    for a in years:
                        tr = masks[a][0]
                        if not tr.any():
                            continue
                        model = make_model(n_estimators, seed + r).fit(data[a][0][tr], data[a][1][tr])
                        for b in years:
                            te = masks[b][1]
                            if te.any():
                                preds[(a, b)] = model.predict(data[b][0][te])
                        del model
                    for a, b in [(years[0], years[1]), (years[1], years[0])]:
                        if (a, b) not in preds or (b, b) not in preds:
                            msg = (f"E4 {site} r{r} {fs} fold {f}: skip {a}->{b} "
                                   f"(n_train {a}={int(masks[a][0].sum())}, "
                                   f"n_train {b}={int(masks[b][0].sum())}, n_test {b}={int(masks[b][1].sum())})")
                            log(msg)
                            skipped.append(msg)
                            continue
                        te = masks[b][1]
                        y_te = data[b][1][te]
                        for variant, sub in e4_test_variants(site, b, dfs[b], shared).items():
                            sub = sub[te]
                            for train_year in (a, b):        # cross-year, then baseline
                                p = preds[(train_year, b)][sub]
                                add_metrics(rows, y_te[sub], p, experiment="E4",
                                            train=name(site, train_year), test=name(site, b),
                                            feature_set=fs, repeat=r, fold=f, variant=variant)
                                o, q = pooled.setdefault((train_year, b, variant), ([], []))
                                o.append(y_te[sub])
                                q.append(p)
                for (a, b, variant), (o, q) in pooled.items():
                    add_metrics(rows, np.concatenate(o), np.concatenate(q), experiment="E4",
                                train=name(site, a), test=name(site, b), feature_set=fs,
                                repeat=r, fold=-1, variant=variant)
                summary = ", ".join(
                    f"{a}->{b}{'' if v == 'main' else ' ' + v} R2="
                    f"{r2_score(np.concatenate(o), np.concatenate(q)):.3f}"
                    for (a, b, v), (o, q) in sorted(pooled.items()))
                log(f"E4 {site} r{r} {fs}: {summary}")
    write_metrics("E4", rows)
    log(f"E4: {len(skipped)} skipped (fold, direction) combinations")


def run_e4_matched(n_estimators: int, n_repeats: int) -> None:
    """Colombia E4 with matched training sets: both years train only on cells present in BOTH years
    (inside the fold's training blocks) and are tested on shared cells in the test blocks, so the
    cross-year model and the in-year baseline differ only in the year. Writes E4 rows, variant "matched"."""
    n_folds, seed = load_config()["model"]["n_folds"], load_config()["model"]["seed"]
    site, years = "colombia", E4_SITES["colombia"]
    dfs_full = {yr: load_table(site, yr) for yr in years}
    dfs = {yr: cell_subset(d) for yr, d in dfs_full.items()}
    shared = set(dfs[years[0]]["cell_id"]) & set(dfs[years[1]]["cell_id"])
    is_shared = {yr: dfs[yr]["cell_id"].isin(shared).to_numpy() for yr in years}
    rows = []
    for r in range(n_repeats):
        folds = e4_partition(dfs_full[years[0]], dfs_full[years[1]], r)
        for fs in ACTIVE_FS:
            data = {yr: xy(dfs[yr], fs) for yr in years}
            pooled = {}
            for f in range(n_folds):
                masks = {}
                for yr in years:
                    tr_yr, te_yr = fold_masks(dfs[yr]["block_id"], folds, f)
                    tr_yr = tr_yr & ~buffer_mask(dfs[yr], folds, f, site)
                    masks[yr] = [tr_yr & is_shared[yr], te_yr & is_shared[yr]]
                # identical training and test cells in both years
                ids = {yr: (set(dfs[yr]["cell_id"][masks[yr][0]]), set(dfs[yr]["cell_id"][masks[yr][1]]))
                       for yr in years}
                assert ids[years[0]] == ids[years[1]], "matched E4: train/test cells differ between years"
                preds = {}
                for a in years:
                    tr = masks[a][0]
                    model = make_model(n_estimators, seed + r).fit(data[a][0][tr], data[a][1][tr])
                    for b in years:
                        preds[(a, b)] = model.predict(data[b][0][masks[b][1]])
                for b in years:
                    y_te = data[b][1][masks[b][1]]
                    for a in years:
                        add_metrics(rows, y_te, preds[(a, b)], experiment="E4", train=name(site, a),
                                    test=name(site, b), feature_set=fs, repeat=r, fold=f, variant="matched")
                        o, q = pooled.setdefault((a, b), ([], []))
                        o.append(y_te)
                        q.append(preds[(a, b)])
            for (a, b), (o, q) in pooled.items():
                add_metrics(rows, np.concatenate(o), np.concatenate(q), experiment="E4", train=name(site, a),
                            test=name(site, b), feature_set=fs, repeat=r, fold=-1, variant="matched")
            summary = ", ".join(f"{a}->{b} R2={r2_score(np.concatenate(o), np.concatenate(q)):.3f}"
                                for (a, b), (o, q) in sorted(pooled.items()))
            log(f"E4 matched {site} r{r} {fs}: {summary} (n train/fold ~{int(masks[years[0]][0].sum())})")
    write_metrics("E4", rows, variants=["matched"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, choices=["E1", "E2", "E2rev", "E3", "E4", "E4matched", "all"])
    ap.add_argument("--quick", action="store_true", help="smoke test: 50 boosting rounds, 1 repeat, 2 draws")
    ap.add_argument("--n-draws", type=int, default=None, help="override fewshot.n_draws for E3")
    ap.add_argument("--repeats", type=int, default=None, help="override model.n_repeats")
    ap.add_argument("--fs", nargs="+", choices=FEATURE_SETS, default=None,
                    help="run only these feature sets (others' results are kept)")
    ap.add_argument("--e3-variants", nargs="+", choices=["tz+k", "k-only"], default=None,
                    help="E3: run only these training variants")
    ap.add_argument("--e3-k", nargs="+", type=int, default=None, help="E3: run only these k values")
    args = ap.parse_args()

    cfg = load_config()
    n_estimators, n_repeats = cfg["model"]["n_estimators"], cfg["model"]["n_repeats"]
    n_draws = cfg["fewshot"]["n_draws"]
    if args.quick:
        n_estimators, n_repeats, n_draws = 50, 1, 2
    if args.n_draws is not None:
        n_draws = args.n_draws
    if args.repeats is not None:
        n_repeats = args.repeats
    if args.fs is not None:
        ACTIVE_FS[:] = args.fs
    if args.e3_variants is not None:
        E3_VARIANTS[:] = args.e3_variants
    if args.e3_k is not None:
        global E3_K
        E3_K = args.e3_k
    check_runtime = not args.quick and args.n_draws is None
    experiment_dir().mkdir(parents=True, exist_ok=True)
    log(f"exp={args.exp} n_estimators={n_estimators} n_repeats={n_repeats} n_draws={n_draws} fs={ACTIVE_FS}")

    exps = ["E1", "E2", "E4", "E4matched", "E3"] if args.exp == "all" else [args.exp]
    for exp in exps:
        t0 = time.time()
        if exp == "E1":
            run_e1(n_estimators, n_repeats)
        elif exp == "E2":
            run_e2(n_estimators, n_repeats)
        elif exp == "E2rev":
            run_e2rev(n_estimators, n_repeats)
        elif exp == "E3":
            run_e3(n_estimators, n_draws, check_runtime)
        elif exp == "E4":
            run_e4(n_estimators, n_repeats)
        elif exp == "E4matched":
            run_e4_matched(n_estimators, n_repeats)
        log(f"{exp} done in {(time.time() - t0) / 60:.1f} min")
        log_run(exp, n_estimators=n_estimators, n_repeats=1 if exp == "E3" else n_repeats,
                n_draws=n_draws if exp == "E3" else -1, quick=args.quick,
                feature_sets="|".join(ACTIVE_FS), cells=load_config()["pipeline"]["cells"],
                learner=load_config()["model"]["learner"],
                buffer_cells=load_config()["pipeline"]["buffer_cells"],
                selection=selection_label())


if __name__ == "__main__":
    main()
