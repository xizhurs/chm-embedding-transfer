# Canopy height (P95, 100 m): handcrafted S1/S2 vs AlphaEarth and Tessera embeddings

TabICLv2 (TabICLRegressor, n_estimators = {ti}, CPU; no tuning) on 1 km spatial block splits. FS: HC = 98 handcrafted S1/S2 features (incl. 24 GLCM), AEF = 128 AlphaEarth features (64 mean + 64 std), TES = 256 Tessera features (128 mean + 128 std), HC+AEF / HC+TES = combinations. Metrics: R², RMSE (m), rRMSE = RMSE / mean(obs), bias = mean(pred − obs). Figures in `results/tabicl/figures/`.

**Pipeline.** Cells: only cells with Tessera data (`covered`), identical for every feature set; block partitions are built on all cells first, so folds match earlier all-cell runs. Feature selection: features whose mean |SHAP| is above the mean over all features (threshold = mean) of a selector XGBoost (same hyper-parameters), fitted inside every training fold (SHAP on at most 3000 training rows); the final XGBoost is refitted on the selected features. The number of selected features therefore varies by feature set and fold (see the selection table).

## Key results

- E1 Colombia 2021: R² HC 0.43, AEF 0.61; AEF − HC = 0.185.
- E1 Colombia 2022: R² HC 0.54, AEF 0.55; AEF − HC = 0.015.
- E1 Tanzania 2022: R² HC 0.68, AEF 0.76; AEF − HC = 0.087.
- E1 Tanzania 2024: R² HC 0.62, AEF 0.73; AEF − HC = 0.102.
- E2 (Tanzania 2022 → Colombia 2022) fails for every FS: R² HC -1.26, AEF -4.53; mean bias HC -7.8 m, AEF -13.9 m.
- E3 at k = 20 blocks: tz+k R² HC 0.45, AEF 0.46 (tz+k only; k-only not run).
- E4 Colombia 2021 → 2022 (overlap): temporal loss ΔR² HC 0.086, AEF 0.227.
- E4 Colombia 2021 → 2022 (full): temporal loss ΔR² HC 0.083, AEF 0.329.
- E4 Colombia 2022 → 2021: temporal loss ΔR² HC 0.006, AEF 0.225.
- E4 Colombia 2021 → 2022 (matched): temporal loss ΔR² HC 0.042, AEF 0.236.
- E4 Colombia 2022 → 2021 (matched): temporal loss ΔR² HC -0.000, AEF 0.183.
- E4 Tanzania 2022 → 2024: temporal loss ΔR² HC 0.064, AEF 0.330.
- E4 Tanzania 2024 → 2022: temporal loss ΔR² HC 0.066, AEF 0.305.

## Run settings

Latest run per experiment (from `results/run_log.csv`; n_draws only for E3).

| experiment | time | n_estimators | n_repeats | n_draws | quick |
|---|---|---|---|---|---|
| E1 | 2026-10-03 09:07:57 | 200 | 1 | -1 | False |
| E2 | 2026-10-03 09:12:13 | 200 | 1 | -1 | False |
| E4 | 2026-10-03 10:14:35 | 200 | 1 | -1 | False |
| E4matched | 2026-10-03 10:28:09 | 200 | 1 | -1 | False |
| E3 | 2026-10-03 11:02:21 | 200 | 1 | 1 | False |

## E1: in-domain blocked CV

Pooled out-of-fold metrics per repeat, mean ± sd over 1 repeats (5 folds of 1 km blocks, folds balanced by cell count).

| site-year | n cells | FS | R² | RMSE (m) | rRMSE | bias (m) |
|---|---|---|---|---|---|---|
| Colombia 2021 | 1745 | HC | 0.426 | 4.82 | 0.287 | -0.20 |
| Colombia 2021 | 1745 | AEF | 0.611 | 3.97 | 0.236 | -0.14 |
| Colombia 2022 | 6323 | HC | 0.538 | 4.41 | 0.250 | -0.04 |
| Colombia 2022 | 6323 | AEF | 0.553 | 4.33 | 0.246 | 0.08 |
| Tanzania 2022 | 2427 | HC | 0.676 | 2.25 | 0.307 | -0.07 |
| Tanzania 2022 | 2427 | AEF | 0.763 | 1.93 | 0.262 | -0.06 |
| Tanzania 2024 | 2212 | HC | 0.625 | 2.49 | 0.344 | -0.10 |
| Tanzania 2024 | 2212 | AEF | 0.727 | 2.12 | 0.294 | 0.02 |

Paired difference in pooled R² (same partition within each repeat), mean ± sd over repeats:

| site-year | AEF − HC |
|---|---|
| Colombia 2021 | 0.185 |
| Colombia 2022 | 0.015 |
| Tanzania 2022 | 0.087 |
| Tanzania 2024 | 0.102 |

Spatial buffer (training cells adjacent to the fold's test blocks dropped), mean over folds:

| site-year | training cells / fold (after buffer) | removed by buffer / fold | removed % |
|---|---|---|---|
| Colombia 2021 | 1273 | 123 | 8.8 |
| Colombia 2022 | 4618 | 441 | 8.7 |
| Tanzania 2022 | 1792 | 150 | 7.7 |
| Tanzania 2024 | 1604 | 165 | 9.3 |

Features kept by the in-fold selection, mean (min–max) over folds and repeats:

| site-year | HC | AEF |
|---|---|---|
| Colombia 2021 | 28 (25–31) | 29 (28–31) |
| Colombia 2022 | 27 (23–29) | 29 (26–33) |
| Tanzania 2022 | 16 (14–18) | 29 (28–30) |
| Tanzania 2024 | 20 (17–23) | 25 (23–28) |

Per-fold R² ranges from 0.20 to 0.80; the lowest single fold is 0.20 (Colombia 2021, HC).

## E2: cross-country transfer (train Tanzania 2022 → test Colombia 2022)

One model per FS and repeat fitted on all Tanzania 2022 cells, evaluated on all Colombia 2022 cells (mean ± sd over 1 repeats). Transfer loss = E1 − E2 pooled R² within the same repeat; 'per-fold ΔR²' uses the matched E1 test folds (mean ± sd over repeats × folds).

| FS | E1 R² (CO22 in-domain) | E2 R² (TZ22 → CO22) | transfer loss ΔR² (E1 − E2) | per-fold ΔR² | E1 RMSE | E2 RMSE | ΔRMSE (E2 − E1) | E2 bias (m) |
|---|---|---|---|---|---|---|---|---|
| HC | 0.538 | -1.265 | 1.802 | 1.824 ± 0.263 | 4.41 | 9.76 | 5.35 | -7.85 |
| AEF | 0.553 | -4.534 | 5.088 | 5.148 ± 0.498 | 4.33 | 15.26 | 10.92 | -13.88 |

Target ranges differ strongly: Tanzania 2022 height mean 7.4 m (P95 15.2, max 28.4); Colombia 2022 mean 17.8 m (P95 27.9, max 62.5). 4% of Colombia cells are taller than the tallest Tanzania cell and 66% exceed the Tanzania P95. Tree ensembles (including XGBoost) cannot extrapolate beyond the training target range, so E2 is dominated by a large negative bias; negative R² here means worse than predicting the Colombia mean.

## E3: few-shot recovery in Colombia 2022

Repeat 0 only (the E1 colombia-2022 partition of repeat 0). For each of the 5 folds, k shot blocks are drawn from that fold's training blocks (1 draws per k); models are tested on the fold's test blocks and pooled over the 5 folds. Values: mean ± sd over draws. 'shot cells/fold' = mean number of Colombia cells in the k shot blocks (blocks hold 1–100 cells).

Reference levels (pooled, repeat 0):

| FS | E1 R² (r0) | E2 R² (r0) |
|---|---|---|
| HC | 0.538 | -1.265 |
| AEF | 0.553 | -4.534 |

| FS | k | tz+k R² | tz+k recovery | shot cells/fold |
|---|---|---|---|---|
| HC | 1 | -0.087 | 0.65 | 79 ± nan |
| HC | 5 | 0.316 | 0.88 | 362 ± nan |
| HC | 20 | 0.453 | 0.95 | 1124 ± nan |
| AEF | 1 | 0.191 | 0.93 | 79 ± nan |
| AEF | 5 | 0.371 | 0.96 | 362 ± nan |
| AEF | 20 | 0.464 | 0.98 | 1124 ± nan |

## E4: cross-year transfer

One partition per site over the union of both years' blocks; A→B trains on year A's training-block cells and tests on year B's test-block cells; the baseline B→B trains on year B's training-block cells and tests on the same cells. Temporal loss = (B→B) − (A→B) pooled R² within a repeat; mean ± sd over 1 repeats. Colombia 2021→2022: 'overlap' (main) tests only on 2022 cells that also exist in 2021, 'full' on all 2022 test cells. 'matched' (Colombia, both directions): both years train only on cells present in both years inside the training blocks and are tested on the same shared test cells, so cross-year and baseline differ only in the year (equal training-set size and footprint). Use 'matched' for the Colombia temporal loss; 'overlap'/'full' confound year with the 3–4× larger 2022 training set.

| direction | FS | baseline B→B R² | cross-year A→B R² | temporal loss ΔR² | baseline RMSE | cross RMSE | cross bias (m) | n test |
|---|---|---|---|---|---|---|---|---|
| Colombia 2021 → 2022 (overlap) | HC | 0.573 | 0.487 | 0.086 | 4.18 | 4.58 | -0.37 | 1462 |
| Colombia 2021 → 2022 (overlap) | AEF | 0.602 | 0.375 | 0.227 | 4.03 | 5.05 | -2.55 | 1462 |
| Colombia 2021 → 2022 (full) | HC | 0.535 | 0.452 | 0.083 | 4.42 | 4.80 | -0.36 | 6323 |
| Colombia 2021 → 2022 (full) | AEF | 0.555 | 0.226 | 0.329 | 4.32 | 5.70 | -3.26 | 6323 |
| Colombia 2022 → 2021 | HC | 0.441 | 0.435 | 0.006 | 4.75 | 4.78 | 1.03 | 1745 |
| Colombia 2022 → 2021 | AEF | 0.588 | 0.362 | 0.225 | 4.08 | 5.08 | 2.60 | 1745 |
| Colombia 2021 → 2022 (matched) | HC | 0.523 | 0.481 | 0.042 | 4.41 | 4.60 | -0.80 | 1462 |
| Colombia 2021 → 2022 (matched) | AEF | 0.549 | 0.313 | 0.236 | 4.29 | 5.30 | -2.86 | 1462 |
| Colombia 2022 → 2021 (matched) | HC | 0.476 | 0.476 | -0.000 | 4.62 | 4.62 | 0.09 | 1462 |
| Colombia 2022 → 2021 (matched) | AEF | 0.611 | 0.428 | 0.183 | 3.98 | 4.82 | 1.77 | 1462 |
| Tanzania 2022 → 2024 | HC | 0.621 | 0.556 | 0.064 | 2.50 | 2.71 | -0.80 | 2212 |
| Tanzania 2022 → 2024 | AEF | 0.737 | 0.406 | 0.330 | 2.09 | 3.13 | -1.75 | 2212 |
| Tanzania 2024 → 2022 | HC | 0.670 | 0.605 | 0.066 | 2.27 | 2.49 | 0.53 | 2427 |
| Tanzania 2024 → 2022 | AEF | 0.746 | 0.442 | 0.305 | 2.00 | 2.96 | 1.91 | 2427 |

## Caveats

- **Tessera coverage:** Tessera leaves pixels with fewer than 15 cloud-free observations per year empty; about 21 % (2021) and 18 % (2022) of Colombia cells have no Tessera data. With `pipeline.cells = covered` these cells are excluded for every feature set, so results describe the less cloudy part of the Colombia site; Tanzania is fully covered. Earlier all-cell results (no Tessera, no selection) are archived in `results/archive_xgb_allcells_noselect/`.
- Tessera (TESSERA, Feng et al. 2025, arXiv:2506.20380) was accessed on 2026-10-02 from https://registry.opendata.aws/tessera (dClimate v1.1 store, CC-BY-SA).
- **E3 uses repeat 0 only** (one block partition); its sd is over draws of shot blocks, not over partitions. 1 draws per k were run (config: 3).
- **Recovery is sensitive to the E2 floor.** Where E2 R² is strongly negative (systematic bias from transferring a model trained on much lower Tanzanian canopies), even a weak model trained on a few Colombia blocks removes most of the bias and scores a high 'recovery'. Compare tz+k against k-only, and both against the E1 ceiling, rather than reading recovery alone.
- **Cross-year gaps differ:** 1 year for Colombia (2021→2022) and 2 years for Tanzania (2022→2024). Real canopy change, lidar acquisition differences and image-composite differences are all mixed into the temporal loss.
- **S1 acquisition density differs between site-years** (S1A descending only; Colombia 2021 has only 22 S1A scenes), so the S1 part of HC is noisier in some years. AEF annual embeddings are built from their own sensor stack and are not affected the same way.
- **Spatial buffer:** training cells within 1 cell(s) (100 m) of a fold's test blocks are dropped (E1, E3 shot cells, E4), so no test cell has an adjacent training cell; see the E1 table of removed cells. Spatial autocorrelation beyond 100 m can still make in-domain scores somewhat optimistic.
- **Uneven block sizes** (1–100 cells): folds are balanced by cell count, but the number of blocks per fold differs, and few-shot k blocks contain a variable number of cells (reported as shot cells).
- Colombia 2021 is small (2212 cells, 37 blocks) and lies inside the 2022 footprint; E1 numbers for it are noisier.
- Gain importances are biased toward correlated / continuous features and are descriptive only.
- No hyper-parameter tuning; one fixed XGBoost configuration for all feature sets. Rankings could change with tuning or another learner.

## Attribution

The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind.
