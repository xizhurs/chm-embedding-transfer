# Canopy height (P95, 100 m): handcrafted S1/S2 vs AlphaEarth and Tessera embeddings

XGBoost (n_estimators = 200, learning_rate = 0.1, max_depth = 6, subsample = 0.8, colsample_bytree = 0.8, no tuning) on 1 km spatial block splits. FS: HC = 98 handcrafted S1/S2 features (incl. 24 GLCM), AEF = 128 AlphaEarth features (64 mean + 64 std), TES = 256 Tessera features (128 mean + 128 std), HC+AEF / HC+TES = combinations. Metrics: R², RMSE (m), rRMSE = RMSE / mean(obs), bias = mean(pred − obs). Figures in `results/figures/`.

**Pipeline.** Cells: only cells with Tessera data (`covered`), identical for every feature set; block partitions are built on all cells first, so folds match earlier all-cell runs. Feature selection: features whose mean |SHAP| is above the mean over all features (threshold = mean) of a selector XGBoost (same hyper-parameters), fitted inside every training fold (SHAP on at most 3000 training rows); the final XGBoost is refitted on the selected features. The number of selected features therefore varies by feature set and fold (see the selection table).

## Key results

- E1 Colombia 2021: R² HC 0.35, AEF 0.54, HC+AEF 0.58, TES 0.54, HC+TES 0.54; AEF − HC = 0.189, TES − HC = 0.181, TES − AEF = -0.008, HC+AEF − AEF = 0.035, HC+TES − TES = 0.006.
- E1 Colombia 2022: R² HC 0.47, AEF 0.50, HC+AEF 0.55, TES 0.59, HC+TES 0.61; AEF − HC = 0.026, TES − HC = 0.120, TES − AEF = 0.094, HC+AEF − AEF = 0.049, HC+TES − TES = 0.012.
- E1 Tanzania 2022: R² HC 0.65, AEF 0.72, HC+AEF 0.74, TES 0.67, HC+TES 0.70; AEF − HC = 0.073, TES − HC = 0.020, TES − AEF = -0.053, HC+AEF − AEF = 0.020, HC+TES − TES = 0.025.
- E1 Tanzania 2024: R² HC 0.60, AEF 0.69, HC+AEF 0.70, TES 0.65, HC+TES 0.68; AEF − HC = 0.098, TES − HC = 0.053, TES − AEF = -0.046, HC+AEF − AEF = 0.007, HC+TES − TES = 0.031.
- E2 (Tanzania 2022 → Colombia 2022) fails for every FS: R² HC -0.92, AEF -2.98, HC+AEF -1.44, TES -0.90, HC+TES -1.03; mean bias HC -6.8 m, AEF -11.4 m, HC+AEF -8.2 m, TES -6.5 m, HC+TES -7.1 m.
- E3 at k = 20 blocks: tz+k R² HC 0.36, AEF 0.39, HC+AEF 0.43, TES 0.48, HC+TES 0.50; k-only R² HC 0.37, AEF 0.39, HC+AEF 0.44, TES 0.49, HC+TES 0.52. Mean tz+k − k-only ΔR² over all k and FS ranges from -0.054 to 0.405.
- E4 Colombia 2021 → 2022 (overlap): temporal loss ΔR² HC 0.082, AEF 0.100, HC+AEF 0.123, TES 0.128, HC+TES 0.121.
- E4 Colombia 2021 → 2022 (full): temporal loss ΔR² HC 0.105, AEF 0.209, HC+AEF 0.176, TES 0.226, HC+TES 0.198.
- E4 Colombia 2022 → 2021: temporal loss ΔR² HC 0.024, AEF 0.170, HC+AEF 0.118, TES 0.135, HC+TES 0.109.
- E4 Colombia 2021 → 2022 (matched): temporal loss ΔR² HC 0.052, AEF 0.103, HC+AEF 0.079, TES 0.096, HC+TES 0.067.
- E4 Colombia 2022 → 2021 (matched): temporal loss ΔR² HC 0.013, AEF 0.128, HC+AEF 0.110, TES 0.131, HC+TES 0.088.
- E4 Tanzania 2022 → 2024: temporal loss ΔR² HC 0.043, AEF 0.253, HC+AEF 0.166, TES 0.142, HC+TES 0.076.
- E4 Tanzania 2024 → 2022: temporal loss ΔR² HC 0.094, AEF 0.328, HC+AEF 0.293, TES 0.176, HC+TES 0.093.

## Run settings

Latest run per experiment (from `results/run_log.csv`; n_draws only for E3).

| experiment | time | n_estimators | n_repeats | n_draws | quick |
|---|---|---|---|---|---|
| E1 | 2026-10-02 18:02:20 | 200 | 1 | -1 | False |
| E2 | 2026-10-02 18:03:20 | 200 | 1 | -1 | False |
| E4 | 2026-10-02 18:16:28 | 200 | 1 | -1 | False |
| E4matched | 2026-10-02 18:22:12 | 200 | 1 | -1 | False |
| E3 | 2026-10-02 19:44:48 | 200 | 1 | 3 | False |

## E1: in-domain blocked CV

Pooled out-of-fold metrics per repeat, mean ± sd over 1 repeats (5 folds of 1 km blocks, folds balanced by cell count).

| site-year | n cells | FS | R² | RMSE (m) | rRMSE | bias (m) |
|---|---|---|---|---|---|---|
| Colombia 2021 | 1745 | HC | 0.354 | 5.11 | 0.305 | 0.03 |
| Colombia 2021 | 1745 | AEF | 0.543 | 4.30 | 0.256 | -0.01 |
| Colombia 2021 | 1745 | HC+AEF | 0.579 | 4.13 | 0.246 | 0.09 |
| Colombia 2021 | 1745 | TES | 0.535 | 4.34 | 0.258 | 0.24 |
| Colombia 2021 | 1745 | HC+TES | 0.541 | 4.31 | 0.257 | 0.05 |
| Colombia 2022 | 6323 | HC | 0.474 | 4.70 | 0.267 | 0.02 |
| Colombia 2022 | 6323 | AEF | 0.500 | 4.59 | 0.261 | 0.09 |
| Colombia 2022 | 6323 | HC+AEF | 0.549 | 4.36 | 0.247 | 0.12 |
| Colombia 2022 | 6323 | TES | 0.594 | 4.13 | 0.235 | 0.08 |
| Colombia 2022 | 6323 | HC+TES | 0.606 | 4.07 | 0.231 | 0.06 |
| Tanzania 2022 | 2427 | HC | 0.651 | 2.34 | 0.318 | 0.01 |
| Tanzania 2022 | 2427 | AEF | 0.724 | 2.08 | 0.283 | 0.02 |
| Tanzania 2022 | 2427 | HC+AEF | 0.744 | 2.00 | 0.272 | 0.01 |
| Tanzania 2022 | 2427 | TES | 0.671 | 2.27 | 0.309 | 0.00 |
| Tanzania 2022 | 2427 | HC+TES | 0.695 | 2.19 | 0.297 | -0.01 |
| Tanzania 2024 | 2212 | HC | 0.596 | 2.58 | 0.357 | 0.05 |
| Tanzania 2024 | 2212 | AEF | 0.694 | 2.25 | 0.311 | 0.09 |
| Tanzania 2024 | 2212 | HC+AEF | 0.701 | 2.22 | 0.307 | 0.05 |
| Tanzania 2024 | 2212 | TES | 0.649 | 2.41 | 0.333 | 0.05 |
| Tanzania 2024 | 2212 | HC+TES | 0.679 | 2.30 | 0.318 | 0.08 |

Paired difference in pooled R² (same partition within each repeat), mean ± sd over repeats:

| site-year | AEF − HC | TES − HC | TES − AEF | HC+AEF − AEF | HC+TES − TES |
|---|---|---|---|---|---|
| Colombia 2021 | 0.189 | 0.181 | -0.008 | 0.035 | 0.006 |
| Colombia 2022 | 0.026 | 0.120 | 0.094 | 0.049 | 0.012 |
| Tanzania 2022 | 0.073 | 0.020 | -0.053 | 0.020 | 0.025 |
| Tanzania 2024 | 0.098 | 0.053 | -0.046 | 0.007 | 0.031 |

Features kept by the in-fold selection, mean (min–max) over folds and repeats:

| site-year | HC | AEF | HC+AEF | TES | HC+TES |
|---|---|---|---|---|---|
| Colombia 2021 | 28 (27–31) | 29 (28–31) | 49 (46–54) | 65 (63–70) | 86 (81–90) |
| Colombia 2022 | 27 (23–29) | 29 (26–31) | 53 (50–57) | 70 (66–73) | 90 (87–97) |
| Tanzania 2022 | 17 (16–18) | 28 (27–29) | 44 (38–47) | 56 (52–58) | 68 (64–74) |
| Tanzania 2024 | 19 (17–22) | 28 (25–32) | 46 (43–49) | 54 (50–58) | 68 (62–74) |

Per-fold R² ranges from 0.12 to 0.78; the lowest single fold is 0.12 (Colombia 2021, HC).

## E2: cross-country transfer (train Tanzania 2022 → test Colombia 2022)

One model per FS and repeat fitted on all Tanzania 2022 cells, evaluated on all Colombia 2022 cells (mean ± sd over 1 repeats). Transfer loss = E1 − E2 pooled R² within the same repeat; 'per-fold ΔR²' uses the matched E1 test folds (mean ± sd over repeats × folds).

| FS | E1 R² (CO22 in-domain) | E2 R² (TZ22 → CO22) | transfer loss ΔR² (E1 − E2) | per-fold ΔR² | E1 RMSE | E2 RMSE | ΔRMSE (E2 − E1) | E2 bias (m) |
|---|---|---|---|---|---|---|---|---|
| HC | 0.474 | -0.918 | 1.393 | 1.406 ± 0.191 | 4.70 | 8.98 | 4.28 | -6.84 |
| AEF | 0.500 | -2.983 | 3.483 | 3.530 ± 0.452 | 4.59 | 12.94 | 8.36 | -11.37 |
| HC+AEF | 0.549 | -1.436 | 1.985 | 2.013 ± 0.303 | 4.36 | 10.12 | 5.77 | -8.24 |
| TES | 0.594 | -0.895 | 1.489 | 1.508 ± 0.171 | 4.13 | 8.93 | 4.80 | -6.50 |
| HC+TES | 0.606 | -1.030 | 1.635 | 1.654 ± 0.209 | 4.07 | 9.24 | 5.17 | -7.14 |

Target ranges differ strongly: Tanzania 2022 height mean 7.4 m (P95 15.2, max 28.4); Colombia 2022 mean 17.8 m (P95 27.9, max 62.5). 4% of Colombia cells are taller than the tallest Tanzania cell and 66% exceed the Tanzania P95. Tree ensembles (including XGBoost) cannot extrapolate beyond the training target range, so E2 is dominated by a large negative bias; negative R² here means worse than predicting the Colombia mean.

## E3: few-shot recovery in Colombia 2022

Repeat 0 only (the E1 colombia-2022 partition of repeat 0). For each of the 5 folds, k shot blocks are drawn from that fold's training blocks (3 draws per k); models are tested on the fold's test blocks and pooled over the 5 folds. Values: mean ± sd over draws. 'shot cells/fold' = mean number of Colombia cells in the k shot blocks (blocks hold 1–100 cells).

Reference levels (pooled, repeat 0):

| FS | E1 R² (r0) | E2 R² (r0) |
|---|---|---|
| HC | 0.474 | -0.918 |
| AEF | 0.500 | -2.983 |
| HC+AEF | 0.549 | -1.436 |
| TES | 0.594 | -0.895 |
| HC+TES | 0.606 | -1.030 |

| FS | k | tz+k R² | tz+k recovery | shot cells/fold | k-only R² | k-only recovery | tz+k − k-only ΔR² |
|---|---|---|---|---|---|---|---|
| HC | 1 | 0.003 ± 0.158 | 0.66 ± 0.11 | 82 ± 8 | -0.051 ± 0.224 | 0.62 ± 0.16 | 0.054 ± 0.082 |
| HC | 2 | 0.176 ± 0.070 | 0.79 ± 0.05 | 134 ± 8 | 0.018 ± 0.125 | 0.67 ± 0.09 | 0.158 ± 0.135 |
| HC | 5 | 0.250 ± 0.023 | 0.84 ± 0.02 | 333 ± 62 | 0.235 ± 0.039 | 0.83 ± 0.03 | 0.015 ± 0.028 |
| HC | 10 | 0.307 ± 0.011 | 0.88 ± 0.01 | 645 ± 31 | 0.305 ± 0.070 | 0.88 ± 0.05 | 0.003 ± 0.063 |
| HC | 20 | 0.358 ± 0.014 | 0.92 ± 0.01 | 1305 ± 59 | 0.366 ± 0.015 | 0.92 ± 0.01 | -0.008 ± 0.002 |
| AEF | 1 | -0.004 ± 0.130 | 0.86 ± 0.04 | 82 ± 8 | -0.357 ± 0.364 | 0.75 ± 0.10 | 0.352 ± 0.335 |
| AEF | 2 | 0.184 ± 0.081 | 0.91 ± 0.02 | 134 ± 8 | -0.118 ± 0.408 | 0.82 ± 0.12 | 0.302 ± 0.381 |
| AEF | 5 | 0.269 ± 0.018 | 0.93 ± 0.01 | 333 ± 62 | 0.207 ± 0.035 | 0.92 ± 0.01 | 0.062 ± 0.024 |
| AEF | 10 | 0.340 ± 0.060 | 0.95 ± 0.02 | 645 ± 31 | 0.327 ± 0.054 | 0.95 ± 0.02 | 0.013 ± 0.016 |
| AEF | 20 | 0.388 ± 0.020 | 0.97 ± 0.01 | 1305 ± 59 | 0.389 ± 0.035 | 0.97 ± 0.01 | -0.001 ± 0.018 |
| HC+AEF | 1 | 0.080 ± 0.107 | 0.76 ± 0.05 | 82 ± 8 | -0.325 ± 0.459 | 0.56 ± 0.23 | 0.405 ± 0.385 |
| HC+AEF | 2 | 0.246 ± 0.059 | 0.85 ± 0.03 | 134 ± 8 | -0.063 ± 0.104 | 0.69 ± 0.05 | 0.309 ± 0.150 |
| HC+AEF | 5 | 0.306 ± 0.020 | 0.88 ± 0.01 | 333 ± 62 | 0.241 ± 0.058 | 0.85 ± 0.03 | 0.064 ± 0.038 |
| HC+AEF | 10 | 0.399 ± 0.029 | 0.92 ± 0.01 | 645 ± 31 | 0.375 ± 0.078 | 0.91 ± 0.04 | 0.023 ± 0.050 |
| HC+AEF | 20 | 0.432 ± 0.017 | 0.94 ± 0.01 | 1305 ± 59 | 0.441 ± 0.017 | 0.95 ± 0.01 | -0.009 ± 0.003 |
| TES | 1 | -0.020 ± 0.175 | 0.59 ± 0.12 | 82 ± 8 | 0.034 ± 0.036 | 0.62 ± 0.02 | -0.054 ± 0.140 |
| TES | 2 | 0.127 ± 0.066 | 0.69 ± 0.04 | 134 ± 8 | 0.082 ± 0.058 | 0.66 ± 0.04 | 0.045 ± 0.066 |
| TES | 5 | 0.305 ± 0.049 | 0.81 ± 0.03 | 333 ± 62 | 0.298 ± 0.065 | 0.80 ± 0.04 | 0.007 ± 0.064 |
| TES | 10 | 0.433 ± 0.019 | 0.89 ± 0.01 | 645 ± 31 | 0.415 ± 0.058 | 0.88 ± 0.04 | 0.017 ± 0.039 |
| TES | 20 | 0.481 ± 0.004 | 0.92 ± 0.00 | 1305 ± 59 | 0.491 ± 0.017 | 0.93 ± 0.01 | -0.011 ± 0.017 |
| HC+TES | 1 | 0.048 ± 0.138 | 0.66 ± 0.08 | 82 ± 8 | 0.073 ± 0.070 | 0.67 ± 0.04 | -0.025 ± 0.070 |
| HC+TES | 2 | 0.200 ± 0.068 | 0.75 ± 0.04 | 134 ± 8 | 0.037 ± 0.118 | 0.65 ± 0.07 | 0.163 ± 0.135 |
| HC+TES | 5 | 0.333 ± 0.042 | 0.83 ± 0.03 | 333 ± 62 | 0.287 ± 0.075 | 0.81 ± 0.05 | 0.046 ± 0.037 |
| HC+TES | 10 | 0.448 ± 0.015 | 0.90 ± 0.01 | 645 ± 31 | 0.423 ± 0.055 | 0.89 ± 0.03 | 0.025 ± 0.040 |
| HC+TES | 20 | 0.499 ± 0.008 | 0.93 ± 0.00 | 1305 ± 59 | 0.517 ± 0.005 | 0.95 ± 0.00 | -0.018 ± 0.009 |

## E4: cross-year transfer

One partition per site over the union of both years' blocks; A→B trains on year A's training-block cells and tests on year B's test-block cells; the baseline B→B trains on year B's training-block cells and tests on the same cells. Temporal loss = (B→B) − (A→B) pooled R² within a repeat; mean ± sd over 1 repeats. Colombia 2021→2022: 'overlap' (main) tests only on 2022 cells that also exist in 2021, 'full' on all 2022 test cells. 'matched' (Colombia, both directions): both years train only on cells present in both years inside the training blocks and are tested on the same shared test cells, so cross-year and baseline differ only in the year (equal training-set size and footprint). Use 'matched' for the Colombia temporal loss; 'overlap'/'full' confound year with the 3–4× larger 2022 training set.

| direction | FS | baseline B→B R² | cross-year A→B R² | temporal loss ΔR² | baseline RMSE | cross RMSE | cross bias (m) | n test |
|---|---|---|---|---|---|---|---|---|
| Colombia 2021 → 2022 (overlap) | HC | 0.523 | 0.440 | 0.082 | 4.42 | 4.78 | -0.67 | 1462 |
| Colombia 2021 → 2022 (overlap) | AEF | 0.540 | 0.440 | 0.100 | 4.33 | 4.78 | -1.05 | 1462 |
| Colombia 2021 → 2022 (overlap) | HC+AEF | 0.610 | 0.487 | 0.123 | 3.99 | 4.58 | -0.74 | 1462 |
| Colombia 2021 → 2022 (overlap) | TES | 0.609 | 0.482 | 0.128 | 3.99 | 4.60 | -1.53 | 1462 |
| Colombia 2021 → 2022 (overlap) | HC+TES | 0.632 | 0.511 | 0.121 | 3.88 | 4.47 | -1.44 | 1462 |
| Colombia 2021 → 2022 (full) | HC | 0.475 | 0.371 | 0.105 | 4.70 | 5.14 | -0.15 | 6323 |
| Colombia 2021 → 2022 (full) | AEF | 0.504 | 0.295 | 0.209 | 4.57 | 5.44 | -2.05 | 6323 |
| Colombia 2021 → 2022 (full) | HC+AEF | 0.552 | 0.376 | 0.176 | 4.34 | 5.12 | -1.61 | 6323 |
| Colombia 2021 → 2022 (full) | TES | 0.582 | 0.356 | 0.226 | 4.20 | 5.20 | -1.92 | 6323 |
| Colombia 2021 → 2022 (full) | HC+TES | 0.594 | 0.396 | 0.198 | 4.13 | 5.04 | -1.65 | 6323 |
| Colombia 2022 → 2021 | HC | 0.378 | 0.354 | 0.024 | 5.02 | 5.11 | 0.86 | 1745 |
| Colombia 2022 → 2021 | AEF | 0.540 | 0.369 | 0.170 | 4.32 | 5.05 | 2.07 | 1745 |
| Colombia 2022 → 2021 | HC+AEF | 0.546 | 0.428 | 0.118 | 4.29 | 4.81 | 1.70 | 1745 |
| Colombia 2022 → 2021 | TES | 0.516 | 0.380 | 0.135 | 4.43 | 5.01 | 2.12 | 1745 |
| Colombia 2022 → 2021 | HC+TES | 0.544 | 0.435 | 0.109 | 4.30 | 4.78 | 1.46 | 1745 |
| Colombia 2021 → 2022 (matched) | HC | 0.480 | 0.428 | 0.052 | 4.61 | 4.83 | -0.49 | 1462 |
| Colombia 2021 → 2022 (matched) | AEF | 0.527 | 0.425 | 0.103 | 4.39 | 4.85 | -1.21 | 1462 |
| Colombia 2021 → 2022 (matched) | HC+AEF | 0.574 | 0.495 | 0.079 | 4.17 | 4.54 | -1.12 | 1462 |
| Colombia 2021 → 2022 (matched) | TES | 0.558 | 0.462 | 0.096 | 4.25 | 4.69 | -1.54 | 1462 |
| Colombia 2021 → 2022 (matched) | HC+TES | 0.589 | 0.522 | 0.067 | 4.10 | 4.42 | -1.10 | 1462 |
| Colombia 2022 → 2021 (matched) | HC | 0.392 | 0.380 | 0.013 | 4.97 | 5.02 | 0.44 | 1462 |
| Colombia 2022 → 2021 (matched) | AEF | 0.553 | 0.425 | 0.128 | 4.26 | 4.83 | 1.53 | 1462 |
| Colombia 2022 → 2021 (matched) | HC+AEF | 0.575 | 0.465 | 0.110 | 4.15 | 4.66 | 1.41 | 1462 |
| Colombia 2022 → 2021 (matched) | TES | 0.557 | 0.426 | 0.131 | 4.25 | 4.83 | 1.26 | 1462 |
| Colombia 2022 → 2021 (matched) | HC+TES | 0.578 | 0.490 | 0.088 | 4.14 | 4.55 | 1.13 | 1462 |
| Tanzania 2022 → 2024 | HC | 0.579 | 0.536 | 0.043 | 2.64 | 2.77 | -0.96 | 2212 |
| Tanzania 2022 → 2024 | AEF | 0.683 | 0.430 | 0.253 | 2.29 | 3.07 | -1.26 | 2212 |
| Tanzania 2022 → 2024 | HC+AEF | 0.699 | 0.533 | 0.166 | 2.23 | 2.78 | -1.08 | 2212 |
| Tanzania 2022 → 2024 | TES | 0.620 | 0.479 | 0.142 | 2.51 | 2.93 | 0.59 | 2212 |
| Tanzania 2022 → 2024 | HC+TES | 0.651 | 0.576 | 0.076 | 2.40 | 2.65 | -0.64 | 2212 |
| Tanzania 2024 → 2022 | HC | 0.635 | 0.540 | 0.094 | 2.39 | 2.69 | 0.89 | 2427 |
| Tanzania 2024 → 2022 | AEF | 0.719 | 0.392 | 0.328 | 2.10 | 3.09 | 1.79 | 2427 |
| Tanzania 2024 → 2022 | HC+AEF | 0.733 | 0.440 | 0.293 | 2.05 | 2.96 | 1.72 | 2427 |
| Tanzania 2024 → 2022 | TES | 0.661 | 0.486 | 0.176 | 2.31 | 2.84 | 0.22 | 2427 |
| Tanzania 2024 → 2022 | HC+TES | 0.691 | 0.598 | 0.093 | 2.20 | 2.51 | 0.72 | 2427 |

## Caveats

- **Tessera coverage:** Tessera leaves pixels with fewer than 15 cloud-free observations per year empty; about 21 % (2021) and 18 % (2022) of Colombia cells have no Tessera data. With `pipeline.cells = covered` these cells are excluded for every feature set, so results describe the less cloudy part of the Colombia site; Tanzania is fully covered. Earlier all-cell results (no Tessera, no selection) are archived in `results/archive_xgb_allcells_noselect/`.
- Tessera (TESSERA, Feng et al. 2025, arXiv:2506.20380) was accessed on 2026-10-02 from https://registry.opendata.aws/tessera (dClimate v1.1 store, CC-BY-SA).
- **E3 uses repeat 0 only** (one block partition); its sd is over draws of shot blocks, not over partitions. 3 draws per k were run (config: 3).
- **Recovery is sensitive to the E2 floor.** Where E2 R² is strongly negative (systematic bias from transferring a model trained on much lower Tanzanian canopies), even a weak model trained on a few Colombia blocks removes most of the bias and scores a high 'recovery'. Compare tz+k against k-only, and both against the E1 ceiling, rather than reading recovery alone.
- **Cross-year gaps differ:** 1 year for Colombia (2021→2022) and 2 years for Tanzania (2022→2024). Real canopy change, lidar acquisition differences and image-composite differences are all mixed into the temporal loss.
- **S1 acquisition density differs between site-years** (S1A descending only; Colombia 2021 has only 22 S1A scenes), so the S1 part of HC is noisier in some years. AEF annual embeddings are built from their own sensor stack and are not affected the same way.
- **Spatial autocorrelation across block edges:** test blocks touch training blocks without a buffer, so cells near block edges have correlated neighbours in training. This affects every FS equally but makes all in-domain scores somewhat optimistic.
- **Uneven block sizes** (1–100 cells): folds are balanced by cell count, but the number of blocks per fold differs, and few-shot k blocks contain a variable number of cells (reported as shot cells).
- Colombia 2021 is small (2212 cells, 37 blocks) and lies inside the 2022 footprint; E1 numbers for it are noisier.
- Gain importances are biased toward correlated / continuous features and are descriptive only.
- No hyper-parameter tuning; one fixed XGBoost configuration for all feature sets. Rankings could change with tuning or another learner.

## Attribution

The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind.
