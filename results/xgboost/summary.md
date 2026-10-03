# Canopy height (P95, 100 m): handcrafted S1/S2 vs AlphaEarth and Tessera embeddings

XGBoost (n_estimators = {n}, learning_rate = {lr}, max_depth = {md}, subsample = {ss}, colsample_bytree = {cs}, no tuning) on 1 km spatial block splits. FS: HC = 98 handcrafted S1/S2 features (incl. 24 GLCM), AEF = 128 AlphaEarth features (64 mean + 64 std), TES = 256 Tessera features (128 mean + 128 std), HC+AEF / HC+TES = combinations. Metrics: R², RMSE (m), rRMSE = RMSE / mean(obs), bias = mean(pred − obs). Figures in `results/xgboost/figures/`.

**Pipeline.** Cells: only cells with Tessera data (`covered`), identical for every feature set; block partitions are built on all cells first, so folds match earlier all-cell runs. Feature selection: features whose mean |SHAP| is above the mean over all features (threshold = mean) of a selector XGBoost (same hyper-parameters), fitted inside every training fold (SHAP on at most 3000 training rows); the final XGBoost is refitted on the selected features. The number of selected features therefore varies by feature set and fold (see the selection table).

## Key results

- E1 Colombia 2021: R² HC 0.34, AEF 0.53, HC+AEF 0.53, TES 0.50, HC+TES 0.50; AEF − HC = 0.187, TES − HC = 0.160, TES − AEF = -0.027, HC+AEF − AEF = 0.005, HC+TES − TES = 0.002.
- E1 Colombia 2022: R² HC 0.47, AEF 0.49, HC+AEF 0.54, TES 0.58, HC+TES 0.59; AEF − HC = 0.021, TES − HC = 0.114, TES − AEF = 0.092, HC+AEF − AEF = 0.052, HC+TES − TES = 0.008.
- E1 Tanzania 2022: R² HC 0.64, AEF 0.72, HC+AEF 0.73, TES 0.65, HC+TES 0.68; AEF − HC = 0.082, TES − HC = 0.006, TES − AEF = -0.076, HC+AEF − AEF = 0.013, HC+TES − TES = 0.038.
- E1 Tanzania 2024: R² HC 0.58, AEF 0.69, HC+AEF 0.69, TES 0.62, HC+TES 0.65; AEF − HC = 0.106, TES − HC = 0.042, TES − AEF = -0.064, HC+AEF − AEF = 0.000, HC+TES − TES = 0.029.
- E2 (Tanzania 2022 → Colombia 2022) fails for every FS: R² HC -0.92, AEF -2.98, HC+AEF -1.44, TES -0.90, HC+TES -1.03; mean bias HC -6.8 m, AEF -11.4 m, HC+AEF -8.2 m, TES -6.5 m, HC+TES -7.1 m.
- E3 at k = 20 blocks: tz+k R² HC 0.36, AEF 0.38, HC+AEF 0.42, TES 0.48, HC+TES 0.49; k-only R² HC 0.37, AEF 0.38, HC+AEF 0.43, TES 0.48, HC+TES 0.51. Mean tz+k − k-only ΔR² over all k and FS ranges from -0.068 to 0.357.
- E4 Colombia 2021 → 2022 (overlap): temporal loss ΔR² HC 0.099, AEF 0.081, HC+AEF 0.106, TES 0.153, HC+TES 0.120.
- E4 Colombia 2021 → 2022 (full): temporal loss ΔR² HC 0.107, AEF 0.190, HC+AEF 0.183, TES 0.248, HC+TES 0.213.
- E4 Colombia 2022 → 2021: temporal loss ΔR² HC -0.028, AEF 0.164, HC+AEF 0.087, TES 0.111, HC+TES 0.078.
- E4 Colombia 2021 → 2022 (matched): temporal loss ΔR² HC 0.055, AEF 0.076, HC+AEF 0.056, TES 0.081, HC+TES 0.063.
- E4 Colombia 2022 → 2021 (matched): temporal loss ΔR² HC 0.022, AEF 0.150, HC+AEF 0.138, TES 0.110, HC+TES 0.097.
- E4 Tanzania 2022 → 2024: temporal loss ΔR² HC 0.029, AEF 0.264, HC+AEF 0.115, TES 0.154, HC+TES 0.073.
- E4 Tanzania 2024 → 2022: temporal loss ΔR² HC 0.098, AEF 0.376, HC+AEF 0.422, TES 0.157, HC+TES 0.076.

## Run settings

Latest run per experiment (from `results/run_log.csv`; n_draws only for E3).

| experiment | time | n_estimators | n_repeats | n_draws | quick |
|---|---|---|---|---|---|
| E1 | 2026-10-02 20:20:08 | 200 | 1 | -1 | False |
| E2 | 2026-10-02 20:20:55 | 200 | 1 | -1 | False |
| E4 | 2026-10-02 20:34:46 | 200 | 1 | -1 | False |
| E4matched | 2026-10-02 20:40:40 | 200 | 1 | -1 | False |
| E3 | 2026-10-02 23:38:55 | 200 | 1 | 3 | False |

## E1: in-domain blocked CV

Pooled out-of-fold metrics per repeat, mean ± sd over 1 repeats (5 folds of 1 km blocks, folds balanced by cell count).

| site-year | n cells | FS | R² | RMSE (m) | rRMSE | bias (m) |
|---|---|---|---|---|---|---|
| Colombia 2021 | 1745 | HC | 0.341 | 5.16 | 0.308 | -0.04 |
| Colombia 2021 | 1745 | AEF | 0.529 | 4.37 | 0.260 | -0.12 |
| Colombia 2021 | 1745 | HC+AEF | 0.533 | 4.35 | 0.259 | 0.10 |
| Colombia 2021 | 1745 | TES | 0.501 | 4.49 | 0.268 | 0.07 |
| Colombia 2021 | 1745 | HC+TES | 0.504 | 4.48 | 0.267 | 0.05 |
| Colombia 2022 | 6323 | HC | 0.470 | 4.72 | 0.268 | 0.03 |
| Colombia 2022 | 6323 | AEF | 0.491 | 4.63 | 0.263 | 0.14 |
| Colombia 2022 | 6323 | HC+AEF | 0.543 | 4.38 | 0.249 | 0.14 |
| Colombia 2022 | 6323 | TES | 0.584 | 4.18 | 0.238 | 0.07 |
| Colombia 2022 | 6323 | HC+TES | 0.592 | 4.14 | 0.235 | 0.06 |
| Tanzania 2022 | 2427 | HC | 0.640 | 2.38 | 0.323 | 0.04 |
| Tanzania 2022 | 2427 | AEF | 0.722 | 2.09 | 0.284 | 0.00 |
| Tanzania 2022 | 2427 | HC+AEF | 0.735 | 2.04 | 0.278 | 0.07 |
| Tanzania 2022 | 2427 | TES | 0.646 | 2.36 | 0.321 | 0.01 |
| Tanzania 2022 | 2427 | HC+TES | 0.685 | 2.22 | 0.303 | 0.01 |
| Tanzania 2024 | 2212 | HC | 0.580 | 2.63 | 0.364 | 0.04 |
| Tanzania 2024 | 2212 | AEF | 0.686 | 2.28 | 0.315 | 0.02 |
| Tanzania 2024 | 2212 | HC+AEF | 0.686 | 2.28 | 0.315 | 0.01 |
| Tanzania 2024 | 2212 | TES | 0.622 | 2.50 | 0.345 | 0.04 |
| Tanzania 2024 | 2212 | HC+TES | 0.650 | 2.40 | 0.332 | 0.02 |

Paired difference in pooled R² (same partition within each repeat), mean ± sd over repeats:

| site-year | AEF − HC | TES − HC | TES − AEF | HC+AEF − AEF | HC+TES − TES |
|---|---|---|---|---|---|
| Colombia 2021 | 0.187 | 0.160 | -0.027 | 0.005 | 0.002 |
| Colombia 2022 | 0.021 | 0.114 | 0.092 | 0.052 | 0.008 |
| Tanzania 2022 | 0.082 | 0.006 | -0.076 | 0.013 | 0.038 |
| Tanzania 2024 | 0.106 | 0.042 | -0.064 | 0.000 | 0.029 |

Spatial buffer (training cells adjacent to the fold's test blocks dropped), mean over folds:

| site-year | training cells / fold (after buffer) | removed by buffer / fold | removed % |
|---|---|---|---|
| Colombia 2021 | 1273 | 123 | 8.8 |
| Colombia 2022 | 4618 | 441 | 8.7 |
| Tanzania 2022 | 1792 | 150 | 7.7 |
| Tanzania 2024 | 1604 | 165 | 9.3 |

Features kept by the in-fold selection, mean (min–max) over folds and repeats:

| site-year | HC | AEF | HC+AEF | TES | HC+TES |
|---|---|---|---|---|---|
| Colombia 2021 | 28 (25–31) | 29 (28–31) | 47 (45–49) | 70 (65–73) | 84 (69–94) |
| Colombia 2022 | 27 (23–29) | 29 (26–33) | 52 (50–56) | 68 (66–70) | 90 (79–97) |
| Tanzania 2022 | 16 (14–18) | 29 (28–30) | 42 (41–43) | 54 (50–57) | 70 (65–73) |
| Tanzania 2024 | 20 (17–23) | 25 (23–28) | 46 (41–50) | 54 (48–60) | 66 (61–74) |

Per-fold R² ranges from 0.09 to 0.77; the lowest single fold is 0.09 (Colombia 2021, HC).

## E2: cross-country transfer (train Tanzania 2022 → test Colombia 2022)

One model per FS and repeat fitted on all Tanzania 2022 cells, evaluated on all Colombia 2022 cells (mean ± sd over 1 repeats). Transfer loss = E1 − E2 pooled R² within the same repeat; 'per-fold ΔR²' uses the matched E1 test folds (mean ± sd over repeats × folds).

| FS | E1 R² (CO22 in-domain) | E2 R² (TZ22 → CO22) | transfer loss ΔR² (E1 − E2) | per-fold ΔR² | E1 RMSE | E2 RMSE | ΔRMSE (E2 − E1) | E2 bias (m) |
|---|---|---|---|---|---|---|---|---|
| HC | 0.470 | -0.918 | 1.389 | 1.401 ± 0.184 | 4.72 | 8.98 | 4.26 | -6.84 |
| AEF | 0.491 | -2.983 | 3.474 | 3.521 ± 0.456 | 4.63 | 12.94 | 8.32 | -11.37 |
| HC+AEF | 0.543 | -1.436 | 1.979 | 2.006 ± 0.305 | 4.38 | 10.12 | 5.74 | -8.24 |
| TES | 0.584 | -0.895 | 1.479 | 1.498 ± 0.177 | 4.18 | 8.93 | 4.74 | -6.50 |
| HC+TES | 0.592 | -1.030 | 1.622 | 1.640 ± 0.209 | 4.14 | 9.24 | 5.10 | -7.14 |

Target ranges differ strongly: Tanzania 2022 height mean 7.4 m (P95 15.2, max 28.4); Colombia 2022 mean 17.8 m (P95 27.9, max 62.5). 4% of Colombia cells are taller than the tallest Tanzania cell and 66% exceed the Tanzania P95. Tree ensembles (including XGBoost) cannot extrapolate beyond the training target range, so E2 is dominated by a large negative bias; negative R² here means worse than predicting the Colombia mean.

## E3: few-shot recovery in Colombia 2022

Repeat 0 only (the E1 colombia-2022 partition of repeat 0). For each of the 5 folds, k shot blocks are drawn from that fold's training blocks (3 draws per k); models are tested on the fold's test blocks and pooled over the 5 folds. Values: mean ± sd over draws. 'shot cells/fold' = mean number of Colombia cells in the k shot blocks (blocks hold 1–100 cells).

Reference levels (pooled, repeat 0):

| FS | E1 R² (r0) | E2 R² (r0) |
|---|---|---|
| HC | 0.470 | -0.918 |
| AEF | 0.491 | -2.983 |
| HC+AEF | 0.543 | -1.436 |
| TES | 0.584 | -0.895 |
| HC+TES | 0.592 | -1.030 |

| FS | k | tz+k R² | tz+k recovery | shot cells/fold | k-only R² | k-only recovery | tz+k − k-only ΔR² |
|---|---|---|---|---|---|---|---|
| HC | 1 | -0.010 ± 0.153 | 0.65 ± 0.11 | 76 ± 6 | -0.036 ± 0.189 | 0.64 ± 0.14 | 0.026 ± 0.095 |
| HC | 2 | 0.190 ± 0.046 | 0.80 ± 0.03 | 125 ± 11 | -0.009 ± 0.111 | 0.66 ± 0.08 | 0.199 ± 0.122 |
| HC | 5 | 0.230 ± 0.018 | 0.83 ± 0.01 | 309 ± 55 | 0.209 ± 0.064 | 0.81 ± 0.05 | 0.021 ± 0.051 |
| HC | 10 | 0.296 ± 0.023 | 0.87 ± 0.02 | 598 ± 33 | 0.310 ± 0.056 | 0.88 ± 0.04 | -0.013 ± 0.037 |
| HC | 20 | 0.359 ± 0.017 | 0.92 ± 0.01 | 1199 ± 66 | 0.368 ± 0.018 | 0.93 ± 0.01 | -0.008 ± 0.006 |
| AEF | 1 | -0.042 ± 0.172 | 0.85 ± 0.05 | 76 ± 6 | -0.341 ± 0.320 | 0.76 ± 0.09 | 0.300 ± 0.209 |
| AEF | 2 | 0.177 ± 0.084 | 0.91 ± 0.02 | 125 ± 11 | -0.147 ± 0.396 | 0.82 ± 0.11 | 0.324 ± 0.382 |
| AEF | 5 | 0.258 ± 0.037 | 0.93 ± 0.01 | 309 ± 55 | 0.209 ± 0.046 | 0.92 ± 0.01 | 0.049 ± 0.050 |
| AEF | 10 | 0.337 ± 0.057 | 0.96 ± 0.02 | 598 ± 33 | 0.315 ± 0.057 | 0.95 ± 0.02 | 0.022 ± 0.027 |
| AEF | 20 | 0.379 ± 0.015 | 0.97 ± 0.00 | 1199 ± 66 | 0.376 ± 0.040 | 0.97 ± 0.01 | 0.003 ± 0.026 |
| HC+AEF | 1 | 0.058 ± 0.122 | 0.76 ± 0.06 | 76 ± 6 | -0.299 ± 0.458 | 0.57 ± 0.23 | 0.357 ± 0.362 |
| HC+AEF | 2 | 0.222 ± 0.048 | 0.84 ± 0.02 | 125 ± 11 | -0.068 ± 0.089 | 0.69 ± 0.05 | 0.290 ± 0.120 |
| HC+AEF | 5 | 0.290 ± 0.025 | 0.87 ± 0.01 | 309 ± 55 | 0.240 ± 0.065 | 0.85 ± 0.03 | 0.050 ± 0.044 |
| HC+AEF | 10 | 0.401 ± 0.032 | 0.93 ± 0.02 | 598 ± 33 | 0.376 ± 0.061 | 0.92 ± 0.03 | 0.025 ± 0.030 |
| HC+AEF | 20 | 0.423 ± 0.014 | 0.94 ± 0.01 | 1199 ± 66 | 0.426 ± 0.028 | 0.94 ± 0.01 | -0.003 ± 0.015 |
| TES | 1 | -0.033 ± 0.154 | 0.58 ± 0.10 | 76 ± 6 | 0.035 ± 0.078 | 0.63 ± 0.05 | -0.068 ± 0.086 |
| TES | 2 | 0.118 ± 0.075 | 0.69 ± 0.05 | 125 ± 11 | 0.045 ± 0.045 | 0.64 ± 0.03 | 0.073 ± 0.030 |
| TES | 5 | 0.249 ± 0.070 | 0.77 ± 0.05 | 309 ± 55 | 0.063 ± 0.448 | 0.65 ± 0.30 | 0.186 ± 0.400 |
| TES | 10 | 0.418 ± 0.020 | 0.89 ± 0.01 | 598 ± 33 | 0.393 ± 0.063 | 0.87 ± 0.04 | 0.026 ± 0.043 |
| TES | 20 | 0.479 ± 0.010 | 0.93 ± 0.01 | 1199 ± 66 | 0.478 ± 0.006 | 0.93 ± 0.00 | 0.001 ± 0.008 |
| HC+TES | 1 | 0.015 ± 0.157 | 0.64 ± 0.10 | 76 ± 6 | 0.080 ± 0.090 | 0.68 ± 0.06 | -0.065 ± 0.068 |
| HC+TES | 2 | 0.191 ± 0.040 | 0.75 ± 0.02 | 125 ± 11 | 0.005 ± 0.113 | 0.64 ± 0.07 | 0.186 ± 0.125 |
| HC+TES | 5 | 0.318 ± 0.060 | 0.83 ± 0.04 | 309 ± 55 | 0.151 ± 0.265 | 0.73 ± 0.16 | 0.168 ± 0.221 |
| HC+TES | 10 | 0.439 ± 0.017 | 0.91 ± 0.01 | 598 ± 33 | 0.408 ± 0.060 | 0.89 ± 0.04 | 0.031 ± 0.046 |
| HC+TES | 20 | 0.495 ± 0.012 | 0.94 ± 0.01 | 1199 ± 66 | 0.506 ± 0.006 | 0.95 ± 0.00 | -0.011 ± 0.008 |

## E4: cross-year transfer

One partition per site over the union of both years' blocks; A→B trains on year A's training-block cells and tests on year B's test-block cells; the baseline B→B trains on year B's training-block cells and tests on the same cells. Temporal loss = (B→B) − (A→B) pooled R² within a repeat; mean ± sd over 1 repeats. Colombia 2021→2022: 'overlap' (main) tests only on 2022 cells that also exist in 2021, 'full' on all 2022 test cells. 'matched' (Colombia, both directions): both years train only on cells present in both years inside the training blocks and are tested on the same shared test cells, so cross-year and baseline differ only in the year (equal training-set size and footprint). Use 'matched' for the Colombia temporal loss; 'overlap'/'full' confound year with the 3–4× larger 2022 training set.

| direction | FS | baseline B→B R² | cross-year A→B R² | temporal loss ΔR² | baseline RMSE | cross RMSE | cross bias (m) | n test |
|---|---|---|---|---|---|---|---|---|
| Colombia 2021 → 2022 (overlap) | HC | 0.510 | 0.411 | 0.099 | 4.47 | 4.90 | -0.46 | 1462 |
| Colombia 2021 → 2022 (overlap) | AEF | 0.503 | 0.422 | 0.081 | 4.51 | 4.86 | -1.14 | 1462 |
| Colombia 2021 → 2022 (overlap) | HC+AEF | 0.590 | 0.484 | 0.106 | 4.09 | 4.59 | -0.76 | 1462 |
| Colombia 2021 → 2022 (overlap) | TES | 0.598 | 0.445 | 0.153 | 4.05 | 4.76 | -1.59 | 1462 |
| Colombia 2021 → 2022 (overlap) | HC+TES | 0.610 | 0.490 | 0.120 | 3.99 | 4.56 | -1.31 | 1462 |
| Colombia 2021 → 2022 (full) | HC | 0.461 | 0.354 | 0.107 | 4.76 | 5.21 | -0.16 | 6323 |
| Colombia 2021 → 2022 (full) | AEF | 0.488 | 0.298 | 0.190 | 4.64 | 5.44 | -2.07 | 6323 |
| Colombia 2021 → 2022 (full) | HC+AEF | 0.549 | 0.366 | 0.183 | 4.36 | 5.17 | -1.62 | 6323 |
| Colombia 2021 → 2022 (full) | TES | 0.580 | 0.332 | 0.248 | 4.20 | 5.30 | -2.00 | 6323 |
| Colombia 2021 → 2022 (full) | HC+TES | 0.588 | 0.375 | 0.213 | 4.16 | 5.13 | -1.68 | 6323 |
| Colombia 2022 → 2021 | HC | 0.337 | 0.365 | -0.028 | 5.18 | 5.07 | 0.75 | 1745 |
| Colombia 2022 → 2021 | AEF | 0.498 | 0.334 | 0.164 | 4.51 | 5.19 | 2.11 | 1745 |
| Colombia 2022 → 2021 | HC+AEF | 0.516 | 0.430 | 0.087 | 4.42 | 4.80 | 1.55 | 1745 |
| Colombia 2022 → 2021 | TES | 0.487 | 0.376 | 0.111 | 4.56 | 5.03 | 2.22 | 1745 |
| Colombia 2022 → 2021 | HC+TES | 0.505 | 0.426 | 0.078 | 4.48 | 4.82 | 1.59 | 1745 |
| Colombia 2021 → 2022 (matched) | HC | 0.454 | 0.399 | 0.055 | 4.72 | 4.95 | -0.73 | 1462 |
| Colombia 2021 → 2022 (matched) | AEF | 0.474 | 0.398 | 0.076 | 4.63 | 4.96 | -1.21 | 1462 |
| Colombia 2021 → 2022 (matched) | HC+AEF | 0.521 | 0.464 | 0.056 | 4.42 | 4.68 | -0.99 | 1462 |
| Colombia 2021 → 2022 (matched) | TES | 0.527 | 0.446 | 0.081 | 4.39 | 4.75 | -1.38 | 1462 |
| Colombia 2021 → 2022 (matched) | HC+TES | 0.541 | 0.478 | 0.063 | 4.33 | 4.62 | -1.29 | 1462 |
| Colombia 2022 → 2021 (matched) | HC | 0.387 | 0.365 | 0.022 | 4.99 | 5.08 | 0.46 | 1462 |
| Colombia 2022 → 2021 (matched) | AEF | 0.523 | 0.373 | 0.150 | 4.40 | 5.05 | 1.69 | 1462 |
| Colombia 2022 → 2021 (matched) | HC+AEF | 0.534 | 0.396 | 0.138 | 4.35 | 4.95 | 1.77 | 1462 |
| Colombia 2022 → 2021 (matched) | TES | 0.534 | 0.424 | 0.110 | 4.35 | 4.84 | 1.12 | 1462 |
| Colombia 2022 → 2021 (matched) | HC+TES | 0.548 | 0.451 | 0.097 | 4.29 | 4.73 | 0.99 | 1462 |
| Tanzania 2022 → 2024 | HC | 0.573 | 0.544 | 0.029 | 2.66 | 2.74 | -0.88 | 2212 |
| Tanzania 2022 → 2024 | AEF | 0.683 | 0.418 | 0.264 | 2.29 | 3.10 | -1.17 | 2212 |
| Tanzania 2022 → 2024 | HC+AEF | 0.675 | 0.560 | 0.115 | 2.32 | 2.70 | -0.93 | 2212 |
| Tanzania 2022 → 2024 | TES | 0.614 | 0.460 | 0.154 | 2.53 | 2.99 | 0.89 | 2212 |
| Tanzania 2022 → 2024 | HC+TES | 0.648 | 0.575 | 0.073 | 2.41 | 2.65 | -0.53 | 2212 |
| Tanzania 2024 → 2022 | HC | 0.628 | 0.529 | 0.098 | 2.42 | 2.72 | 0.94 | 2427 |
| Tanzania 2024 → 2022 | AEF | 0.708 | 0.332 | 0.376 | 2.14 | 3.24 | 1.83 | 2427 |
| Tanzania 2024 → 2022 | HC+AEF | 0.719 | 0.297 | 0.422 | 2.10 | 3.32 | 2.18 | 2427 |
| Tanzania 2024 → 2022 | TES | 0.648 | 0.491 | 0.157 | 2.35 | 2.83 | 0.23 | 2427 |
| Tanzania 2024 → 2022 | HC+TES | 0.674 | 0.599 | 0.076 | 2.26 | 2.51 | 0.62 | 2427 |

## Caveats

- **Tessera coverage:** Tessera leaves pixels with fewer than 15 cloud-free observations per year empty; about 21 % (2021) and 18 % (2022) of Colombia cells have no Tessera data. With `pipeline.cells = covered` these cells are excluded for every feature set, so results describe the less cloudy part of the Colombia site; Tanzania is fully covered. Earlier all-cell results (no Tessera, no selection) are archived in `results/archive_xgb_allcells_noselect/`.
- Tessera (TESSERA, Feng et al. 2025, arXiv:2506.20380) was accessed on 2026-10-02 from https://registry.opendata.aws/tessera (dClimate v1.1 store, CC-BY-SA).
- **E3 uses repeat 0 only** (one block partition); its sd is over draws of shot blocks, not over partitions. 3 draws per k were run (config: 3).
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
