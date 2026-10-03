# Canopy height (P95, 100 m): handcrafted S1/S2 vs AlphaEarth and Tessera embeddings

PLS regression ({pc} components, standardised features, predictions clipped at 0 m; linear baseline) on 1 km spatial block splits. FS: HC = 98 handcrafted S1/S2 features (incl. 24 GLCM), AEF = 128 AlphaEarth features (64 mean + 64 std), TES = 256 Tessera features (128 mean + 128 std), HC+AEF / HC+TES = combinations. Metrics: R², RMSE (m), rRMSE = RMSE / mean(obs), bias = mean(pred − obs). Figures in `results/plsr/figures/`.

**Pipeline.** Cells: only cells with Tessera data (`covered`), identical for every feature set; block partitions are built on all cells first, so folds match earlier all-cell runs. Feature selection: features whose mean |SHAP| is above the mean over all features (threshold = mean) of a selector XGBoost (same hyper-parameters), fitted inside every training fold (SHAP on at most 3000 training rows); the final XGBoost is refitted on the selected features. The number of selected features therefore varies by feature set and fold (see the selection table).

## Key results

- E1 Colombia 2021: R² HC 0.30, AEF 0.53, HC+AEF 0.54, TES 0.50, HC+TES 0.52; AEF − HC = 0.230, TES − HC = 0.207, TES − AEF = -0.023, HC+AEF − AEF = 0.014, HC+TES − TES = 0.014.
- E1 Colombia 2022: R² HC 0.36, AEF 0.50, HC+AEF 0.50, TES 0.57, HC+TES 0.58; AEF − HC = 0.140, TES − HC = 0.213, TES − AEF = 0.073, HC+AEF − AEF = 0.006, HC+TES − TES = 0.006.
- E1 Tanzania 2022: R² HC 0.58, AEF 0.70, HC+AEF 0.72, TES 0.63, HC+TES 0.65; AEF − HC = 0.121, TES − HC = 0.046, TES − AEF = -0.075, HC+AEF − AEF = 0.014, HC+TES − TES = 0.018.
- E1 Tanzania 2024: R² HC 0.55, AEF 0.70, HC+AEF 0.70, TES 0.61, HC+TES 0.62; AEF − HC = 0.153, TES − HC = 0.063, TES − AEF = -0.090, HC+AEF − AEF = 0.002, HC+TES − TES = 0.013.
- E2 (Tanzania 2022 → Colombia 2022) fails for every FS: R² HC -0.73, AEF -3.33, HC+AEF -3.47, TES -1.94, HC+TES -0.71; mean bias HC -5.6 m, AEF -12.4 m, HC+AEF -12.7 m, TES -9.1 m, HC+TES -6.0 m.
- E3 at k = 20 blocks: tz+k R² HC 0.25, AEF 0.44, HC+AEF 0.44, TES 0.41, HC+TES 0.44; k-only R² HC 0.31, AEF 0.45, HC+AEF 0.46, TES 0.52, HC+TES 0.53. Mean tz+k − k-only ΔR² over all k and FS ranges from -0.209 to 0.784.
- E4 Colombia 2021 → 2022 (overlap): temporal loss ΔR² HC 0.306, AEF 0.353, HC+AEF 0.279, TES 0.341, HC+TES 0.394.
- E4 Colombia 2021 → 2022 (full): temporal loss ΔR² HC 0.311, AEF 0.413, HC+AEF 0.362, TES 0.526, HC+TES 0.583.
- E4 Colombia 2022 → 2021: temporal loss ΔR² HC 0.128, AEF 0.101, HC+AEF 0.040, TES 0.208, HC+TES 0.254.
- E4 Colombia 2021 → 2022 (matched): temporal loss ΔR² HC 0.294, AEF 0.320, HC+AEF 0.303, TES 0.193, HC+TES 0.241.
- E4 Colombia 2022 → 2021 (matched): temporal loss ΔR² HC 0.066, AEF 0.158, HC+AEF 0.120, TES 0.166, HC+TES 0.160.
- E4 Tanzania 2022 → 2024: temporal loss ΔR² HC 0.051, AEF 0.566, HC+AEF 0.335, TES 0.069, HC+TES 0.086.
- E4 Tanzania 2024 → 2022: temporal loss ΔR² HC 0.078, AEF 0.324, HC+AEF 0.202, TES 0.181, HC+TES 0.178.

## Run settings

Latest run per experiment (from `results/run_log.csv`; n_draws only for E3).

| experiment | time | n_estimators | n_repeats | n_draws | quick |
|---|---|---|---|---|---|
| E1 | 2026-10-03 06:08:20 | 200 | 1 | -1 | False |
| E2 | 2026-10-03 06:09:09 | 200 | 1 | -1 | False |
| E4 | 2026-10-03 06:20:12 | 200 | 1 | -1 | False |
| E4matched | 2026-10-03 06:24:43 | 200 | 1 | -1 | False |
| E3 | 2026-10-03 07:31:01 | 200 | 1 | 3 | False |

## E1: in-domain blocked CV

Pooled out-of-fold metrics per repeat, mean ± sd over 1 repeats (5 folds of 1 km blocks, folds balanced by cell count).

| site-year | n cells | FS | R² | RMSE (m) | rRMSE | bias (m) |
|---|---|---|---|---|---|---|
| Colombia 2021 | 1745 | HC | 0.298 | 5.33 | 0.318 | -0.24 |
| Colombia 2021 | 1745 | AEF | 0.528 | 4.37 | 0.261 | -0.26 |
| Colombia 2021 | 1745 | HC+AEF | 0.541 | 4.31 | 0.257 | -0.13 |
| Colombia 2021 | 1745 | TES | 0.505 | 4.48 | 0.267 | -0.02 |
| Colombia 2021 | 1745 | HC+TES | 0.518 | 4.42 | 0.263 | -0.09 |
| Colombia 2022 | 6323 | HC | 0.358 | 5.20 | 0.295 | -0.12 |
| Colombia 2022 | 6323 | AEF | 0.498 | 4.60 | 0.261 | -0.07 |
| Colombia 2022 | 6323 | HC+AEF | 0.504 | 4.57 | 0.260 | -0.12 |
| Colombia 2022 | 6323 | TES | 0.571 | 4.25 | 0.241 | 0.00 |
| Colombia 2022 | 6323 | HC+TES | 0.577 | 4.22 | 0.240 | -0.05 |
| Tanzania 2022 | 2427 | HC | 0.583 | 2.56 | 0.348 | 0.02 |
| Tanzania 2022 | 2427 | AEF | 0.704 | 2.16 | 0.293 | -0.02 |
| Tanzania 2022 | 2427 | HC+AEF | 0.718 | 2.10 | 0.286 | 0.04 |
| Tanzania 2022 | 2427 | TES | 0.629 | 2.41 | 0.328 | -0.01 |
| Tanzania 2022 | 2427 | HC+TES | 0.647 | 2.35 | 0.320 | -0.00 |
| Tanzania 2024 | 2212 | HC | 0.547 | 2.74 | 0.378 | -0.05 |
| Tanzania 2024 | 2212 | AEF | 0.700 | 2.22 | 0.307 | -0.07 |
| Tanzania 2024 | 2212 | HC+AEF | 0.703 | 2.22 | 0.306 | -0.03 |
| Tanzania 2024 | 2212 | TES | 0.610 | 2.54 | 0.351 | -0.09 |
| Tanzania 2024 | 2212 | HC+TES | 0.623 | 2.50 | 0.345 | -0.03 |

Paired difference in pooled R² (same partition within each repeat), mean ± sd over repeats:

| site-year | AEF − HC | TES − HC | TES − AEF | HC+AEF − AEF | HC+TES − TES |
|---|---|---|---|---|---|
| Colombia 2021 | 0.230 | 0.207 | -0.023 | 0.014 | 0.014 |
| Colombia 2022 | 0.140 | 0.213 | 0.073 | 0.006 | 0.006 |
| Tanzania 2022 | 0.121 | 0.046 | -0.075 | 0.014 | 0.018 |
| Tanzania 2024 | 0.153 | 0.063 | -0.090 | 0.002 | 0.013 |

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

Per-fold R² ranges from 0.11 to 0.77; the lowest single fold is 0.11 (Colombia 2021, HC).

## E2: cross-country transfer (train Tanzania 2022 → test Colombia 2022)

One model per FS and repeat fitted on all Tanzania 2022 cells, evaluated on all Colombia 2022 cells (mean ± sd over 1 repeats). Transfer loss = E1 − E2 pooled R² within the same repeat; 'per-fold ΔR²' uses the matched E1 test folds (mean ± sd over repeats × folds).

| FS | E1 R² (CO22 in-domain) | E2 R² (TZ22 → CO22) | transfer loss ΔR² (E1 − E2) | per-fold ΔR² | E1 RMSE | E2 RMSE | ΔRMSE (E2 − E1) | E2 bias (m) |
|---|---|---|---|---|---|---|---|---|
| HC | 0.358 | -0.733 | 1.091 | 1.126 ± 0.334 | 5.20 | 8.54 | 3.34 | -5.64 |
| AEF | 0.498 | -3.331 | 3.829 | 3.885 ± 0.457 | 4.60 | 13.50 | 8.90 | -12.35 |
| HC+AEF | 0.504 | -3.474 | 3.977 | 4.047 ± 0.604 | 4.57 | 13.72 | 9.15 | -12.67 |
| TES | 0.571 | -1.939 | 2.510 | 2.544 ± 0.367 | 4.25 | 11.12 | 6.87 | -9.09 |
| HC+TES | 0.577 | -0.707 | 1.283 | 1.303 ± 0.200 | 4.22 | 8.47 | 4.25 | -6.03 |

Target ranges differ strongly: Tanzania 2022 height mean 7.4 m (P95 15.2, max 28.4); Colombia 2022 mean 17.8 m (P95 27.9, max 62.5). 4% of Colombia cells are taller than the tallest Tanzania cell and 66% exceed the Tanzania P95. Tree ensembles (including XGBoost) cannot extrapolate beyond the training target range, so E2 is dominated by a large negative bias; negative R² here means worse than predicting the Colombia mean.

## E3: few-shot recovery in Colombia 2022

Repeat 0 only (the E1 colombia-2022 partition of repeat 0). For each of the 5 folds, k shot blocks are drawn from that fold's training blocks (3 draws per k); models are tested on the fold's test blocks and pooled over the 5 folds. Values: mean ± sd over draws. 'shot cells/fold' = mean number of Colombia cells in the k shot blocks (blocks hold 1–100 cells).

Reference levels (pooled, repeat 0):

| FS | E1 R² (r0) | E2 R² (r0) |
|---|---|---|
| HC | 0.358 | -0.733 |
| AEF | 0.498 | -3.331 |
| HC+AEF | 0.504 | -3.474 |
| TES | 0.571 | -1.939 |
| HC+TES | 0.577 | -0.707 |

| FS | k | tz+k R² | tz+k recovery | shot cells/fold | k-only R² | k-only recovery | tz+k − k-only ΔR² |
|---|---|---|---|---|---|---|---|
| HC | 1 | -0.105 ± 0.174 | 0.58 ± 0.16 | 76 ± 6 | -0.434 ± 0.300 | 0.27 ± 0.28 | 0.329 ± 0.235 |
| HC | 2 | 0.113 ± 0.042 | 0.78 ± 0.04 | 125 ± 11 | -0.108 ± 0.154 | 0.57 ± 0.14 | 0.221 ± 0.160 |
| HC | 5 | 0.193 ± 0.025 | 0.85 ± 0.02 | 309 ± 55 | 0.149 ± 0.041 | 0.81 ± 0.04 | 0.044 ± 0.016 |
| HC | 10 | 0.232 ± 0.034 | 0.88 ± 0.03 | 598 ± 33 | 0.258 ± 0.105 | 0.91 ± 0.10 | -0.026 ± 0.072 |
| HC | 20 | 0.254 ± 0.014 | 0.90 ± 0.01 | 1199 ± 66 | 0.308 ± 0.012 | 0.95 ± 0.01 | -0.054 ± 0.012 |
| AEF | 1 | 0.266 ± 0.054 | 0.94 ± 0.01 | 76 ± 6 | -0.517 ± 0.217 | 0.73 ± 0.06 | 0.784 ± 0.267 |
| AEF | 2 | 0.336 ± 0.034 | 0.96 ± 0.01 | 125 ± 11 | 0.136 ± 0.127 | 0.91 ± 0.03 | 0.199 ± 0.093 |
| AEF | 5 | 0.362 ± 0.022 | 0.96 ± 0.01 | 309 ± 55 | 0.337 ± 0.040 | 0.96 ± 0.01 | 0.025 ± 0.035 |
| AEF | 10 | 0.399 ± 0.035 | 0.97 ± 0.01 | 598 ± 33 | 0.413 ± 0.040 | 0.98 ± 0.01 | -0.013 ± 0.007 |
| AEF | 20 | 0.440 ± 0.007 | 0.98 ± 0.00 | 1199 ± 66 | 0.452 ± 0.011 | 0.99 ± 0.00 | -0.012 ± 0.005 |
| HC+AEF | 1 | 0.171 ± 0.102 | 0.92 ± 0.03 | 76 ± 6 | -0.610 ± 0.412 | 0.72 ± 0.10 | 0.781 ± 0.347 |
| HC+AEF | 2 | 0.297 ± 0.027 | 0.95 ± 0.01 | 125 ± 11 | 0.046 ± 0.059 | 0.88 ± 0.01 | 0.251 ± 0.064 |
| HC+AEF | 5 | 0.345 ± 0.033 | 0.96 ± 0.01 | 309 ± 55 | 0.307 ± 0.070 | 0.95 ± 0.02 | 0.038 ± 0.037 |
| HC+AEF | 10 | 0.401 ± 0.009 | 0.97 ± 0.00 | 598 ± 33 | 0.405 ± 0.051 | 0.98 ± 0.01 | -0.005 ± 0.048 |
| HC+AEF | 20 | 0.438 ± 0.006 | 0.98 ± 0.00 | 1199 ± 66 | 0.465 ± 0.016 | 0.99 ± 0.00 | -0.027 ± 0.016 |
| TES | 1 | -0.104 ± 0.150 | 0.73 ± 0.06 | 76 ± 6 | -0.268 ± 0.291 | 0.67 ± 0.12 | 0.164 ± 0.191 |
| TES | 2 | 0.046 ± 0.073 | 0.79 ± 0.03 | 125 ± 11 | 0.130 ± 0.048 | 0.82 ± 0.02 | -0.084 ± 0.025 |
| TES | 5 | 0.173 ± 0.056 | 0.84 ± 0.02 | 309 ± 55 | 0.383 ± 0.057 | 0.93 ± 0.02 | -0.209 ± 0.004 |
| TES | 10 | 0.337 ± 0.038 | 0.91 ± 0.01 | 598 ± 33 | 0.475 ± 0.018 | 0.96 ± 0.01 | -0.138 ± 0.032 |
| TES | 20 | 0.413 ± 0.014 | 0.94 ± 0.01 | 1199 ± 66 | 0.522 ± 0.003 | 0.98 ± 0.00 | -0.108 ± 0.016 |
| HC+TES | 1 | -0.009 ± 0.022 | 0.54 ± 0.02 | 76 ± 6 | -0.446 ± 0.333 | 0.20 ± 0.26 | 0.437 ± 0.330 |
| HC+TES | 2 | 0.088 ± 0.073 | 0.62 ± 0.06 | 125 ± 11 | -0.031 ± 0.089 | 0.53 ± 0.07 | 0.119 ± 0.155 |
| HC+TES | 5 | 0.206 ± 0.080 | 0.71 ± 0.06 | 309 ± 55 | 0.405 ± 0.047 | 0.87 ± 0.04 | -0.199 ± 0.037 |
| HC+TES | 10 | 0.346 ± 0.024 | 0.82 ± 0.02 | 598 ± 33 | 0.490 ± 0.036 | 0.93 ± 0.03 | -0.144 ± 0.043 |
| HC+TES | 20 | 0.444 ± 0.018 | 0.90 ± 0.01 | 1199 ± 66 | 0.532 ± 0.001 | 0.97 ± 0.00 | -0.088 ± 0.019 |

## E4: cross-year transfer

One partition per site over the union of both years' blocks; A→B trains on year A's training-block cells and tests on year B's test-block cells; the baseline B→B trains on year B's training-block cells and tests on the same cells. Temporal loss = (B→B) − (A→B) pooled R² within a repeat; mean ± sd over 1 repeats. Colombia 2021→2022: 'overlap' (main) tests only on 2022 cells that also exist in 2021, 'full' on all 2022 test cells. 'matched' (Colombia, both directions): both years train only on cells present in both years inside the training blocks and are tested on the same shared test cells, so cross-year and baseline differ only in the year (equal training-set size and footprint). Use 'matched' for the Colombia temporal loss; 'overlap'/'full' confound year with the 3–4× larger 2022 training set.

| direction | FS | baseline B→B R² | cross-year A→B R² | temporal loss ΔR² | baseline RMSE | cross RMSE | cross bias (m) | n test |
|---|---|---|---|---|---|---|---|---|
| Colombia 2021 → 2022 (overlap) | HC | 0.392 | 0.086 | 0.306 | 4.98 | 6.11 | -2.46 | 1462 |
| Colombia 2021 → 2022 (overlap) | AEF | 0.509 | 0.156 | 0.353 | 4.48 | 5.87 | -3.60 | 1462 |
| Colombia 2021 → 2022 (overlap) | HC+AEF | 0.543 | 0.263 | 0.279 | 4.32 | 5.48 | -2.88 | 1462 |
| Colombia 2021 → 2022 (overlap) | TES | 0.539 | 0.198 | 0.341 | 4.34 | 5.72 | -3.25 | 1462 |
| Colombia 2021 → 2022 (overlap) | HC+TES | 0.579 | 0.185 | 0.394 | 4.15 | 5.77 | -3.27 | 1462 |
| Colombia 2021 → 2022 (full) | HC | 0.349 | 0.038 | 0.311 | 5.23 | 6.36 | -2.75 | 6323 |
| Colombia 2021 → 2022 (full) | AEF | 0.496 | 0.084 | 0.413 | 4.60 | 6.21 | -3.95 | 6323 |
| Colombia 2021 → 2022 (full) | HC+AEF | 0.508 | 0.146 | 0.362 | 4.55 | 5.99 | -3.56 | 6323 |
| Colombia 2021 → 2022 (full) | TES | 0.565 | 0.039 | 0.526 | 4.28 | 6.36 | -4.14 | 6323 |
| Colombia 2021 → 2022 (full) | HC+TES | 0.575 | -0.007 | 0.583 | 4.23 | 6.51 | -4.39 | 6323 |
| Colombia 2022 → 2021 | HC | 0.306 | 0.178 | 0.128 | 5.30 | 5.77 | 0.93 | 1745 |
| Colombia 2022 → 2021 | AEF | 0.502 | 0.401 | 0.101 | 4.49 | 4.92 | 1.98 | 1745 |
| Colombia 2022 → 2021 | HC+AEF | 0.516 | 0.476 | 0.040 | 4.43 | 4.61 | 1.58 | 1745 |
| Colombia 2022 → 2021 | TES | 0.505 | 0.297 | 0.208 | 4.47 | 5.33 | 2.77 | 1745 |
| Colombia 2022 → 2021 | HC+TES | 0.524 | 0.270 | 0.254 | 4.39 | 5.44 | 2.83 | 1745 |
| Colombia 2021 → 2022 (matched) | HC | 0.395 | 0.101 | 0.294 | 4.97 | 6.06 | -2.36 | 1462 |
| Colombia 2021 → 2022 (matched) | AEF | 0.474 | 0.154 | 0.320 | 4.64 | 5.88 | -3.67 | 1462 |
| Colombia 2021 → 2022 (matched) | HC+AEF | 0.485 | 0.182 | 0.303 | 4.59 | 5.78 | -3.32 | 1462 |
| Colombia 2021 → 2022 (matched) | TES | 0.484 | 0.291 | 0.193 | 4.59 | 5.38 | -2.71 | 1462 |
| Colombia 2021 → 2022 (matched) | HC+TES | 0.554 | 0.312 | 0.241 | 4.27 | 5.30 | -2.88 | 1462 |
| Colombia 2022 → 2021 (matched) | HC | 0.342 | 0.276 | 0.066 | 5.17 | 5.43 | 0.67 | 1462 |
| Colombia 2022 → 2021 (matched) | AEF | 0.548 | 0.390 | 0.158 | 4.29 | 4.98 | 2.25 | 1462 |
| Colombia 2022 → 2021 (matched) | HC+AEF | 0.528 | 0.408 | 0.120 | 4.38 | 4.91 | 1.49 | 1462 |
| Colombia 2022 → 2021 (matched) | TES | 0.551 | 0.385 | 0.166 | 4.27 | 5.00 | 1.23 | 1462 |
| Colombia 2022 → 2021 (matched) | HC+TES | 0.579 | 0.418 | 0.160 | 4.14 | 4.86 | 1.57 | 1462 |
| Tanzania 2022 → 2024 | HC | 0.544 | 0.493 | 0.051 | 2.74 | 2.89 | -0.85 | 2212 |
| Tanzania 2022 → 2024 | AEF | 0.694 | 0.129 | 0.566 | 2.25 | 3.79 | -2.83 | 2212 |
| Tanzania 2022 → 2024 | HC+AEF | 0.706 | 0.371 | 0.335 | 2.20 | 3.22 | -2.23 | 2212 |
| Tanzania 2022 → 2024 | TES | 0.608 | 0.539 | 0.069 | 2.55 | 2.76 | -0.13 | 2212 |
| Tanzania 2022 → 2024 | HC+TES | 0.632 | 0.546 | 0.086 | 2.46 | 2.74 | -0.13 | 2212 |
| Tanzania 2024 → 2022 | HC | 0.580 | 0.501 | 0.078 | 2.57 | 2.80 | 0.86 | 2427 |
| Tanzania 2024 → 2022 | AEF | 0.712 | 0.387 | 0.324 | 2.13 | 3.10 | 2.08 | 2427 |
| Tanzania 2024 → 2022 | HC+AEF | 0.725 | 0.523 | 0.202 | 2.08 | 2.74 | 1.59 | 2427 |
| Tanzania 2024 → 2022 | TES | 0.634 | 0.453 | 0.181 | 2.40 | 2.93 | -1.22 | 2427 |
| Tanzania 2024 → 2022 | HC+TES | 0.650 | 0.472 | 0.178 | 2.34 | 2.88 | -1.47 | 2427 |

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
