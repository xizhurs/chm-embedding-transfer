# Canopy height (P95, 100 m): handcrafted S1/S2 vs AlphaEarth and Tessera embeddings

XGBoost (n_estimators = 200, learning_rate = 0.1, max_depth = 6, subsample = 0.8, colsample_bytree = 0.8, no tuning) on 1 km spatial block splits. FS: HC = 98 handcrafted S1/S2 features (incl. 24 GLCM), AEF = 128 AlphaEarth features (64 mean + 64 std), TES = 256 Tessera features (128 mean + 128 std), HC+AEF / HC+TES = combinations. Metrics: R², RMSE (m), rRMSE = RMSE / mean(obs), bias = mean(pred − obs). Figures in `results/figures/`.

## Key results

- E1 Colombia 2021: R² HC 0.35, AEF 0.53, HC+AEF 0.54; AEF − HC = 0.181 ± 0.007, HC+AEF − AEF = 0.008 ± 0.012.
- E1 Colombia 2022: R² HC 0.44, AEF 0.50, HC+AEF 0.53; AEF − HC = 0.065 ± 0.005, HC+AEF − AEF = 0.029 ± 0.003.
- E1 Tanzania 2022: R² HC 0.66, AEF 0.73, HC+AEF 0.74; AEF − HC = 0.074 ± 0.009, HC+AEF − AEF = 0.010 ± 0.008.
- E1 Tanzania 2024: R² HC 0.61, AEF 0.70, HC+AEF 0.70; AEF − HC = 0.087 ± 0.010, HC+AEF − AEF = -0.003 ± 0.011.
- E2 (Tanzania 2022 → Colombia 2022) fails for every FS: R² HC -1.33, AEF -2.91, HC+AEF -1.69; mean bias HC -7.9 m, AEF -11.1 m, HC+AEF -8.7 m.
- E3 at k = 20 blocks: tz+k R² HC 0.36, AEF 0.40, HC+AEF 0.44; k-only R² HC 0.36, AEF 0.40, HC+AEF 0.43. Mean tz+k − k-only ΔR² over all k and FS ranges from -0.015 to 0.248.
- E4 Colombia 2021 → 2022 (overlap): temporal loss ΔR² HC 0.153, AEF 0.116, HC+AEF 0.132.
- E4 Colombia 2021 → 2022 (full): temporal loss ΔR² HC 0.134, AEF 0.215, HC+AEF 0.188.
- E4 Colombia 2022 → 2021: temporal loss ΔR² HC 0.014, AEF 0.097, HC+AEF 0.075.
- E4 Colombia 2021 → 2022 (matched): temporal loss ΔR² HC 0.132, AEF 0.096, HC+AEF 0.098.
- E4 Colombia 2022 → 2021 (matched): temporal loss ΔR² HC 0.035, AEF 0.111, HC+AEF 0.095.
- E4 Tanzania 2022 → 2024: temporal loss ΔR² HC 0.051, AEF 0.237, HC+AEF 0.136.
- E4 Tanzania 2024 → 2022: temporal loss ΔR² HC 0.109, AEF 0.302, HC+AEF 0.286.

## Run settings

Latest run per experiment (from `results/run_log.csv`; n_draws only for E3).

| experiment | time | n_estimators | n_repeats | n_draws | quick |
|---|---|---|---|---|---|
| E1 | 2026-10-02 14:56:00 | 200 | 5 | -1 | False |
| E2 | 2026-10-02 14:56:00 | 200 | 5 | -1 | False |
| E4 | 2026-10-02 14:56:00 | 200 | 5 | -1 | False |
| E3 | 2026-10-02 15:43:22 | 200 | 1 | 3 | False |
| E4matched | 2026-10-02 16:28:00 | 200 | 5 | -1 | False |

## E1: in-domain blocked CV

Pooled out-of-fold metrics per repeat, mean ± sd over 5 repeats (5 folds of 1 km blocks, folds balanced by cell count).

| site-year | n cells | FS | R² | RMSE (m) | rRMSE | bias (m) |
|---|---|---|---|---|---|---|
| Colombia 2021 | 2212 | HC | 0.353 ± 0.009 | 5.16 ± 0.03 | 0.299 ± 0.002 | 0.12 ± 0.05 |
| Colombia 2021 | 2212 | AEF | 0.534 ± 0.008 | 4.38 ± 0.04 | 0.254 ± 0.002 | -0.03 ± 0.02 |
| Colombia 2021 | 2212 | HC+AEF | 0.541 ± 0.007 | 4.34 ± 0.03 | 0.252 ± 0.002 | 0.03 ± 0.02 |
| Colombia 2022 | 7744 | HC | 0.438 ± 0.006 | 4.85 ± 0.02 | 0.273 ± 0.001 | 0.02 ± 0.01 |
| Colombia 2022 | 7744 | AEF | 0.504 ± 0.005 | 4.56 ± 0.02 | 0.257 ± 0.001 | 0.05 ± 0.02 |
| Colombia 2022 | 7744 | HC+AEF | 0.533 ± 0.005 | 4.42 ± 0.02 | 0.249 ± 0.001 | 0.05 ± 0.04 |
| Tanzania 2022 | 2427 | HC | 0.658 ± 0.007 | 2.32 ± 0.02 | 0.315 ± 0.003 | 0.03 ± 0.02 |
| Tanzania 2022 | 2427 | AEF | 0.732 ± 0.006 | 2.05 ± 0.02 | 0.279 ± 0.003 | 0.02 ± 0.03 |
| Tanzania 2022 | 2427 | HC+AEF | 0.742 ± 0.006 | 2.01 ± 0.02 | 0.274 ± 0.003 | 0.03 ± 0.02 |
| Tanzania 2024 | 2212 | HC | 0.612 ± 0.006 | 2.53 ± 0.02 | 0.350 ± 0.003 | 0.07 ± 0.03 |
| Tanzania 2024 | 2212 | AEF | 0.699 ± 0.006 | 2.23 ± 0.02 | 0.308 ± 0.003 | 0.05 ± 0.03 |
| Tanzania 2024 | 2212 | HC+AEF | 0.697 ± 0.009 | 2.24 ± 0.03 | 0.309 ± 0.005 | 0.06 ± 0.03 |

Paired difference in pooled R² (same partition within each repeat), mean ± sd over repeats:

| site-year | AEF − HC | HC+AEF − AEF | HC+AEF − HC |
|---|---|---|---|
| Colombia 2021 | 0.181 ± 0.007 | 0.008 ± 0.012 | 0.188 ± 0.007 |
| Colombia 2022 | 0.065 ± 0.005 | 0.029 ± 0.003 | 0.095 ± 0.006 |
| Tanzania 2022 | 0.074 ± 0.009 | 0.010 ± 0.008 | 0.084 ± 0.011 |
| Tanzania 2024 | 0.087 ± 0.010 | -0.003 ± 0.011 | 0.084 ± 0.008 |

Per-fold R² ranges from 0.13 to 0.79; the lowest single fold is 0.13 (Colombia 2021, HC).

## E2: cross-country transfer (train Tanzania 2022 → test Colombia 2022)

One model per FS and repeat fitted on all Tanzania 2022 cells, evaluated on all Colombia 2022 cells (mean ± sd over 5 repeats). Transfer loss = E1 − E2 pooled R² within the same repeat; 'per-fold ΔR²' uses the matched E1 test folds (mean ± sd over repeats × folds).

| FS | E1 R² (CO22 in-domain) | E2 R² (TZ22 → CO22) | transfer loss ΔR² (E1 − E2) | per-fold ΔR² | E1 RMSE | E2 RMSE | ΔRMSE (E2 − E1) | E2 bias (m) |
|---|---|---|---|---|---|---|---|---|
| HC | 0.438 ± 0.006 | -1.334 ± 0.044 | 1.772 ± 0.041 | 1.797 ± 0.225 | 4.85 ± 0.02 | 9.88 ± 0.09 | 5.03 ± 0.08 | -7.90 ± 0.13 |
| AEF | 0.504 ± 0.005 | -2.908 ± 0.237 | 3.412 ± 0.234 | 3.459 ± 0.484 | 4.56 ± 0.02 | 12.79 ± 0.39 | 8.23 ± 0.38 | -11.12 ± 0.43 |
| HC+AEF | 0.533 ± 0.005 | -1.693 ± 0.137 | 2.226 ± 0.139 | 2.259 ± 0.348 | 4.42 ± 0.02 | 10.61 ± 0.27 | 6.19 ± 0.28 | -8.67 ± 0.34 |

Target ranges differ strongly: Tanzania 2022 height mean 7.4 m (P95 15.2, max 28.4); Colombia 2022 mean 17.8 m (P95 27.9, max 62.5). 4% of Colombia cells are taller than the tallest Tanzania cell and 66% exceed the Tanzania P95. Tree ensembles (including XGBoost) cannot extrapolate beyond the training target range, so E2 is dominated by a large negative bias; negative R² here means worse than predicting the Colombia mean.

## E3: few-shot recovery in Colombia 2022

Repeat 0 only (the E1 colombia-2022 partition of repeat 0). For each of the 5 folds, k shot blocks are drawn from that fold's training blocks (3 draws per k); models are tested on the fold's test blocks and pooled over the 5 folds. Values: mean ± sd over draws. 'shot cells/fold' = mean number of Colombia cells in the k shot blocks (blocks hold 1–100 cells).

Reference levels (pooled, repeat 0):

| FS | E1 R² (r0) | E2 R² (r0) |
|---|---|---|
| HC | 0.430 | -1.383 |
| AEF | 0.496 | -3.039 |
| HC+AEF | 0.526 | -1.594 |

| FS | k | tz+k R² | tz+k recovery | shot cells/fold | k-only R² | k-only recovery | tz+k − k-only ΔR² |
|---|---|---|---|---|---|---|---|
| HC | 1 | -0.055 ± 0.072 | 0.73 ± 0.04 | 85 ± 10 | -0.058 ± 0.213 | 0.73 ± 0.12 | 0.003 ± 0.204 |
| HC | 2 | 0.097 ± 0.023 | 0.82 ± 0.01 | 165 ± 12 | -0.007 ± 0.147 | 0.76 ± 0.08 | 0.104 ± 0.165 |
| HC | 5 | 0.206 ± 0.024 | 0.88 ± 0.01 | 401 ± 61 | 0.221 ± 0.037 | 0.88 ± 0.02 | -0.015 ± 0.016 |
| HC | 10 | 0.303 ± 0.022 | 0.93 ± 0.01 | 798 ± 49 | 0.302 ± 0.029 | 0.93 ± 0.02 | 0.001 ± 0.034 |
| HC | 20 | 0.359 ± 0.010 | 0.96 ± 0.01 | 1617 ± 38 | 0.355 ± 0.015 | 0.96 ± 0.01 | 0.003 ± 0.006 |
| AEF | 1 | -0.106 ± 0.181 | 0.83 ± 0.05 | 85 ± 10 | -0.354 ± 0.451 | 0.76 ± 0.13 | 0.248 ± 0.285 |
| AEF | 2 | 0.164 ± 0.058 | 0.91 ± 0.02 | 165 ± 12 | 0.112 ± 0.023 | 0.89 ± 0.01 | 0.051 ± 0.065 |
| AEF | 5 | 0.267 ± 0.031 | 0.94 ± 0.01 | 401 ± 61 | 0.218 ± 0.035 | 0.92 ± 0.01 | 0.050 ± 0.038 |
| AEF | 10 | 0.362 ± 0.034 | 0.96 ± 0.01 | 798 ± 49 | 0.338 ± 0.045 | 0.96 ± 0.01 | 0.025 ± 0.011 |
| AEF | 20 | 0.403 ± 0.031 | 0.97 ± 0.01 | 1617 ± 38 | 0.395 ± 0.041 | 0.97 ± 0.01 | 0.008 ± 0.010 |
| HC+AEF | 1 | -0.066 ± 0.096 | 0.72 ± 0.05 | 85 ± 10 | -0.265 ± 0.398 | 0.63 ± 0.19 | 0.198 ± 0.326 |
| HC+AEF | 2 | 0.114 ± 0.015 | 0.81 ± 0.01 | 165 ± 12 | 0.034 ± 0.190 | 0.77 ± 0.09 | 0.080 ± 0.191 |
| HC+AEF | 5 | 0.261 ± 0.025 | 0.87 ± 0.01 | 401 ± 61 | 0.229 ± 0.048 | 0.86 ± 0.02 | 0.031 ± 0.062 |
| HC+AEF | 10 | 0.379 ± 0.024 | 0.93 ± 0.01 | 798 ± 49 | 0.360 ± 0.042 | 0.92 ± 0.02 | 0.019 ± 0.020 |
| HC+AEF | 20 | 0.437 ± 0.017 | 0.96 ± 0.01 | 1617 ± 38 | 0.433 ± 0.018 | 0.96 ± 0.01 | 0.004 ± 0.000 |

## E4: cross-year transfer

One partition per site over the union of both years' blocks; A→B trains on year A's training-block cells and tests on year B's test-block cells; the baseline B→B trains on year B's training-block cells and tests on the same cells. Temporal loss = (B→B) − (A→B) pooled R² within a repeat; mean ± sd over 5 repeats. Colombia 2021→2022: 'overlap' (main) tests only on 2022 cells that also exist in 2021, 'full' on all 2022 test cells. 'matched' (Colombia, both directions): both years train only on cells present in both years inside the training blocks and are tested on the same shared test cells, so cross-year and baseline differ only in the year (equal training-set size and footprint). Use 'matched' for the Colombia temporal loss; 'overlap'/'full' confound year with the 3–4× larger 2022 training set.

| direction | FS | baseline B→B R² | cross-year A→B R² | temporal loss ΔR² | baseline RMSE | cross RMSE | cross bias (m) | n test |
|---|---|---|---|---|---|---|---|---|
| Colombia 2021 → 2022 (overlap) | HC | 0.445 ± 0.011 | 0.292 ± 0.021 | 0.153 ± 0.023 | 4.79 ± 0.05 | 5.41 ± 0.08 | -1.25 ± 0.19 | 2198 |
| Colombia 2021 → 2022 (overlap) | AEF | 0.503 ± 0.005 | 0.387 ± 0.013 | 0.116 ± 0.013 | 4.53 ± 0.02 | 5.04 ± 0.05 | -1.21 ± 0.10 | 2198 |
| Colombia 2021 → 2022 (overlap) | HC+AEF | 0.536 ± 0.008 | 0.404 ± 0.016 | 0.132 ± 0.013 | 4.38 ± 0.04 | 4.96 ± 0.07 | -1.12 ± 0.06 | 2198 |
| Colombia 2021 → 2022 (full) | HC | 0.432 ± 0.005 | 0.298 ± 0.013 | 0.134 ± 0.013 | 4.88 ± 0.02 | 5.42 ± 0.05 | -0.97 ± 0.14 | 7744 |
| Colombia 2021 → 2022 (full) | AEF | 0.501 ± 0.003 | 0.285 ± 0.011 | 0.215 ± 0.014 | 4.57 ± 0.02 | 5.47 ± 0.04 | -1.96 ± 0.08 | 7744 |
| Colombia 2021 → 2022 (full) | HC+AEF | 0.526 ± 0.004 | 0.338 ± 0.008 | 0.188 ± 0.008 | 4.46 ± 0.02 | 5.27 ± 0.03 | -1.65 ± 0.06 | 7744 |
| Colombia 2022 → 2021 | HC | 0.351 ± 0.007 | 0.337 ± 0.022 | 0.014 ± 0.019 | 5.16 ± 0.03 | 5.22 ± 0.08 | 0.78 ± 0.16 | 2212 |
| Colombia 2022 → 2021 | AEF | 0.535 ± 0.006 | 0.437 ± 0.010 | 0.097 ± 0.011 | 4.37 ± 0.03 | 4.81 ± 0.04 | 1.69 ± 0.06 | 2212 |
| Colombia 2022 → 2021 | HC+AEF | 0.535 ± 0.005 | 0.460 ± 0.007 | 0.075 ± 0.008 | 4.37 ± 0.02 | 4.71 ± 0.03 | 1.49 ± 0.12 | 2212 |
| Colombia 2021 → 2022 (matched) | HC | 0.409 ± 0.005 | 0.277 ± 0.017 | 0.132 ± 0.020 | 4.95 ± 0.02 | 5.47 ± 0.07 | -1.29 ± 0.14 | 2198 |
| Colombia 2021 → 2022 (matched) | AEF | 0.469 ± 0.012 | 0.373 ± 0.010 | 0.096 ± 0.014 | 4.69 ± 0.05 | 5.10 ± 0.04 | -1.21 ± 0.12 | 2198 |
| Colombia 2021 → 2022 (matched) | HC+AEF | 0.506 ± 0.010 | 0.408 ± 0.009 | 0.098 ± 0.010 | 4.52 ± 0.05 | 4.95 ± 0.04 | -1.12 ± 0.11 | 2198 |
| Colombia 2022 → 2021 (matched) | HC | 0.345 ± 0.007 | 0.310 ± 0.006 | 0.035 ± 0.013 | 5.19 ± 0.03 | 5.32 ± 0.02 | 0.48 ± 0.10 | 2198 |
| Colombia 2022 → 2021 (matched) | AEF | 0.529 ± 0.005 | 0.419 ± 0.012 | 0.111 ± 0.009 | 4.40 ± 0.02 | 4.89 ± 0.05 | 1.34 ± 0.14 | 2198 |
| Colombia 2022 → 2021 (matched) | HC+AEF | 0.533 ± 0.009 | 0.439 ± 0.010 | 0.095 ± 0.015 | 4.38 ± 0.04 | 4.80 ± 0.04 | 1.09 ± 0.17 | 2198 |
| Tanzania 2022 → 2024 | HC | 0.595 ± 0.011 | 0.544 ± 0.011 | 0.051 ± 0.017 | 2.59 ± 0.03 | 2.75 ± 0.03 | -0.88 ± 0.03 | 2212 |
| Tanzania 2022 → 2024 | AEF | 0.695 ± 0.007 | 0.459 ± 0.056 | 0.237 ± 0.055 | 2.24 ± 0.03 | 2.99 ± 0.15 | -1.09 ± 0.22 | 2212 |
| Tanzania 2022 → 2024 | HC+AEF | 0.692 ± 0.008 | 0.556 ± 0.018 | 0.136 ± 0.012 | 2.26 ± 0.03 | 2.71 ± 0.06 | -1.01 ± 0.03 | 2212 |
| Tanzania 2024 → 2022 | HC | 0.653 ± 0.009 | 0.545 ± 0.018 | 0.109 ± 0.013 | 2.33 ± 0.03 | 2.67 ± 0.05 | 0.90 ± 0.03 | 2427 |
| Tanzania 2024 → 2022 | AEF | 0.734 ± 0.007 | 0.432 ± 0.067 | 0.302 ± 0.064 | 2.04 ± 0.03 | 2.98 ± 0.17 | 1.55 ± 0.27 | 2427 |
| Tanzania 2024 → 2022 | HC+AEF | 0.738 ± 0.010 | 0.453 ± 0.021 | 0.286 ± 0.023 | 2.03 ± 0.04 | 2.93 ± 0.06 | 1.67 ± 0.08 | 2427 |

## Caveats

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
