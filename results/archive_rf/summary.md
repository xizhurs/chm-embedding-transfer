# Canopy height (P95, 100 m): handcrafted S1/S2 vs AlphaEarth embeddings

Random Forest (n_estimators = 500, max_features = 0.3333, no tuning) on 1 km spatial block splits. FS: HC = 98 handcrafted S1/S2 features (incl. 24 GLCM), AEF = 128 AlphaEarth features (64 mean + 64 std), HC+AEF = both. Metrics: R², RMSE (m), rRMSE = RMSE / mean(obs), bias = mean(pred − obs). Figures in `results/figures/`.

## Key results

- E1 Colombia 2021: R² HC 0.36, AEF 0.51, HC+AEF 0.50; AEF − HC = 0.142 ± 0.003, HC+AEF − AEF = -0.003 ± 0.002.
- E1 Colombia 2022: R² HC 0.44, AEF 0.48, HC+AEF 0.50; AEF − HC = 0.040 ± 0.003, HC+AEF − AEF = 0.019 ± 0.001.
- E1 Tanzania 2022: R² HC 0.64, AEF 0.71, HC+AEF 0.72; AEF − HC = 0.073 ± 0.003, HC+AEF − AEF = 0.004 ± 0.003.
- E1 Tanzania 2024: R² HC 0.60, AEF 0.68, HC+AEF 0.67; AEF − HC = 0.074 ± 0.005, HC+AEF − AEF = -0.004 ± 0.004.
- E2 (Tanzania 2022 → Colombia 2022) fails for every FS: R² HC -1.05, AEF -2.64, HC+AEF -1.17; mean bias HC -7.1 m, AEF -10.6 m, HC+AEF -7.4 m.
- E3 at k = 20 blocks: tz+k R² HC 0.35, AEF 0.35, HC+AEF 0.38; k-only R² HC 0.35, AEF 0.35, HC+AEF 0.39. Mean tz+k − k-only ΔR² over all k and FS ranges from -0.118 to 0.035.
- E4 Colombia 2021 → 2022 (overlap): temporal loss ΔR² HC 0.078, AEF 0.078, HC+AEF 0.087.
- E4 Colombia 2021 → 2022 (full): temporal loss ΔR² HC 0.085, AEF 0.139, HC+AEF 0.117.
- E4 Colombia 2022 → 2021: temporal loss ΔR² HC -0.005, AEF 0.062, HC+AEF 0.065.
- E4 Tanzania 2022 → 2024: temporal loss ΔR² HC 0.041, AEF 0.227, HC+AEF 0.101.
- E4 Tanzania 2024 → 2022: temporal loss ΔR² HC 0.100, AEF 0.401, HC+AEF 0.250.

## E1: in-domain blocked CV

Pooled out-of-fold metrics per repeat, mean ± sd over 5 repeats (5 folds of 1 km blocks, folds balanced by cell count).

| site-year | n cells | FS | R² | RMSE (m) | rRMSE | bias (m) |
|---|---|---|---|---|---|---|
| Colombia 2021 | 2212 | HC | 0.365 ± 0.006 | 5.11 ± 0.02 | 0.296 ± 0.001 | 0.11 ± 0.03 |
| Colombia 2021 | 2212 | AEF | 0.506 ± 0.006 | 4.50 ± 0.03 | 0.261 ± 0.002 | 0.04 ± 0.05 |
| Colombia 2021 | 2212 | HC+AEF | 0.503 ± 0.005 | 4.52 ± 0.02 | 0.262 ± 0.001 | 0.07 ± 0.04 |
| Colombia 2022 | 7744 | HC | 0.439 ± 0.002 | 4.84 ± 0.01 | 0.273 ± 0.001 | 0.07 ± 0.01 |
| Colombia 2022 | 7744 | AEF | 0.479 ± 0.003 | 4.67 ± 0.01 | 0.263 ± 0.001 | 0.11 ± 0.03 |
| Colombia 2022 | 7744 | HC+AEF | 0.498 ± 0.002 | 4.58 ± 0.01 | 0.258 ± 0.001 | 0.09 ± 0.03 |
| Tanzania 2022 | 2427 | HC | 0.640 ± 0.004 | 2.38 ± 0.01 | 0.324 ± 0.002 | 0.08 ± 0.01 |
| Tanzania 2022 | 2427 | AEF | 0.713 ± 0.002 | 2.12 ± 0.01 | 0.289 ± 0.001 | 0.09 ± 0.01 |
| Tanzania 2022 | 2427 | HC+AEF | 0.717 ± 0.004 | 2.11 ± 0.01 | 0.287 ± 0.002 | 0.08 ± 0.01 |
| Tanzania 2024 | 2212 | HC | 0.603 ± 0.005 | 2.56 ± 0.02 | 0.354 ± 0.002 | 0.14 ± 0.02 |
| Tanzania 2024 | 2212 | AEF | 0.677 ± 0.003 | 2.31 ± 0.01 | 0.319 ± 0.001 | 0.13 ± 0.02 |
| Tanzania 2024 | 2212 | HC+AEF | 0.673 ± 0.006 | 2.32 ± 0.02 | 0.321 ± 0.003 | 0.13 ± 0.02 |

Paired difference in pooled R² (same partition within each repeat), mean ± sd over repeats:

| site-year | AEF − HC | HC+AEF − AEF | HC+AEF − HC |
|---|---|---|---|
| Colombia 2021 | 0.142 ± 0.003 | -0.003 ± 0.002 | 0.138 ± 0.002 |
| Colombia 2022 | 0.040 ± 0.003 | 0.019 ± 0.001 | 0.059 ± 0.002 |
| Tanzania 2022 | 0.073 ± 0.003 | 0.004 ± 0.003 | 0.077 ± 0.004 |
| Tanzania 2024 | 0.074 ± 0.005 | -0.004 ± 0.004 | 0.070 ± 0.004 |

Per-fold R² ranges from 0.16 to 0.77; the lowest single fold is 0.16 (Colombia 2021, HC).

## E2: cross-country transfer (train Tanzania 2022 → test Colombia 2022)

One model per FS and repeat fitted on all Tanzania 2022 cells, evaluated on all Colombia 2022 cells (mean ± sd over 5 repeats). Transfer loss = E1 − E2 pooled R² within the same repeat; 'per-fold ΔR²' uses the matched E1 test folds (mean ± sd over repeats × folds).

| FS | E1 R² (CO22 in-domain) | E2 R² (TZ22 → CO22) | transfer loss ΔR² (E1 − E2) | per-fold ΔR² | E1 RMSE | E2 RMSE | ΔRMSE (E2 − E1) | E2 bias (m) |
|---|---|---|---|---|---|---|---|---|
| HC | 0.439 ± 0.002 | -1.047 ± 0.025 | 1.487 ± 0.024 | 1.508 ± 0.216 | 4.84 ± 0.01 | 9.26 ± 0.06 | 4.41 ± 0.05 | -7.08 ± 0.07 |
| AEF | 0.479 ± 0.003 | -2.637 ± 0.153 | 3.117 ± 0.151 | 3.161 ± 0.443 | 4.67 ± 0.01 | 12.34 ± 0.26 | 7.67 ± 0.25 | -10.60 ± 0.31 |
| HC+AEF | 0.498 ± 0.002 | -1.174 ± 0.042 | 1.672 ± 0.043 | 1.697 ± 0.276 | 4.58 ± 0.01 | 9.54 ± 0.09 | 4.96 ± 0.10 | -7.35 ± 0.11 |

Target ranges differ strongly: Tanzania 2022 height mean 7.4 m (P95 15.2, max 28.4); Colombia 2022 mean 17.8 m (P95 27.9, max 62.5). 4% of Colombia cells are taller than the tallest Tanzania cell and 66% exceed the Tanzania P95. A Random Forest cannot predict outside its training target range, so E2 is dominated by a large negative bias; negative R² here means worse than predicting the Colombia mean.

## E3: few-shot recovery in Colombia 2022

Repeat 0 only (the E1 colombia-2022 partition of repeat 0). For each of the 5 folds, k shot blocks are drawn from that fold's training blocks (2 draws per k); models are tested on the fold's test blocks and pooled over the 5 folds. Values: mean ± sd over draws. 'shot cells/fold' = mean number of Colombia cells in the k shot blocks (blocks hold 1–100 cells).

Reference levels (pooled, repeat 0):

| FS | E1 R² (r0) | E2 R² (r0) |
|---|---|---|
| HC | 0.437 | -1.051 |
| AEF | 0.474 | -2.777 |
| HC+AEF | 0.494 | -1.142 |

| FS | k | tz+k R² | tz+k recovery | shot cells/fold | k-only R² | k-only recovery | tz+k − k-only ΔR² |
|---|---|---|---|---|---|---|---|
| HC | 1 | -0.025 ± 0.076 | 0.69 ± 0.05 | 89 ± 11 | 0.085 ± 0.116 | 0.76 ± 0.08 | -0.109 ± 0.040 |
| HC | 2 | 0.109 ± 0.001 | 0.78 ± 0.00 | 159 ± 9 | 0.140 ± 0.052 | 0.80 ± 0.04 | -0.031 ± 0.051 |
| HC | 5 | 0.208 ± 0.032 | 0.85 ± 0.02 | 426 ± 58 | 0.230 ± 0.028 | 0.86 ± 0.02 | -0.022 ± 0.004 |
| HC | 10 | 0.303 ± 0.011 | 0.91 ± 0.01 | 776 ± 43 | 0.314 ± 0.012 | 0.92 ± 0.01 | -0.011 ± 0.001 |
| HC | 20 | 0.354 ± 0.015 | 0.94 ± 0.01 | 1602 ± 38 | 0.352 ± 0.012 | 0.94 ± 0.01 | 0.002 ± 0.003 |
| AEF | 1 | -0.118 ± 0.330 | 0.82 ± 0.10 | 89 ± 11 | -0.045 ± 0.169 | 0.84 ± 0.05 | -0.073 ± 0.161 |
| AEF | 2 | 0.148 ± 0.073 | 0.90 ± 0.02 | 159 ± 9 | 0.129 ± 0.015 | 0.89 ± 0.00 | 0.019 ± 0.058 |
| AEF | 5 | 0.255 ± 0.000 | 0.93 ± 0.00 | 426 ± 58 | 0.220 ± 0.032 | 0.92 ± 0.01 | 0.035 ± 0.032 |
| AEF | 10 | 0.363 ± 0.004 | 0.97 ± 0.00 | 776 ± 43 | 0.345 ± 0.015 | 0.96 ± 0.00 | 0.018 ± 0.010 |
| AEF | 20 | 0.351 ± 0.058 | 0.96 ± 0.02 | 1602 ± 38 | 0.346 ± 0.068 | 0.96 ± 0.02 | 0.005 ± 0.010 |
| HC+AEF | 1 | -0.055 ± 0.134 | 0.66 ± 0.08 | 89 ± 11 | 0.063 ± 0.131 | 0.74 ± 0.08 | -0.118 ± 0.003 |
| HC+AEF | 2 | 0.146 ± 0.017 | 0.79 ± 0.01 | 159 ± 9 | 0.186 ± 0.039 | 0.81 ± 0.02 | -0.040 ± 0.022 |
| HC+AEF | 5 | 0.238 ± 0.009 | 0.84 ± 0.01 | 426 ± 58 | 0.250 ± 0.031 | 0.85 ± 0.02 | -0.013 ± 0.021 |
| HC+AEF | 10 | 0.348 ± 0.011 | 0.91 ± 0.01 | 776 ± 43 | 0.371 ± 0.005 | 0.92 ± 0.00 | -0.023 ± 0.005 |
| HC+AEF | 20 | 0.380 ± 0.045 | 0.93 ± 0.03 | 1602 ± 38 | 0.387 ± 0.034 | 0.93 ± 0.02 | -0.007 ± 0.012 |

## E4: cross-year transfer

One partition per site over the union of both years' blocks; A→B trains on year A's training-block cells and tests on year B's test-block cells; the baseline B→B trains on year B's training-block cells and tests on the same cells. Temporal loss = (B→B) − (A→B) pooled R² within a repeat; mean ± sd over 1 repeats. Colombia 2021→2022: 'overlap' (main) tests only on 2022 cells that also exist in 2021, 'full' on all 2022 test cells.

| direction | FS | baseline B→B R² | cross-year A→B R² | temporal loss ΔR² | baseline RMSE | cross RMSE | cross bias (m) | n test |
|---|---|---|---|---|---|---|---|---|
| Colombia 2021 → 2022 (overlap) | HC | 0.435 ± nan | 0.356 ± nan | 0.078 ± nan | 4.84 ± nan | 5.16 ± nan | -0.69 ± nan | 2198 |
| Colombia 2021 → 2022 (overlap) | AEF | 0.472 ± nan | 0.394 ± nan | 0.078 ± nan | 4.67 ± nan | 5.01 ± nan | -0.68 ± nan | 2198 |
| Colombia 2021 → 2022 (overlap) | HC+AEF | 0.502 ± nan | 0.414 ± nan | 0.087 ± nan | 4.54 ± nan | 4.92 ± nan | -0.65 ± nan | 2198 |
| Colombia 2021 → 2022 (full) | HC | 0.422 ± nan | 0.337 ± nan | 0.085 ± nan | 4.92 ± nan | 5.27 ± nan | -0.42 ± nan | 7744 |
| Colombia 2021 → 2022 (full) | AEF | 0.461 ± nan | 0.322 ± nan | 0.139 ± nan | 4.75 ± nan | 5.33 ± nan | -1.25 ± nan | 7744 |
| Colombia 2021 → 2022 (full) | HC+AEF | 0.482 ± nan | 0.365 ± nan | 0.117 ± nan | 4.65 ± nan | 5.16 ± nan | -0.94 ± nan | 7744 |
| Colombia 2022 → 2021 | HC | 0.354 ± nan | 0.359 ± nan | -0.005 ± nan | 5.15 ± nan | 5.13 ± nan | 0.41 ± nan | 2212 |
| Colombia 2022 → 2021 | AEF | 0.498 ± nan | 0.436 ± nan | 0.062 ± nan | 4.54 ± nan | 4.81 ± nan | 1.09 ± nan | 2212 |
| Colombia 2022 → 2021 | HC+AEF | 0.491 ± nan | 0.426 ± nan | 0.065 ± nan | 4.57 ± nan | 4.86 ± nan | 0.99 ± nan | 2212 |
| Tanzania 2022 → 2024 | HC | 0.584 ± nan | 0.543 ± nan | 0.041 ± nan | 2.62 ± nan | 2.75 ± nan | -0.69 ± nan | 2212 |
| Tanzania 2022 → 2024 | AEF | 0.659 ± nan | 0.432 ± nan | 0.227 ± nan | 2.37 ± nan | 3.06 ± nan | -1.13 ± nan | 2212 |
| Tanzania 2022 → 2024 | HC+AEF | 0.656 ± nan | 0.555 ± nan | 0.101 ± nan | 2.39 ± nan | 2.71 ± nan | -0.85 ± nan | 2212 |
| Tanzania 2024 → 2022 | HC | 0.633 ± nan | 0.533 ± nan | 0.100 ± nan | 2.40 ± nan | 2.71 ± nan | 1.04 ± nan | 2427 |
| Tanzania 2024 → 2022 | AEF | 0.700 ± nan | 0.299 ± nan | 0.401 ± nan | 2.17 ± nan | 3.32 ± nan | 2.02 ± nan | 2427 |
| Tanzania 2024 → 2022 | HC+AEF | 0.700 ± nan | 0.450 ± nan | 0.250 ± nan | 2.17 ± nan | 2.94 ± nan | 1.63 ± nan | 2427 |

## Caveats

- **E3 uses repeat 0 only** (one block partition); its sd is over draws of shot blocks, not over partitions. 2 draws per k were run (config: 10).
- **Recovery is sensitive to the E2 floor.** Where E2 R² is strongly negative (systematic bias from transferring a model trained on much lower Tanzanian canopies), even a weak model trained on a few Colombia blocks removes most of the bias and scores a high 'recovery'. Compare tz+k against k-only, and both against the E1 ceiling, rather than reading recovery alone.
- **Cross-year gaps differ:** 1 year for Colombia (2021→2022) and 2 years for Tanzania (2022→2024). Real canopy change, lidar acquisition differences and image-composite differences are all mixed into the temporal loss.
- **S1 acquisition density differs between site-years** (S1A descending only; Colombia 2021 has only 22 S1A scenes), so the S1 part of HC is noisier in some years. AEF annual embeddings are built from their own sensor stack and are not affected the same way.
- **Spatial autocorrelation across block edges:** test blocks touch training blocks without a buffer, so cells near block edges have correlated neighbours in training. This affects every FS equally but makes all in-domain scores somewhat optimistic.
- **Uneven block sizes** (1–100 cells): folds are balanced by cell count, but the number of blocks per fold differs, and few-shot k blocks contain a variable number of cells (reported as shot cells).
- Colombia 2021 is small (2212 cells, 37 blocks) and lies inside the 2022 footprint; E1 numbers for it are noisier.
- Impurity importances are biased toward correlated / continuous features and are descriptive only.
- No hyper-parameter tuning; one fixed RF for all feature sets. Rankings could change with tuning or another learner.

## Attribution

The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind.
