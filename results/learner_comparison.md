# Learner comparison: xgboost vs plsr

Identical splits, seeds, cells, buffer and in-fold feature selection; only the final learner differs. Δ = plsr − xgboost (pooled R², mean over repeats).

## E1 in-domain R²

| test | feature_set | xgboost | plsr | Δ |
|---|---|---|---|---|
| colombia_2021 | HC | 0.341 | 0.298 | -0.044 |
| colombia_2021 | AEF | 0.529 | 0.528 | -0.001 |
| colombia_2021 | HC+AEF | 0.533 | 0.541 | 0.008 |
| colombia_2021 | TES | 0.501 | 0.505 | 0.003 |
| colombia_2021 | HC+TES | 0.504 | 0.518 | 0.015 |
| colombia_2022 | HC | 0.470 | 0.358 | -0.112 |
| colombia_2022 | AEF | 0.491 | 0.498 | 0.007 |
| colombia_2022 | HC+AEF | 0.543 | 0.504 | -0.039 |
| colombia_2022 | TES | 0.584 | 0.571 | -0.013 |
| colombia_2022 | HC+TES | 0.592 | 0.577 | -0.015 |
| tanzania_2022 | HC | 0.640 | 0.583 | -0.057 |
| tanzania_2022 | AEF | 0.722 | 0.704 | -0.018 |
| tanzania_2022 | HC+AEF | 0.735 | 0.718 | -0.017 |
| tanzania_2022 | TES | 0.646 | 0.629 | -0.018 |
| tanzania_2022 | HC+TES | 0.685 | 0.647 | -0.038 |
| tanzania_2024 | HC | 0.580 | 0.547 | -0.033 |
| tanzania_2024 | AEF | 0.686 | 0.700 | 0.015 |
| tanzania_2024 | HC+AEF | 0.686 | 0.703 | 0.017 |
| tanzania_2024 | TES | 0.622 | 0.610 | -0.011 |
| tanzania_2024 | HC+TES | 0.650 | 0.623 | -0.027 |

## E2 cross-country (Tanzania 2022 → Colombia 2022)

| feature_set | xgboost r2 | xgboost bias | plsr r2 | plsr bias | Δ r2 |
|---|---|---|---|---|---|
| AEF | -2.983 | -11.370 | -3.331 | -12.350 | -0.348 |
| HC | -0.918 | -6.844 | -0.733 | -5.643 | 0.185 |
| HC+AEF | -1.436 | -8.242 | -3.474 | -12.668 | -2.037 |
| HC+TES | -1.030 | -7.145 | -0.707 | -6.027 | 0.323 |
| TES | -0.895 | -6.502 | -1.939 | -9.092 | -1.044 |

## E3 few-shot R² (mean over draws)

| variant | k | feature_set | xgboost | plsr | Δ |
|---|---|---|---|---|---|
| k-only | 1 | AEF | -0.341 | -0.517 | -0.176 |
| k-only | 1 | HC | -0.036 | -0.434 | -0.398 |
| k-only | 1 | HC+AEF | -0.299 | -0.610 | -0.311 |
| k-only | 1 | HC+TES | 0.080 | -0.446 | -0.526 |
| k-only | 1 | TES | 0.035 | -0.268 | -0.303 |
| k-only | 2 | AEF | -0.147 | 0.136 | 0.283 |
| k-only | 2 | HC | -0.009 | -0.108 | -0.099 |
| k-only | 2 | HC+AEF | -0.068 | 0.046 | 0.114 |
| k-only | 2 | HC+TES | 0.005 | -0.031 | -0.035 |
| k-only | 2 | TES | 0.045 | 0.130 | 0.085 |
| k-only | 5 | AEF | 0.209 | 0.337 | 0.128 |
| k-only | 5 | HC | 0.209 | 0.149 | -0.060 |
| k-only | 5 | HC+AEF | 0.240 | 0.307 | 0.066 |
| k-only | 5 | HC+TES | 0.151 | 0.405 | 0.255 |
| k-only | 5 | TES | 0.063 | 0.383 | 0.320 |
| k-only | 10 | AEF | 0.315 | 0.413 | 0.098 |
| k-only | 10 | HC | 0.310 | 0.258 | -0.052 |
| k-only | 10 | HC+AEF | 0.376 | 0.405 | 0.029 |
| k-only | 10 | HC+TES | 0.408 | 0.490 | 0.082 |
| k-only | 10 | TES | 0.393 | 0.475 | 0.082 |
| k-only | 20 | AEF | 0.376 | 0.452 | 0.076 |
| k-only | 20 | HC | 0.368 | 0.308 | -0.060 |
| k-only | 20 | HC+AEF | 0.426 | 0.465 | 0.039 |
| k-only | 20 | HC+TES | 0.506 | 0.532 | 0.026 |
| k-only | 20 | TES | 0.478 | 0.522 | 0.043 |
| tz+k | 1 | AEF | -0.042 | 0.266 | 0.308 |
| tz+k | 1 | HC | -0.010 | -0.105 | -0.095 |
| tz+k | 1 | HC+AEF | 0.058 | 0.171 | 0.113 |
| tz+k | 1 | HC+TES | 0.015 | -0.009 | -0.024 |
| tz+k | 1 | TES | -0.033 | -0.104 | -0.071 |
| tz+k | 2 | AEF | 0.177 | 0.336 | 0.159 |
| tz+k | 2 | HC | 0.190 | 0.113 | -0.077 |
| tz+k | 2 | HC+AEF | 0.222 | 0.297 | 0.075 |
| tz+k | 2 | HC+TES | 0.191 | 0.088 | -0.103 |
| tz+k | 2 | TES | 0.118 | 0.046 | -0.072 |
| tz+k | 5 | AEF | 0.258 | 0.362 | 0.104 |
| tz+k | 5 | HC | 0.230 | 0.193 | -0.037 |
| tz+k | 5 | HC+AEF | 0.290 | 0.345 | 0.055 |
| tz+k | 5 | HC+TES | 0.318 | 0.206 | -0.112 |
| tz+k | 5 | TES | 0.249 | 0.173 | -0.076 |
| tz+k | 10 | AEF | 0.337 | 0.399 | 0.063 |
| tz+k | 10 | HC | 0.296 | 0.232 | -0.064 |
| tz+k | 10 | HC+AEF | 0.401 | 0.401 | -0.000 |
| tz+k | 10 | HC+TES | 0.439 | 0.346 | -0.093 |
| tz+k | 10 | TES | 0.418 | 0.337 | -0.081 |
| tz+k | 20 | AEF | 0.379 | 0.440 | 0.061 |
| tz+k | 20 | HC | 0.359 | 0.254 | -0.106 |
| tz+k | 20 | HC+AEF | 0.423 | 0.438 | 0.015 |
| tz+k | 20 | HC+TES | 0.495 | 0.444 | -0.050 |
| tz+k | 20 | TES | 0.479 | 0.413 | -0.066 |

## E4 cross-year: cross-year R² and temporal loss

| direction | variant | feature_set | plsr cross | xgboost cross | plsr loss | xgboost loss |
|---|---|---|---|---|---|---|
| colombia_2021 → colombia_2022 | full | AEF | 0.084 | 0.298 | 0.413 | 0.190 |
| colombia_2021 → colombia_2022 | full | HC | 0.038 | 0.354 | 0.311 | 0.107 |
| colombia_2021 → colombia_2022 | full | HC+AEF | 0.146 | 0.366 | 0.362 | 0.183 |
| colombia_2021 → colombia_2022 | full | HC+TES | -0.007 | 0.375 | 0.583 | 0.213 |
| colombia_2021 → colombia_2022 | full | TES | 0.039 | 0.332 | 0.526 | 0.248 |
| colombia_2021 → colombia_2022 | matched | AEF | 0.154 | 0.398 | 0.320 | 0.076 |
| colombia_2021 → colombia_2022 | matched | HC | 0.101 | 0.399 | 0.294 | 0.055 |
| colombia_2021 → colombia_2022 | matched | HC+AEF | 0.182 | 0.464 | 0.303 | 0.056 |
| colombia_2021 → colombia_2022 | matched | HC+TES | 0.312 | 0.478 | 0.241 | 0.063 |
| colombia_2021 → colombia_2022 | matched | TES | 0.291 | 0.446 | 0.193 | 0.081 |
| colombia_2021 → colombia_2022 | overlap | AEF | 0.156 | 0.422 | 0.353 | 0.081 |
| colombia_2021 → colombia_2022 | overlap | HC | 0.086 | 0.411 | 0.306 | 0.099 |
| colombia_2021 → colombia_2022 | overlap | HC+AEF | 0.263 | 0.484 | 0.279 | 0.106 |
| colombia_2021 → colombia_2022 | overlap | HC+TES | 0.185 | 0.490 | 0.394 | 0.120 |
| colombia_2021 → colombia_2022 | overlap | TES | 0.198 | 0.445 | 0.341 | 0.153 |
| colombia_2022 → colombia_2021 | main | AEF | 0.401 | 0.334 | 0.101 | 0.164 |
| colombia_2022 → colombia_2021 | main | HC | 0.178 | 0.365 | 0.128 | -0.028 |
| colombia_2022 → colombia_2021 | main | HC+AEF | 0.476 | 0.430 | 0.040 | 0.087 |
| colombia_2022 → colombia_2021 | main | HC+TES | 0.270 | 0.426 | 0.254 | 0.078 |
| colombia_2022 → colombia_2021 | main | TES | 0.297 | 0.376 | 0.208 | 0.111 |
| colombia_2022 → colombia_2021 | matched | AEF | 0.390 | 0.373 | 0.158 | 0.150 |
| colombia_2022 → colombia_2021 | matched | HC | 0.276 | 0.365 | 0.066 | 0.022 |
| colombia_2022 → colombia_2021 | matched | HC+AEF | 0.408 | 0.396 | 0.120 | 0.138 |
| colombia_2022 → colombia_2021 | matched | HC+TES | 0.418 | 0.451 | 0.160 | 0.097 |
| colombia_2022 → colombia_2021 | matched | TES | 0.385 | 0.424 | 0.166 | 0.110 |
| tanzania_2022 → tanzania_2024 | main | AEF | 0.129 | 0.418 | 0.566 | 0.264 |
| tanzania_2022 → tanzania_2024 | main | HC | 0.493 | 0.544 | 0.051 | 0.029 |
| tanzania_2022 → tanzania_2024 | main | HC+AEF | 0.371 | 0.560 | 0.335 | 0.115 |
| tanzania_2022 → tanzania_2024 | main | HC+TES | 0.546 | 0.575 | 0.086 | 0.073 |
| tanzania_2022 → tanzania_2024 | main | TES | 0.539 | 0.460 | 0.069 | 0.154 |
| tanzania_2024 → tanzania_2022 | main | AEF | 0.387 | 0.332 | 0.324 | 0.376 |
| tanzania_2024 → tanzania_2022 | main | HC | 0.501 | 0.529 | 0.078 | 0.098 |
| tanzania_2024 → tanzania_2022 | main | HC+AEF | 0.523 | 0.297 | 0.202 | 0.422 |
| tanzania_2024 → tanzania_2022 | main | HC+TES | 0.472 | 0.599 | 0.178 | 0.076 |
| tanzania_2024 → tanzania_2022 | main | TES | 0.453 | 0.491 | 0.181 | 0.157 |
