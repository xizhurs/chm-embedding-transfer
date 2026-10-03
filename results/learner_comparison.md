# Learner comparison: xgboost vs tabicl

Identical splits, seeds, cells, buffer and in-fold feature selection; only the final learner differs. Δ = tabicl − xgboost (pooled R², mean over repeats).

## E1 in-domain R²

| test | feature_set | xgboost | tabicl | Δ |
|---|---|---|---|---|
| colombia_2021 | HC | 0.341 | 0.426 | 0.084 |
| colombia_2021 | AEF | 0.529 | 0.611 | 0.083 |
| colombia_2022 | HC | 0.470 | 0.538 | 0.068 |
| colombia_2022 | AEF | 0.491 | 0.553 | 0.062 |
| tanzania_2022 | HC | 0.640 | 0.676 | 0.036 |
| tanzania_2022 | AEF | 0.722 | 0.763 | 0.041 |
| tanzania_2024 | HC | 0.580 | 0.625 | 0.045 |
| tanzania_2024 | AEF | 0.686 | 0.727 | 0.041 |

## E2 cross-country (Tanzania 2022 → Colombia 2022)

| feature_set | xgboost r2 | xgboost bias | tabicl r2 | tabicl bias | Δ r2 |
|---|---|---|---|---|---|
| AEF | -2.983 | -11.370 | -4.534 | -13.880 | -1.551 |
| HC | -0.918 | -6.844 | -1.265 | -7.846 | -0.346 |

## E3 few-shot R² (mean over the 1 draw(s) run by both learners)

| variant | k | feature_set | xgboost | tabicl | Δ |
|---|---|---|---|---|---|
| tz+k | 1 | AEF | -0.123 | 0.191 | 0.314 |
| tz+k | 1 | HC | -0.090 | -0.087 | 0.003 |
| tz+k | 5 | AEF | 0.278 | 0.371 | 0.093 |
| tz+k | 5 | HC | 0.224 | 0.316 | 0.092 |
| tz+k | 20 | AEF | 0.388 | 0.464 | 0.076 |
| tz+k | 20 | HC | 0.368 | 0.453 | 0.085 |

## E4 cross-year: cross-year R² and temporal loss

| direction | variant | feature_set | tabicl cross | xgboost cross | tabicl loss | xgboost loss |
|---|---|---|---|---|---|---|
| colombia_2021 → colombia_2022 | full | AEF | 0.226 | 0.298 | 0.329 | 0.190 |
| colombia_2021 → colombia_2022 | full | HC | 0.452 | 0.354 | 0.083 | 0.107 |
| colombia_2021 → colombia_2022 | matched | AEF | 0.313 | 0.398 | 0.236 | 0.076 |
| colombia_2021 → colombia_2022 | matched | HC | 0.481 | 0.399 | 0.042 | 0.055 |
| colombia_2021 → colombia_2022 | overlap | AEF | 0.375 | 0.422 | 0.227 | 0.081 |
| colombia_2021 → colombia_2022 | overlap | HC | 0.487 | 0.411 | 0.086 | 0.099 |
| colombia_2022 → colombia_2021 | main | AEF | 0.362 | 0.334 | 0.225 | 0.164 |
| colombia_2022 → colombia_2021 | main | HC | 0.435 | 0.365 | 0.006 | -0.028 |
| colombia_2022 → colombia_2021 | matched | AEF | 0.428 | 0.373 | 0.183 | 0.150 |
| colombia_2022 → colombia_2021 | matched | HC | 0.476 | 0.365 | -0.000 | 0.022 |
| tanzania_2022 → tanzania_2024 | main | AEF | 0.406 | 0.418 | 0.330 | 0.264 |
| tanzania_2022 → tanzania_2024 | main | HC | 0.556 | 0.544 | 0.064 | 0.029 |
| tanzania_2024 → tanzania_2022 | main | AEF | 0.442 | 0.332 | 0.305 | 0.376 |
| tanzania_2024 → tanzania_2022 | main | HC | 0.605 | 0.529 | 0.066 | 0.098 |
