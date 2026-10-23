# Canopy height transfer with AlphaEarth embeddings

Predict canopy height (P95 of a lidar canopy height model, per 100 m cell) from satellite features, and test
how well the models **transfer geographically** (Tanzania → Colombia) and **temporally** (year A → year B).

The main question is whether AlphaEarth Foundations (AEF) embeddings transfer better than handcrafted
Sentinel-1/2 features (HC). HC is the baseline in every experiment below.

| | |
|---|---|
| Target | P95 of the lidar CHM (0.5 m) per 100 m cell; cells with < 50 % valid pixels are dropped |
| Sites | Colombia (2021, 2022), Tanzania (2022, 2024) |
| AEF | Google/DeepMind AlphaEarth Foundations Satellite Embedding v1, annual, 64 dims, 10 m |
| HC | 98 handcrafted features: Sentinel-2 medians/indices, Sentinel-1 backscatter statistics, 24 GLCM texture features |
| Model | XGBoost, fixed hyper-parameters, no tuning (200 trees, depth 6, lr 0.1) |
| Validation | 5-fold CV on 1 km spatial blocks, with a 100 m buffer between training and test cells |

Tessera embeddings and a TabICL learner are also implemented (`TES`, `HC+TES`, `model.learner: tabicl`) but are
not the focus here. The numbers below are XGBoost only.

## AEF features

AEF is read from the Source Cooperative mirror (`tge-labs/aef/v1/annual`) and de-quantised with
`((v / 127.5)² · sign(v))`, with NaN where the source is −128 (`src/hgt/download_aef.py`).

Each 100 m cell gets 128 AEF features (`src/hgt/features.py`):
- `aef_Axx_mean`: the mean of the 64-d embedding over the cell's valid 10 m pixels, re-normalised to unit length
- `aef_Axx_std`: the per-dimension standard deviation over the same pixels

Feature selection happens inside each training fold. A selector XGBoost is fitted, and features whose mean |SHAP|
is above the average are kept. That is roughly 25–30 of the 128 AEF features.

## Results (XGBoost, pooled out-of-fold R²)

### E1: in-domain blocked CV

| site-year | HC | AEF | AEF − HC |
|---|---|---|---|
| Colombia 2021 | 0.34 | 0.53 | +0.19 |
| Colombia 2022 | 0.47 | 0.49 | +0.02 |
| Tanzania 2022 | 0.64 | 0.72 | +0.08 |
| Tanzania 2024 | 0.58 | 0.69 | +0.11 |

AEF is better in-domain everywhere, but the gain is negligible in Colombia 2022.

### E2: geographic transfer (train Tanzania 2022, test Colombia 2022)

| | in-domain R² (E1) | transfer R² (E2) | RMSE (m) | bias (m) |
|---|---|---|---|---|
| HC | 0.47 | −0.92 | 8.98 | −6.8 |
| AEF | 0.49 | −2.98 | 12.94 | −11.4 |

Both fail, and AEF fails worse. The cause is the target range: Tanzania 2022 heights have a mean of 7.4 m and a 95th
percentile of 15.2 m, while Colombia 2022 has a mean of 17.8 m. 66 % of Colombian cells are above the Tanzanian P95.
Tree ensembles can't predict above the range they were trained on, so the error is dominated by a large negative bias.

### E3: few-shot recovery in Colombia 2022 (R² at k Colombia blocks)

| k | HC tz+k | HC k-only | AEF tz+k | AEF k-only |
|---|---|---|---|---|
| 1 | −0.01 | −0.04 | −0.04 | −0.34 |
| 2 | 0.19 | −0.01 | 0.18 | −0.15 |
| 5 | 0.23 | 0.21 | 0.26 | 0.21 |
| 10 | 0.30 | 0.31 | 0.34 | 0.32 |
| 20 | 0.36 | 0.37 | 0.38 | 0.38 |

`tz+k` is Tanzania 2022 plus k Colombia blocks, and `k-only` is the k Colombia blocks alone. Tanzania data only helps
at k ≤ 2. From k = 5 on, local data alone does as well. Three draws per k, repeat 0 only.

### E4: temporal transfer (R² as baseline → cross-year, loss = baseline − cross-year)

The baseline trains and tests on year B. The cross-year model trains on year A and tests on year B. For Colombia
the "matched" variant is used, where both models train and test on the same cells.

| direction | HC | HC loss | AEF | AEF loss |
|---|---|---|---|---|
| Colombia 2021 → 2022 | 0.454 → 0.399 | 0.055 | 0.474 → 0.398 | 0.076 |
| Colombia 2022 → 2021 | 0.387 → 0.365 | 0.022 | 0.523 → 0.373 | 0.150 |
| Tanzania 2022 → 2024 | 0.573 → 0.544 | 0.029 | 0.683 → 0.418 | 0.264 |
| Tanzania 2024 → 2022 | 0.628 → 0.529 | 0.098 | 0.708 → 0.332 | 0.376 |

HC loses little across years. AEF loses much more, so its in-domain advantage disappears: in absolute cross-year R²,
AEF is level with HC in Colombia and well behind it in Tanzania.

### Takeaways

- AEF beats HC in-domain, but that gain does not carry over to either kind of transfer.
- Geographic transfer between these two sites fails for both feature sets, mainly because of the height-range shift.
- Temporal transfer is where AEF is weakest. The Tanzania gap is 2 years and the Colombia gap is 1 year.

## Caveats

- **Cells.** `pipeline.cells: covered` keeps only cells with Tessera data, identically for every feature set. About 18–21 %
  of Colombia cells are dropped, so the Colombia numbers describe the less cloudy part of the site. Set `cells: all` to
  use every cell. Earlier all-cell results (HC, AEF, HC+AEF, no feature selection) are in
  `results/archive_xgb_allcells_noselect/`.
- **Noise.** One repeat only, and Colombia 2021 is small (1,745 cells, 37 blocks).
- **Temporal gap.** It differs between sites, and real canopy change, lidar acquisition differences and image-composite
  differences are all mixed into the temporal loss.
- **No tuning.** One fixed XGBoost configuration is used for every feature set, so rankings could change with tuning.
- **Spatial autocorrelation.** Beyond the 100 m buffer it can make in-domain scores somewhat optimistic.
- **Inference maps.** The in-sample R² in `results/xgboost/maps/maps_meta.csv` is a sanity check only, not an accuracy
  estimate.

## Repository layout

```
config.yaml            sites, grids, years, model and pipeline settings (single source of truth)
docs/CONTRACTS.md      file formats and geometry shared by all stages
src/hgt/
  download_aef.py      AlphaEarth annual embeddings -> data/raw/aef/
  download_pc.py       Sentinel-1/2 from Planetary Computer -> data/raw/{s1,s2}/
  download_tessera.py  Tessera v1.1 embeddings -> data/raw/tessera/
  targets.py           lidar CHM -> P95 height per 100 m cell
  features.py          per-cell feature tables (HC, AEF, TES), --maps for all grid cells
  splits.py            1 km block splits, buffer, few-shot blocks
  experiments.py       E1-E4
  report.py            figures and summary.md
  inference.py         wall-to-wall 100 m height maps
scripts/               check_splits.py, compare_learners.py, pilot_tabicl.py
results/xgboost/       summary.md, metrics.csv, figures/, maps/ (the results above)
results/archive_*/     earlier runs, each with a README
tests/                 pytest suite, run in CI (.github/workflows/tests.yml)
```

## Running it

The lidar CHMs (`data/Colombia/`, `data/Tanzania/`) and all downloaded data are gitignored and not in the repo.
Python 3.12 with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                                         # includes the TabICL/PyTorch group; use --no-group fm to skip it
uv run python -m hgt.download_aef               # all site-years; --site S --year Y [--overwrite] for one
uv run python -m hgt.download_pc                # HC inputs
uv run python -m hgt.download_tessera           # only needed for the TES feature sets
uv run python -m hgt.targets
uv run python -m hgt.features
uv run python -m hgt.experiments --exp all --fs HC AEF   # or E1 | E2 | E3 | E4 | E4matched; --quick for a smoke test
uv run python -m hgt.report
uv run python -m hgt.features --maps && uv run python -m hgt.inference --fs HC AEF
uv run pytest -q
```

Outputs go to `results/{model.learner}/`: `summary.md`, `metrics.csv` (long format), figures, and the maps.

## Data and attribution

The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind (CC-BY-4.0).
Tessera (Feng et al. 2025, arXiv:2506.20380) is CC-BY-SA, from the AWS Registry of Open Data.
