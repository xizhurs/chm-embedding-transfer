"""Model wrappers: metrics, in-fold SHAP selection, PLSR, covered-cell subset."""
import numpy as np
import pandas as pd

from hgt.experiments import PLSR, ShapSelectModel, cell_subset, make_xgb, metrics


def synthetic(n=400, p=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype("float32")
    y = 10 + 4 * X[:, 0] - 3 * X[:, 1] + 2 * X[:, 2] + rng.normal(scale=0.3, size=n)
    return X, y


def test_metrics_known_values():
    m = metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert m["n"] == 3 and np.isclose(m["bias"], 1 / 3) and np.isclose(m["mae"], 1 / 3)
    assert np.isclose(m["rmse"], np.sqrt(1 / 3)) and np.isclose(m["rrmse"], np.sqrt(1 / 3) / 2)
    assert np.isclose(m["r2"], 0.5)


def test_shap_mean_selection_finds_informative_features():
    X, y = synthetic()
    model = ShapSelectModel(50, 0, "mean", 30, 3000, final=lambda: make_xgb(50, 0)).fit(X, y)
    assert {0, 1, 2} <= set(model.selected_)
    assert len(model.selected_) < X.shape[1]
    assert model.predict(X).shape == (len(X),)
    imp = model.feature_importances_
    assert imp.shape == (X.shape[1],) and (imp[np.setdiff1d(np.arange(20), model.selected_)] == 0).all()


def test_shap_topk_selects_k():
    X, y = synthetic()
    model = ShapSelectModel(50, 0, "topk", 5, 3000, final=lambda: make_xgb(50, 0)).fit(X, y)
    assert len(model.selected_) == 5 and {0, 1, 2} <= set(model.selected_)


def test_selection_with_plsr_has_no_importances():
    X, y = synthetic()
    model = ShapSelectModel(50, 0, "mean", 30, 3000, final=lambda: PLSR(5, True)).fit(X, y)
    assert not hasattr(model, "feature_importances_")
    assert np.corrcoef(model.predict(X), y)[0, 1] > 0.95


def test_plsr_clips_negative_and_caps_components():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = 5 * X[:, 0]                                   # half of the targets negative
    m = PLSR(5, True).fit(X, y)
    assert m.model_.n_components == 2                 # capped at n_features
    assert (m.predict(X) >= 0).all()


def test_cell_subset_drops_cells_without_tessera():
    df = pd.DataFrame({"cell_id": [1, 2, 3], "hc_a": [1.0, 2.0, 3.0],
                       "tes_E000_mean": [0.1, np.nan, 0.3], "tes_E000_std": [0.1, np.nan, 0.2]})
    out = cell_subset(df)                             # config pipeline.cells = covered
    assert out["cell_id"].tolist() == [1, 3]
