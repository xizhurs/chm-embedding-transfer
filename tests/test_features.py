"""Per-cell feature maths (synthetic arrays) and the lidar percentile target."""
import numpy as np

from hgt.features import cell_mean_std, glcm_features, quantise
from hgt.targets import cell_percentile


def test_cell_mean_std_known_values_and_nan():
    stack = np.arange(2 * 20 * 20, dtype="float32").reshape(2, 20, 20)
    stack[0, :10, :10] = np.nan                     # one all-NaN cell
    stack[1, 0, 0] = np.nan                         # one partly-NaN cell
    mean, std = cell_mean_std(stack, 10)
    assert mean.shape == std.shape == (2, 2, 2)
    assert np.isnan(mean[0, 0, 0]) and np.isnan(std[0, 0, 0])
    block = stack[1, :10, :10]
    np.testing.assert_allclose(mean[1, 0, 0], np.nanmean(block), rtol=1e-6)
    np.testing.assert_allclose(std[1, 0, 0], np.nanstd(block), rtol=1e-5)
    np.testing.assert_allclose(mean[0, 1, 1], stack[0, 10:, 10:].mean(), rtol=1e-6)


def test_quantise_clips_to_levels():
    q = quantise(np.array([-5.0, 0.0, 0.5, 0.999, 1.0, 9.0]), 0.0, 1.0, 16)
    assert q.dtype == np.uint8 and q.min() == 0 and q.max() == 15
    np.testing.assert_array_equal(q, [0, 0, 8, 15, 15, 15])


def test_glcm_uniform_and_textured_patches():
    band = np.full((20, 10), 0.5, dtype="float32")          # 2 cells of 10 x 10 (rows 0 and 1)
    band[10:, :] = np.random.default_rng(0).uniform(-0.2, 1.0, size=(10, 10))
    cols, stats = glcm_features(band, "s2_NDVI_p50", np.array([0, 1]), np.array([0, 0]), 10)
    corr = cols["hc_glcm_s2_NDVI_p50_correlation"]
    ent = cols["hc_glcm_s2_NDVI_p50_entropy"]
    assert stats["n_uniform"] == 1
    assert corr[0] == 1.0 and ent[0] == 0.0                  # uniform patch: defined values, not NaN
    assert ent[1] > 2.0                                      # random texture: high entropy (bits)
    assert all(np.isfinite(v).all() for v in cols.values())


def test_cell_percentile_matches_numpy():
    rng = np.random.default_rng(1)
    x = rng.gamma(2.0, 5.0, size=(50, 400)).astype("float32")
    x[rng.random(x.shape) < 0.3] = np.nan
    for q in (50, 95, 98):
        np.testing.assert_allclose(cell_percentile(x, q), np.nanpercentile(x, q, axis=1), rtol=1e-5)
