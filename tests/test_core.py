import logging

import numpy as np
from numpy.testing import assert_allclose

from susie_numpy import posterior_mean, posterior_variance, susie_suff_stat


def correlated_example():
    rng = np.random.default_rng(42)
    p = 12
    factor = rng.normal(size=(p, p))
    R = factor @ factor.T
    diagonal = np.sqrt(np.diag(R))
    R = R / np.outer(diagonal, diagonal)
    np.fill_diagonal(R, 1.0)
    z = np.array([0.1, -0.4, 4.2, 0.3, -0.2, 0.7, 0.0, -3.4, 0.2, 0.1, -0.5, 0.2])
    return z, R


def test_float64_frozen_regression_result():
    """Guard the standalone extraction against unintended numerical changes."""
    z, R = correlated_example()
    fit = susie_suff_stat(z, R, n=5_000, L=4, dtype=np.float64)

    expected_pip = np.array(
        [
            7.44060532738640e-01,
            4.72936082813402e-04,
            9.99757697805424e-01,
            1.55234867751507e-02,
            3.75119673683533e-04,
            9.99997253823432e-01,
            3.70488800719770e-04,
            9.99999999989176e-01,
            1.57045709654358e-03,
            2.17431177804356e-01,
            3.68333264116782e-04,
            1.63092770392971e-02,
        ]
    )
    expected_V = np.array(
        [0.00531844766322, 0.01951069338201, 0.00927998019855, 0.00284926089247]
    )

    assert_allclose(fit["pip"], expected_pip, rtol=2e-7, atol=1e-10)
    # BLAS implementations differ by a few parts in 10^7 for this update.
    assert_allclose(fit["V"], expected_V, rtol=5e-7, atol=1e-10)
    assert_allclose(fit["sigma2"], 0.9880002141152385, rtol=2e-8)
    assert fit["niter"] == 13
    assert fit["converged"]
    assert [indices.tolist() for indices in fit["sets"]] == [[2], [7], [5], [0, 9]]
    assert fit["cs_index"] == [0, 1, 2, 3]


def test_float32_tracks_float64_and_preserves_credible_sets():
    z, R = correlated_example()
    fit64 = susie_suff_stat(z, R, n=5_000, L=4, dtype=np.float64)
    fit32 = susie_suff_stat(z, R, n=5_000, L=4, dtype=np.float32)

    assert_allclose(fit32["pip"], fit64["pip"], rtol=0, atol=1e-6)
    assert [indices.tolist() for indices in fit32["sets"]] == [
        indices.tolist() for indices in fit64["sets"]
    ]
    assert fit32["cs_index"] == fit64["cs_index"]


def test_output_invariants_and_posterior_helpers():
    z, R = correlated_example()
    fit = susie_suff_stat(z, R, n=5_000, L=4)

    assert_allclose(fit["alpha"].sum(axis=1), 1.0)
    assert np.all((fit["pip"] >= 0) & (fit["pip"] <= 1))
    assert np.all(fit["V"] >= 0)
    assert np.all(np.diff(fit["elbo"]) >= -1e-8)

    mean = posterior_mean(fit)
    variance = posterior_variance(fit)
    assert mean.shape == z.shape
    assert variance.shape == z.shape
    assert np.all(variance >= -1e-14)
    assert_allclose(mean, np.sum(fit["alpha"] * fit["mu"], axis=0))
    assert_allclose(
        variance,
        np.sum(
            fit["alpha"] * fit["mu2"] - (fit["alpha"] * fit["mu"]) ** 2,
            axis=0,
        ),
    )

    for indices, effect in zip(fit["sets"], fit["cs_index"], strict=True):
        assert fit["alpha"][effect, indices].sum() >= 0.95
        if len(indices) > 1:
            submatrix = np.abs(R[np.ix_(indices, indices)])
            off_diagonal = submatrix[np.triu_indices(len(indices), k=1)]
            assert off_diagonal.min() >= 0.5


def test_null_data_and_single_strong_signal():
    null_fit = susie_suff_stat(np.zeros(5), np.eye(5), n=1_000, L=3, dtype=np.float64)
    assert_allclose(null_fit["pip"], 0)
    assert_allclose(null_fit["V"], 0)
    assert null_fit["sets"] == []

    signal_fit = susie_suff_stat(
        np.array([8.0, 0, 0, 0, 0]),
        np.eye(5),
        n=1_000,
        L=3,
        dtype=np.float64,
    )
    assert signal_fit["pip"][0] > 0.999999
    assert [indices.tolist() for indices in signal_fit["sets"]] == [[0]]


def test_fixed_prior_and_residual_variances():
    z, R = correlated_example()
    fit = susie_suff_stat(
        z,
        R,
        n=5_000,
        L=3,
        scaled_prior_variance=0.05,
        residual_variance=0.8,
        estimate_prior_variance=False,
        estimate_residual_variance=False,
        dtype=np.float64,
    )
    assert_allclose(fit["V"], 0.05)
    assert fit["sigma2"] == 0.8


def test_prior_weights_are_normalized_and_zero_weight_is_excluded():
    z, R = correlated_example()
    weights = np.ones(len(z))
    weights[2] = 0.0
    weighted_fit = susie_suff_stat(z, R, n=5_000, L=4, prior_weights=10 * weights)

    assert_allclose(weighted_fit["alpha"][:, 2], 0)
    assert weighted_fit["pip"][2] == 0


def test_prior_weight_normalization_is_scale_robust():
    z, R = correlated_example()
    ordinary = np.array([1.0, 1.0] + [0.0] * (len(z) - 2))
    huge = ordinary * 1e308

    ordinary_fit = susie_suff_stat(z, R, n=5_000, L=2, prior_weights=ordinary)
    huge_fit = susie_suff_stat(z, R, n=5_000, L=2, prior_weights=huge)
    for key in ("alpha", "mu", "mu2", "V", "pip", "elbo"):
        assert_allclose(huge_fit[key], ordinary_fit[key], rtol=0, atol=0)


def test_l_is_capped_at_number_of_variables():
    fit = susie_suff_stat(np.array([6.0, 0.0]), np.eye(2), n=1_000, L=10)
    assert fit["alpha"].shape == (2, 2)


def test_nonconvergence_is_reported(caplog):
    z, R = correlated_example()
    with caplog.at_level(logging.WARNING):
        fit = susie_suff_stat(z, R, n=5_000, L=4, max_iter=1, tol=0)
    assert not fit["converged"]
    assert fit["niter"] == 1
    assert "did not converge" in caplog.text


def test_large_credible_set_purity_sampling_is_reproducible():
    p = 20
    R = np.full((p, p), 0.8)
    np.fill_diagonal(R, 1.0)
    z = np.full(p, 3.0)

    first = susie_suff_stat(
        z,
        R,
        n=2_000,
        L=1,
        n_purity=5,
        min_abs_corr=0,
        seed=1729,
    )
    second = susie_suff_stat(
        z,
        R,
        n=2_000,
        L=1,
        n_purity=5,
        min_abs_corr=0,
        seed=1729,
    )
    assert len(first["sets"][0]) > 5
    assert [indices.tolist() for indices in first["sets"]] == [
        indices.tolist() for indices in second["sets"]
    ]
