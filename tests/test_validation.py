import numpy as np
import pytest

from susie_numpy import susie_suff_stat


def valid_inputs():
    return np.array([1.0, 2.0, 3.0]), np.eye(3)


@pytest.mark.parametrize("n", [0, 2, 3.0, True])
def test_rejects_invalid_sample_size(n):
    z, R = valid_inputs()
    with pytest.raises((TypeError, ValueError), match="n must"):
        susie_suff_stat(z, R, n=n)


@pytest.mark.parametrize("name,value", [("L", 0), ("max_iter", 0), ("n_purity", 0)])
def test_rejects_invalid_integer_controls(name, value):
    z, R = valid_inputs()
    kwargs = {name: value}
    with pytest.raises(ValueError, match=name):
        susie_suff_stat(z, R, n=100, **kwargs)


def test_rejects_non_vector_or_nonfinite_z():
    _, R = valid_inputs()
    with pytest.raises(ValueError, match="one-dimensional"):
        susie_suff_stat(np.ones((3, 1)), R, n=100)
    with pytest.raises(ValueError, match="finite"):
        susie_suff_stat(np.array([1.0, np.nan, 2.0]), R, n=100)
    with pytest.raises(TypeError, match="real numeric"):
        susie_suff_stat(np.array(["1", "2", "3"]), R, n=100)


def test_rejects_bad_ld_shape_or_size():
    z, _ = valid_inputs()
    with pytest.raises(ValueError, match="square"):
        susie_suff_stat(z, np.ones((3, 2)), n=100)
    with pytest.raises(ValueError, match="disagree"):
        susie_suff_stat(z, np.eye(4), n=100)


def test_rejects_nonfinite_asymmetric_or_nonunit_ld():
    z, R = valid_inputs()

    nonfinite = R.copy()
    nonfinite[0, 1] = nonfinite[1, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        susie_suff_stat(z, nonfinite, n=100)

    asymmetric = R.copy()
    asymmetric[0, 1] = 0.2
    with pytest.raises(ValueError, match="symmetric"):
        susie_suff_stat(z, asymmetric, n=100)

    nonunit = R.copy()
    nonunit[0, 0] = 0.999
    with pytest.raises(ValueError, match="unit diagonal"):
        susie_suff_stat(z, nonunit, n=100)

    out_of_bounds = R.copy()
    out_of_bounds[0, 1] = out_of_bounds[1, 0] = 1.1
    with pytest.raises(ValueError, match="correlations"):
        susie_suff_stat(z, out_of_bounds, n=100)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"scaled_prior_variance": -1}, "scaled_prior_variance"),
        ({"residual_variance": 0}, "residual_variance"),
        ({"tol": -1}, "tol"),
        ({"coverage": 0}, "coverage"),
        ({"coverage": 1.1}, "coverage"),
        ({"min_abs_corr": -0.1}, "min_abs_corr"),
        ({"min_abs_corr": 1.1}, "min_abs_corr"),
        ({"prior_tol": -1}, "prior_tol"),
    ],
)
def test_rejects_invalid_scalar_controls(kwargs, match):
    z, R = valid_inputs()
    with pytest.raises(ValueError, match=match):
        susie_suff_stat(z, R, n=100, **kwargs)


@pytest.mark.parametrize(
    "kwargs,name",
    [
        ({"coverage": True}, "coverage"),
        ({"tol": False}, "tol"),
        ({"coverage": "0.9"}, "coverage"),
        ({"scaled_prior_variance": "0.1"}, "scaled_prior_variance"),
    ],
)
def test_rejects_boolean_or_string_scalar_controls(kwargs, name):
    z, R = valid_inputs()
    with pytest.raises(TypeError, match=name):
        susie_suff_stat(z, R, n=100, **kwargs)


@pytest.mark.parametrize(
    "weights,match",
    [
        (np.ones(2), "same length"),
        (np.array([0.0, 0.0, 0.0]), "positive"),
        (np.array([1.0, -1.0, 1.0]), "nonnegative"),
        (np.array([1.0, np.nan, 1.0]), "finite"),
    ],
)
def test_rejects_invalid_prior_weights(weights, match):
    z, R = valid_inputs()
    with pytest.raises(ValueError, match=match):
        susie_suff_stat(z, R, n=100, prior_weights=weights)


@pytest.mark.parametrize("dtype", [np.float16, np.int64, "not-a-dtype"])
def test_rejects_invalid_working_dtype(dtype):
    z, R = valid_inputs()
    with pytest.raises((TypeError, ValueError), match="dtype"):
        susie_suff_stat(z, R, n=100, dtype=dtype)


def test_expensive_ld_scan_can_be_skipped_for_prevalidated_input(monkeypatch):
    z, R = valid_inputs()

    def fail_if_called(_):
        raise AssertionError("validation scan was called")

    monkeypatch.setattr("susie_numpy._core._validate_ld_matrix", fail_if_called)
    fit = susie_suff_stat(z, R, n=100, check_input=False)
    assert fit["pip"].shape == z.shape


def test_unit_diagonal_is_still_required_when_expensive_scan_is_skipped():
    z, R = valid_inputs()
    R[0, 0] = 2.0
    with pytest.raises(ValueError, match="unit diagonal"):
        susie_suff_stat(z, R, n=100, check_input=False)


@pytest.mark.parametrize("seed", [-1, "invalid"])
def test_rejects_invalid_seed_before_inference(seed, monkeypatch):
    z, R = valid_inputs()

    def fail_if_inference_starts(*args, **kwargs):
        raise AssertionError("inference started before seed validation")

    monkeypatch.setattr("susie_numpy._core._single_effect_regression_ss", fail_if_inference_starts)
    with pytest.raises(ValueError, match="seed"):
        susie_suff_stat(z, R, n=100, seed=seed)
