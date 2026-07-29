import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from susie_numpy import susie_suff_stat

REFERENCE_PATH = Path(__file__).parent / "data" / "susier_v0.11.92_reference.json"


def load_reference():
    return json.loads(REFERENCE_PATH.read_text())


def fit_reference_input(dtype):
    fixture = load_reference()
    inputs = fixture["input"]
    parameters = fixture["parameters"]
    fit = susie_suff_stat(
        z=np.asarray(inputs["z"]),
        R=np.asarray(inputs["R"]),
        n=inputs["n"],
        L=inputs["L"],
        scaled_prior_variance=parameters["scaled_prior_variance"],
        estimate_prior_variance=parameters["estimate_prior_variance"],
        estimate_residual_variance=parameters["estimate_residual_variance"],
        max_iter=parameters["max_iter"],
        coverage=parameters["coverage"],
        min_abs_corr=parameters["min_abs_corr"],
        tol=parameters["tol"],
        prior_tol=parameters["prior_tol"],
        n_purity=parameters["n_purity"],
        dtype=dtype,
    )
    return fixture, fit


def test_reference_was_generated_by_exact_pinned_susier():
    fixture = load_reference()
    provenance = fixture["provenance"]
    assert provenance["susieR_version"] == "0.11.92"
    assert provenance["susieR_tag"] == "v0.11.92"
    assert (
        provenance["susieR_commit"]
        == "23606070c3025584a5e5cbd0cb6c7abb2fd4c4d4"
    )


def test_float64_matches_susier_v0_11_92_reference():
    fixture, fit = fit_reference_input(np.float64)
    expected = fixture["expected"]

    for key in ("alpha", "mu", "mu2", "V", "pip"):
        assert_allclose(fit[key], expected[key], rtol=0, atol=1e-8)
    assert_allclose(fit["sigma2"], expected["sigma2"], rtol=0, atol=1e-8)
    assert_allclose(fit["elbo"], expected["elbo"], rtol=0, atol=1e-6)
    assert fit["niter"] == expected["niter"]
    assert fit["converged"] == expected["converged"]
    assert [indices.tolist() for indices in fit["sets"]] == expected["sets"]
    assert fit["cs_index"] == expected["cs_index"]


def test_float32_preserves_susier_reference_credible_sets_and_close_pips():
    fixture, fit = fit_reference_input(np.float32)
    expected = fixture["expected"]

    assert_allclose(fit["pip"], expected["pip"], rtol=0, atol=2e-6)
    assert [indices.tolist() for indices in fit["sets"]] == expected["sets"]
    assert fit["cs_index"] == expected["cs_index"]
