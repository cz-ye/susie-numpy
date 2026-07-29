"""High-performance SuSiE inference for the PolyFun summary-statistics path.

This module implements the specialized call

```
susieR::susie_suff_stat(
    bhat=z,
    shat=1,
    R=R,
    n=n,
    standardize=FALSE,
    ...
)
```

from susieR v0.11.92. It is not a port of the full susieR API. See NOTICE.md
for source provenance and required license notices.

The main performance gains are:

* The LD matrix and matrix products use float32 by default, halving memory
  traffic, while scalar and O(p) calculations remain float64.
* ``XtX = (n - 1) * R`` is never materialized because a unit-diagonal LD
  matrix makes ``diag(XtX)`` constant.
* The expected residual sum of squares is computed once per iteration and
  shared by the ELBO and residual-variance update.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

LOGGER = logging.getLogger(__name__)
_LD_VALIDATION_BLOCK_ROWS = 512


def _require_integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _require_finite_scalar(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    array = np.asarray(value)
    if (
        isinstance(value, (bool, np.bool_))
        or array.ndim != 0
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        below_minimum = result < minimum if minimum_inclusive else result <= minimum
        if below_minimum:
            operator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{name} must be {operator} {minimum}")
    if maximum is not None:
        above_maximum = result > maximum if maximum_inclusive else result >= maximum
        if above_maximum:
            operator = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{name} must be {operator} {maximum}")
    return result


def _as_real_numeric_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(f"{name} must contain real numeric values")
    return array


def _validate_unit_diagonal(R: np.ndarray) -> None:
    """Check the condition required by the constant-diagonal optimization."""
    p = R.shape[0]
    diagonal = np.diag(R)
    if not np.array_equal(diagonal, np.ones(p, dtype=diagonal.dtype)):
        raise ValueError(
            "R must have an exactly unit diagonal; use np.fill_diagonal(R, 1)"
        )


def _validate_ld_matrix(R: np.ndarray) -> None:
    """Validate a dense LD matrix without allocating an O(p²) temporary."""
    if np.issubdtype(R.dtype, np.floating):
        correlation_limit = 1.0 + 8.0 * np.finfo(R.dtype).eps
    else:
        correlation_limit = 1.0

    p = R.shape[0]
    for start in range(0, p, _LD_VALIDATION_BLOCK_ROWS):
        stop = min(start + _LD_VALIDATION_BLOCK_ROWS, p)
        block = R[start:stop]
        if not np.isfinite(block).all():
            raise ValueError("R must contain only finite values")
        if np.any(np.abs(block) > correlation_limit):
            raise ValueError("R entries must be correlations in the interval [-1, 1]")
        if not np.array_equal(block, R[:, start:stop].T):
            raise ValueError("R must be exactly symmetric")


def _loglik(
    V: float,
    betahat: np.ndarray,
    shat2: float,
    prior_weights: np.ndarray,
) -> float:
    """susieR:::loglik: log marginal likelihood of a single-effect model."""
    # The 2*pi and betahat**2 terms partially cancel when the two normal
    # log-densities are subtracted.
    lbf = 0.5 * np.log(shat2 / (V + shat2)) + 0.5 * betahat**2 * (
        1.0 / shat2 - 1.0 / (V + shat2)
    )
    maxlbf = lbf.max()
    weighted = np.exp(lbf - maxlbf) * prior_weights
    return float(np.log(weighted.sum()) + maxlbf)


def _optimize_prior_variance(
    betahat: np.ndarray,
    shat2: float,
    prior_weights: np.ndarray,
    V_init: float,
    check_null_threshold: float = 0.0,
) -> float:
    """susieR:::optimize_prior_variance with estimate_prior_method='optim'."""

    def negative_loglik(log_variance: float) -> float:
        return -_loglik(np.exp(log_variance), betahat, shat2, prior_weights)

    # R's optim(method="Brent") searches this interval with a tolerance of
    # sqrt(.Machine$double.eps). scipy's bounded method is also Brent's method.
    result = minimize_scalar(
        negative_loglik,
        bounds=(-30.0, 15.0),
        method="bounded",
        options={"xatol": np.sqrt(np.finfo(np.float64).eps)},
    )
    log_variance = float(result.x)

    # susieR keeps the incoming value if the optimizer found a worse value.
    if V_init > 0 and negative_loglik(log_variance) > negative_loglik(np.log(V_init)):
        log_variance = float(np.log(V_init))
    variance = float(np.exp(log_variance))

    if (
        _loglik(0.0, betahat, shat2, prior_weights) + check_null_threshold
        >= _loglik(variance, betahat, shat2, prior_weights)
    ):
        variance = 0.0
    return variance


def _single_effect_regression_ss(
    XtR: np.ndarray,
    d: float,
    V: float,
    sigma2: float,
    prior_weights: np.ndarray,
    estimate_prior_variance: bool = True,
    check_null_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """susieR:::single_effect_regression_ss for constant dXtX."""
    betahat = XtR / d
    shat2 = sigma2 / d

    if estimate_prior_variance:
        V = _optimize_prior_variance(
            betahat,
            shat2,
            prior_weights,
            V,
            check_null_threshold=check_null_threshold,
        )

    lbf = 0.5 * np.log(shat2 / (V + shat2)) + 0.5 * betahat**2 * (
        1.0 / shat2 - 1.0 / (V + shat2)
    )
    maxlbf = lbf.max()
    weighted = np.exp(lbf - maxlbf) * prior_weights
    weighted_sum = weighted.sum()
    alpha = weighted / weighted_sum

    # V == 0 is the null effect and therefore has posterior variance zero.
    posterior_variance = 0.0 if V <= 0 else 1.0 / (1.0 / V + d / sigma2)
    # Keep the operation order used by the extracted implementation. Although
    # algebraically equivalent to (posterior_variance / sigma2) * XtR, changing
    # the scalar evaluation order can move the last floating-point bits.
    posterior_mean = (1.0 / sigma2) * posterior_variance * XtR
    posterior_mean2 = posterior_variance + posterior_mean**2
    model_lbf = float(maxlbf + np.log(weighted_sum))
    return alpha, posterior_mean, posterior_mean2, V, model_lbf


def _n_in_cs(alpha_l: np.ndarray, coverage: float) -> int:
    """Return the smallest number of variables attaining the requested coverage."""
    return int(np.sum(np.cumsum(np.sort(alpha_l)[::-1]) < coverage) + 1)


def _get_purity(
    positions: np.ndarray,
    R: np.ndarray,
    n_purity: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return minimum, mean and median absolute within-set correlation."""
    if len(positions) == 1:
        return 1.0, 1.0, 1.0
    if len(positions) > n_purity:
        positions = rng.choice(positions, size=n_purity, replace=False)
    submatrix = np.asarray(R[np.ix_(positions, positions)], dtype=np.float64)
    upper_triangle = np.triu_indices(submatrix.shape[0], k=1)
    correlations = np.abs(submatrix[upper_triangle])
    return (
        float(correlations.min()),
        float(correlations.mean()),
        float(np.median(correlations)),
    )


def susie_suff_stat(
    z: Any,
    R: Any,
    n: int,
    L: int = 10,
    prior_weights: Any | None = None,
    scaled_prior_variance: float = 1e-4,
    residual_variance: float | None = None,
    estimate_prior_variance: bool = True,
    estimate_residual_variance: bool = True,
    max_iter: int = 100,
    tol: float = 1e-3,
    coverage: float = 0.95,
    min_abs_corr: float = 0.5,
    prior_tol: float = 1e-9,
    n_purity: int = 100,
    dtype: Any = np.float32,
    seed: Any = 0,
    check_input: bool = True,
) -> dict[str, Any]:
    """Fit the specialized summary-statistic SuSiE model.

    This is equivalent to the following v0.11.92 susieR call:

    ``susie_suff_stat(bhat=z, shat=1, R=R, n=n, L=L,
    standardize=FALSE, ...)``.

    Parameters
    ----------
    z
        One-dimensional vector of z-scores.
    R
        Dense, exactly symmetric LD correlation matrix with an exactly unit
        diagonal. The z-scores and LD should normally come from the same
        samples.
    n
        GWAS sample size. Must be an integer of at least 3.
    L
        Upper bound on the number of single effects. Values larger than the
        number of variables are capped at that number.
    prior_weights
        Optional nonnegative per-variable prior weights. They are normalized
        to sum to one.
    scaled_prior_variance
        Initial scaled prior variance. The default, 1e-4, is inherited from
        PolyFun and differs from general-purpose susieR defaults.
    residual_variance
        Initial or fixed residual variance. It must be positive when supplied.
    estimate_prior_variance
        Estimate each single effect's prior variance with Brent optimization.
    estimate_residual_variance
        Update the residual variance after every IBSS iteration.
    max_iter
        Maximum number of IBSS iterations.
    tol
        Stop when the ELBO improvement is below this nonnegative tolerance.
    coverage
        Nominal credible-set coverage in the interval (0, 1].
    min_abs_corr
        Minimum absolute within-set correlation for a credible set to pass the
        purity filter.
    prior_tol
        Single effects with prior variance at or below this value are excluded
        from PIP and credible-set calculations.
    n_purity
        Maximum number of variables sampled for a credible-set purity check.
    dtype
        Working precision of the LD matrix and its products. float32 is the
        optimized default; float64 provides a higher-precision comparison.
    seed
        Seed passed to ``numpy.random.default_rng`` for purity subsampling.
    check_input
        Check every LD entry for finiteness, correlation bounds, and exact
        symmetry. The mandatory unit-diagonal check is always performed. Set
        this to False only for a trusted matrix to avoid the O(p²) scan.

    Returns
    -------
    dict
        Keys are ``alpha``, ``mu``, ``mu2``, ``V``, ``sigma2``, ``pip``,
        ``niter``, ``converged``, ``elbo``, ``X_column_scale_factors``,
        ``sets`` and ``cs_index``. Credible-set indices are zero-based.

    Notes
    -----
    The float32 path is intentionally not bit-identical to susieR. Large
    credible sets can also differ at the purity threshold because this
    implementation uses NumPy's seeded RNG for subsampling.
    """
    z_array = _as_real_numeric_array(z, "z")
    if z_array.ndim != 1:
        raise ValueError("z must be one-dimensional")
    if z_array.size == 0:
        raise ValueError("z must contain at least one value")
    z_array = np.asarray(z_array, dtype=np.float64)
    if not np.isfinite(z_array).all():
        raise ValueError("z must contain only finite values")

    R_array = _as_real_numeric_array(R, "R")
    if R_array.ndim != 2 or R_array.shape[0] != R_array.shape[1]:
        raise ValueError("R must be a square two-dimensional matrix")
    p = z_array.size
    if R_array.shape != (p, p):
        raise ValueError("R and z disagree on the number of variables")

    n = _require_integer(n, "n", 3)
    L = _require_integer(L, "L", 1)
    max_iter = _require_integer(max_iter, "max_iter", 1)
    n_purity = _require_integer(n_purity, "n_purity", 1)
    L = min(L, p)

    if not isinstance(check_input, (bool, np.bool_)):
        raise TypeError("check_input must be boolean")
    _validate_unit_diagonal(R_array)
    if check_input:
        _validate_ld_matrix(R_array)

    try:
        rng = np.random.default_rng(seed)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "seed must be nonnegative or another value accepted by "
            "numpy.random.default_rng"
        ) from error

    try:
        working_dtype = np.dtype(dtype)
    except TypeError as error:
        raise TypeError("dtype must be np.float32 or np.float64") from error
    if working_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be np.float32 or np.float64")

    scaled_prior_variance = _require_finite_scalar(
        scaled_prior_variance, "scaled_prior_variance", minimum=0.0
    )
    if residual_variance is not None:
        residual_variance = _require_finite_scalar(
            residual_variance,
            "residual_variance",
            minimum=0.0,
            minimum_inclusive=False,
        )
    tol = _require_finite_scalar(tol, "tol", minimum=0.0)
    coverage = _require_finite_scalar(
        coverage,
        "coverage",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    min_abs_corr = _require_finite_scalar(
        min_abs_corr, "min_abs_corr", minimum=0.0, maximum=1.0
    )
    prior_tol = _require_finite_scalar(prior_tol, "prior_tol", minimum=0.0)

    for value, name in (
        (estimate_prior_variance, "estimate_prior_variance"),
        (estimate_residual_variance, "estimate_residual_variance"),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be boolean")

    if prior_weights is None:
        normalized_prior_weights = np.full(p, 1.0 / p)
    else:
        weights = _as_real_numeric_array(prior_weights, "prior_weights")
        if weights.ndim != 1 or weights.shape[0] != p:
            raise ValueError("prior_weights must have the same length as z")
        weights = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(weights).all():
            raise ValueError("prior_weights must contain only finite values")
        if np.any(weights < 0):
            raise ValueError("prior_weights must be nonnegative")
        maximum_weight = weights.max()
        if maximum_weight <= 0:
            raise ValueError("at least one prior weight must be positive")
        if maximum_weight > np.finfo(np.float64).max / p:
            # Dividing first avoids overflow when several individually finite
            # weights are near float64's maximum.
            scaled_weights = weights / maximum_weight
            normalized_prior_weights = scaled_weights / scaled_weights.sum()
        else:
            # Preserve the operation order of the extracted implementation for
            # ordinary inputs.
            normalized_prior_weights = weights / weights.sum()

    # This is the only large allocation when R is not already C-contiguous in
    # the requested working precision.
    working_R = np.asarray(R_array, dtype=working_dtype, order="C")

    # susie_suff_stat's bhat/shat conversion for bhat=z, shat=1 and var_y unset.
    R2 = z_array**2 / (z_array**2 + n - 2.0)
    sigma2_vector = (n - 1.0) * (1.0 - R2) / (n - 2.0)
    Xty = np.sqrt(sigma2_vector) * np.sqrt(n - 1.0) * z_array
    yty = n - 1.0

    # XtX = (n-1)*R and diag(R)=1, so d is constant and XtX is unnecessary.
    d = n - 1.0
    scale = np.asarray(n - 1.0, dtype=np.float64)

    def XtX_dot(vector: np.ndarray) -> np.ndarray:
        product = working_R @ np.asarray(vector, dtype=working_dtype)
        return scale * product.astype(np.float64)

    def XtX_rmatmul(matrix: np.ndarray) -> np.ndarray:
        product = np.asarray(matrix, dtype=working_dtype) @ working_R
        return scale * product.astype(np.float64)

    sigma2 = 1.0 if residual_variance is None else residual_variance
    V = np.full(L, scaled_prior_variance, dtype=np.float64)
    alpha = np.full((L, p), 1.0 / p)
    mu = np.zeros((L, p))
    mu2 = np.zeros((L, p))
    KL = np.full(L, np.nan)
    XtXr = np.zeros(p)

    previous_elbo = -np.inf
    elbo_history: list[float] = []
    converged = False
    niter = 0

    for iteration in range(1, max_iter + 1):
        niter = iteration

        # susieR:::update_each_effect_ss
        for effect in range(L):
            XtXr -= XtX_dot(alpha[effect] * mu[effect])
            XtR = Xty - XtXr
            (
                effect_alpha,
                effect_mu,
                effect_mu2,
                effect_V,
                model_lbf,
            ) = _single_effect_regression_ss(
                XtR,
                d,
                V[effect],
                sigma2,
                normalized_prior_weights,
                estimate_prior_variance=estimate_prior_variance,
            )
            alpha[effect] = effect_alpha
            mu[effect] = effect_mu
            mu2[effect] = effect_mu2
            V[effect] = effect_V

            expected_beta = effect_alpha * effect_mu
            expected_beta2 = effect_alpha * effect_mu2
            expected_loglik = -0.5 / sigma2 * (
                -2.0 * np.sum(expected_beta * XtR) + d * np.sum(expected_beta2)
            )
            KL[effect] = -model_lbf + expected_loglik
            XtXr += XtX_dot(alpha[effect] * mu[effect])

        # susieR:::get_ER2_ss, shared by the ELBO and residual-variance update.
        expected_effects = alpha * mu
        effects_XtX = XtX_rmatmul(expected_effects)
        within_effect_quadratic = np.sum(effects_XtX * expected_effects)
        posterior_mean_beta = expected_effects.sum(axis=0)
        posterior_second_moment = alpha * mu2
        expected_residual_sum_squares = (
            yty
            - 2.0 * np.sum(posterior_mean_beta * Xty)
            + np.sum(posterior_mean_beta * XtX_dot(posterior_mean_beta))
            - within_effect_quadratic
            + d * np.sum(posterior_second_moment)
        )

        elbo = (
            -n / 2.0 * np.log(2.0 * np.pi * sigma2)
            - expected_residual_sum_squares / (2.0 * sigma2)
            - KL.sum()
        )
        if not np.isfinite(elbo):
            raise ValueError("the SuSiE objective became infinite; check the input")
        elbo_history.append(float(elbo))

        if elbo - previous_elbo < tol:
            converged = True
            break
        previous_elbo = float(elbo)

        if estimate_residual_variance:
            estimated_sigma2 = expected_residual_sum_squares / n
            if estimated_sigma2 < 0:
                raise ValueError("estimating the residual variance failed (negative)")
            sigma2 = float(estimated_sigma2)

    if not converged:
        LOGGER.warning("SuSiE did not converge in %d iterations", max_iter)

    # susieR::susie_get_pip(prune_by_cs=FALSE)
    included_effects = np.where(V > prior_tol)[0]
    if included_effects.size > 0:
        pip = 1.0 - np.prod(1.0 - alpha[included_effects], axis=0)
    else:
        pip = np.zeros(p)

    # susieR::susie_get_cs
    credible_sets: list[np.ndarray] = []
    credible_set_effects: list[int] = []
    minimum_purities: list[float] = []
    seen: set[bytes] = set()
    for effect in range(L):
        if V[effect] <= prior_tol:
            continue
        size = _n_in_cs(alpha[effect], coverage)
        indices = np.sort(np.argsort(-alpha[effect], kind="stable")[:size])
        key = indices.tobytes()
        if key in seen:
            continue
        seen.add(key)
        minimum_purity, _, _ = _get_purity(indices, working_R, n_purity, rng)
        if minimum_purity >= min_abs_corr:
            credible_sets.append(indices)
            credible_set_effects.append(effect)
            minimum_purities.append(minimum_purity)

    # R's order() is stable. Ties are common because singleton sets have purity 1.
    order = (
        np.argsort(-np.asarray(minimum_purities), kind="stable")
        if credible_sets
        else np.array([], dtype=int)
    )
    credible_sets = [credible_sets[index] for index in order]
    credible_set_effects = [credible_set_effects[index] for index in order]

    return {
        "alpha": alpha,
        "mu": mu,
        "mu2": mu2,
        "V": V,
        "sigma2": sigma2,
        "pip": pip,
        "niter": niter,
        "converged": converged,
        "elbo": np.asarray(elbo_history),
        "X_column_scale_factors": np.ones(p),
        "sets": credible_sets,
        "cs_index": credible_set_effects,
    }


def posterior_mean(fit: Mapping[str, Any]) -> np.ndarray:
    """Return the posterior mean effect for each variable."""
    alpha = np.asarray(fit["alpha"])
    mu = np.asarray(fit["mu"])
    scale = np.asarray(fit.get("X_column_scale_factors", 1.0))
    return np.sum(alpha * mu, axis=0) / scale


def posterior_variance(fit: Mapping[str, Any]) -> np.ndarray:
    """Return the posterior variance of each variable's effect."""
    alpha = np.asarray(fit["alpha"])
    mu = np.asarray(fit["mu"])
    mu2 = np.asarray(fit["mu2"])
    scale = np.asarray(fit.get("X_column_scale_factors", 1.0))
    return np.sum(alpha * mu2 - (alpha * mu) ** 2, axis=0) / scale**2
