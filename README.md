# susie-numpy

[![tests](https://github.com/cz-ye/susie-numpy/actions/workflows/tests.yml/badge.svg)](https://github.com/cz-ye/susie-numpy/actions/workflows/tests.yml)

`susie-numpy` is a memory-efficient NumPy implementation of the specialized
SuSiE summary-statistics path used by
[PolyFun](https://github.com/omerwe/polyfun). It accepts GWAS z-scores and a
dense LD correlation matrix, uses float32 for the bandwidth-bound matrix
products, and keeps the remaining calculations in float64.

> [!IMPORTANT]
> This is an independent, specialized port of
> `susieR::susie_suff_stat()` v0.11.92 for `bhat=z`, `shat=1`, unset `var_y`,
> and `standardize=FALSE`. It is not a full Python port of susieR, it is not a
> drop-in replacement for current susieR, and it is not affiliated with the
> Stephens Lab.

## Performance

In the original PolyFun benchmark, the optimized SuSiE stage was 16.8× faster
over 32 UK Biobank fine-mapping regions. For one 13,881-variant region, it
reduced the SuSiE stage from 132.3 seconds to 6.8 seconds. The complete
fine-mapping workflow, which includes separate PolyFun I/O optimizations, was
5.68× faster and reduced peak per-region memory from 14.9 GB to 4.1 GB.

| Benchmark | susieR path | NumPy path | Change |
|---|---:|---:|---:|
| SuSiE, one 13,881-variant region | 132.3 s | 6.8 s | 19.5× |
| SuSiE, 32 regions | 3,940 s | 234 s | 16.8× |
| Full fine-mapping, 32 regions | 4,615 s | 813 s | 5.68× |
| Peak RSS, one region | 14.9 GB | 4.1 GB | 3.6× lower |

These numbers are workload- and hardware-specific. The full-workflow values
include optimizations outside this package. See [BENCHMARKS.md](BENCHMARKS.md)
for the conditions and accuracy results.

The numerical kernel is faster because it:

- keeps the dense LD matrix and its products in float32;
- never materializes `XtX = (n - 1) * R`;
- computes the expected residual sum of squares once per iteration instead of
  twice.

## Installation

The package requires Python 3.10 or newer, NumPy, and SciPy.

```bash
git clone https://github.com/cz-ye/susie-numpy.git
cd susie-numpy
python -m pip install .
```

It can also be installed directly from GitHub:

```bash
python -m pip install "git+https://github.com/cz-ye/susie-numpy.git"
```

No R installation or `rpy2` dependency is needed.

An existing PolyFun checkout contains its own top-level `susie_numpy.py`.
Python will prefer that local file over an installed package with the same
import name, so installing this repository alone does not replace the module
inside an existing PolyFun checkout.

## Quick start

```python
import numpy as np

from susie_numpy import posterior_mean, posterior_variance, susie_suff_stat

# z has shape (p,); R has shape (p, p).
z = np.load("z_scores.npy")
R = np.load("ld.npy")

# The optimized path assumes an exactly unit-diagonal, exactly symmetric R.
np.fill_diagonal(R, 1)

fit = susie_suff_stat(
    z=z,
    R=R,
    n=500_000,
    L=10,
    prior_weights=None,  # Optional functionally informed prior probabilities.
)

pip = fit["pip"]
credible_sets = fit["sets"]  # List of zero-based integer index arrays.
beta_mean = posterior_mean(fit)
beta_sd = np.sqrt(posterior_variance(fit))
```

The input scan verifies that `R` is finite, bounded like a correlation matrix,
exactly symmetric, and has an exactly unit diagonal without allocating a
second matrix. If a trusted matrix was already validated, `check_input=False`
skips the O(p²) finite/bounds/symmetry scan. The cheap, mandatory unit-diagonal
check still runs because the optimized algorithm depends on it.

## Inputs and model scope

The optimized assumptions are intentionally strict:

- `z` is a one-dimensional vector of finite z-scores.
- `R` is a dense, real, exactly symmetric correlation matrix with an exactly
  unit diagonal. Positive semidefiniteness is assumed but is not checked,
  because an eigendecomposition would be prohibitively expensive.
- The LD and z-scores should normally be computed from the same samples. The
  package does not implement modern susieR diagnostics or corrections for LD
  reference mismatch.
- `prior_weights`, when supplied, must be nonnegative and are normalized to
  sum to one.
- The default `scaled_prior_variance=1e-4` is inherited from the PolyFun
  workflow. It is not the general-purpose default used by susieR.

The package does not currently accept individual-level `X, y`, arbitrary
`bhat, shat`, precomputed `XtX, Xty, yty`, sparse LD, refinement
initializations, a null weight, or the broader diagnostics and plotting API in
current susieR.

## Results

`susie_suff_stat` returns a dictionary compatible with the subset consumed by
PolyFun:

| Key | Meaning |
|---|---|
| `alpha`, `mu`, `mu2` | Per-effect inclusion probabilities and posterior moments |
| `V` | Estimated prior variance for each single effect |
| `sigma2` | Final residual variance |
| `pip` | Per-variable posterior inclusion probability |
| `niter`, `converged`, `elbo` | Optimization diagnostics |
| `sets` | Purity-filtered credible sets, as zero-based index arrays |
| `cs_index` | Single-effect index corresponding to each credible set |
| `X_column_scale_factors` | All ones for this nonstandardized path |

`posterior_mean(fit)` and `posterior_variance(fit)` reproduce the posterior
summary formulas used by PolyFun.

## Numerical differences from susieR

The default is deliberately approximate:

- Matrix products use float32, so PIPs are not bit-identical to the float64 R
  implementation. Pass `dtype=np.float64` for a higher-precision comparison.
- Credible sets larger than `n_purity` (100 by default) are purity-checked with
  NumPy's seeded random-number generator rather than R's generator. A set very
  near the purity threshold can therefore be retained in one implementation
  and filtered in the other.
- Prior variance uses SciPy's bounded Brent optimizer rather than R's
  `optim(method="Brent")`.
- BLAS implementation and platform can affect the last numerical bits.

Across the documented 46-region PolyFun validation, credible-set membership
matched the original implementation in every region. The largest absolute PIP
difference was `1.27e-3` (one SNP among 598,091); 23 SNPs differed by more than
`1e-4`. Two regions contained the same credible sets with different numeric
labels because float32 rounded their purity scores into a tie.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python -m build
```

The tests include a frozen result generated directly by exact
`susieR` v0.11.92, a regression result from the original extracted Python
implementation, float32-versus-float64 comparisons, model invariants,
parameter branches, and input validation. The R generator script and reference
JSON are under `tests/data`. These checks do not claim exhaustive parity with
current susieR.

## Citation and provenance

If you use this software, please cite the papers that introduced SuSiE and its
summary-statistics formulation:

- Wang G, Sarkar A, Carbonetto P, Stephens M. “A simple new approach to
  variable selection in regression, with application to genetic fine
  mapping.” *Journal of the Royal Statistical Society: Series B* (2020).
  [doi:10.1111/rssb.12388](https://doi.org/10.1111/rssb.12388)
- Zou Y, Carbonetto P, Wang G, Stephens M. “Fine-mapping from summary data
  with the ‘Sum of Single Effects’ model.” *PLOS Genetics* (2022).
  [doi:10.1371/journal.pgen.1010299](https://doi.org/10.1371/journal.pgen.1010299)

The NumPy implementation originated in PolyFun performance work by Chengzhong
Ye ([`cz-ye`](https://github.com/cz-ye)), with assistance from Claude Code. See
[NOTICE.md](NOTICE.md) for exact source versions and provenance.

## License

MIT. Because this package specializes code from susieR v0.11.92 and originated
inside PolyFun, their full MIT notices are retained under [LICENSES](LICENSES).
