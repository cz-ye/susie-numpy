# Changelog

All notable changes to this project are documented here.

## 0.1.0 - 2026-07-29

- Extract the optimized NumPy SuSiE kernel from the PolyFun performance work.
- Package the public `susie_suff_stat` API with NumPy and SciPy as the only
  runtime dependencies.
- Add posterior mean and variance helpers.
- Add explicit validation for model parameters, z-scores, prior weights, and
  dense LD matrices.
- Add regression, invariant, branch, and validation tests.
- Document scope, benchmark conditions, numerical differences, provenance,
  citations, and retained third-party MIT notices.
