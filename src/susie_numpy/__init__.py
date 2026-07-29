"""A memory-efficient NumPy implementation of summary-statistic SuSiE."""

from ._core import posterior_mean, posterior_variance, susie_suff_stat

__all__ = ["posterior_mean", "posterior_variance", "susie_suff_stat"]
__version__ = "0.1.0"
