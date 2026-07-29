# Benchmarks and numerical validation

These are the measurements recorded during the July 2026 PolyFun
fine-mapping optimization work. They document the implementation's origin;
they are not a promise of the same speedup on every machine or locus.
Raw benchmark inputs and a complete software/hardware environment capture are
not bundled with this repository, so these should be treated as historical
rather than independently reproducible package benchmarks.

## Workload

The single-region benchmark used chromosome 1, 100–103 Mb, with 13,881
variants and the UK Biobank blood trait
`HIGH_LIGHT_SCATTER_RETICULOCYTE_COUNT`. Measurements were taken on an idle
compute node with a warm page cache.

The aggregate benchmark used all 32 regions on chromosomes 21 and 22. The
original and optimized workflows ran on the same node at the same six-way
parallelism so that memory-bandwidth contention was matched.

## Timings

### Single region

| Stage | Original | Optimized |
|---|---:|---:|
| Summary-statistics load | 5.6 s | 1.2 s |
| LD load | 18.6 s | 14.9 s |
| SuSiE | 132.3 s | 6.8 s |
| Wall time | 173.0 s | 27.2 s |
| Peak RSS | 14.9 GB | 4.1 GB |

### 32 regions

| Stage | Original | Optimized | Speedup |
|---|---:|---:|---:|
| Summary statistics | 35 s | 10 s | 3.4× |
| LD load | 640 s | 569 s | 1.1× |
| SuSiE | 3,940 s | 234 s | 16.8× |
| Total | 4,615 s | 813 s | 5.68× |

The summary-statistics and LD-load improvements are PolyFun integration work
and are not included in this standalone package. The SuSiE row is the relevant
kernel comparison.

## Why the memory usage falls

The optimized kernel keeps the dense LD matrix in float32 rather than float64
and does not create `XtX = (n - 1) * R`. These two choices eliminate roughly
two dense float64-sized allocations relative to the original workflow. Peak
RSS includes the surrounding PolyFun process, so it is not a package-only
allocation measurement.

## Numerical validation

The repository also includes a small synthetic fixture generated directly
with exact `susieR` v0.11.92. Normal CI checks the float64 posterior arrays,
ELBO, PIPs, convergence, and credible sets against that fixture, and checks
that the optimized float32 path preserves its credible sets. The fixture and
its R generator are under `tests/data`.

An end-to-end comparison covered 46 regions on chromosomes 21 and 22,
598,091 SNPs in total:

| Quantity | Result |
|---|---|
| Credible-set membership | Identical in all 46 regions |
| CS-filtered counts at PIP > 0.9 / 0.8 / 0.5 | 34/34, 37/37, 68/68 |
| Maximum absolute PIP difference | 1.27e-3 (one SNP) |
| SNPs with absolute PIP difference > 1e-3 / 1e-4 / 1e-5 | 1 / 23 / 213 |

In two regions, the same credible sets received different labels. A
two-variant set with correlation close to one tied a singleton set at purity
1.0 after float32 rounding, changing stable sort order but not set membership.

The main README lists the other expected sources of numerical differences:
float32 products, NumPy-versus-R purity subsampling, SciPy-versus-R Brent
optimization, and platform BLAS behavior.
