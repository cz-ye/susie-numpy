# Provenance and third-party notices

`susie-numpy` is an independent, unofficial project. It is not endorsed by or
affiliated with the Stephens Lab, the susieR authors, the PolyFun authors, or
Anthropic.

The initial numerical implementation was developed by Chengzhong Ye
([`cz-ye`](https://github.com/cz-ye)) in the PolyFun performance-optimization
work recorded in commit
`b5fa3255020daf3689b2540e738a9d697864d626` (July 14, 2026), with implementation
assistance from Claude Code. It was extracted into this standalone package,
given a public API, validation, tests, packaging, and documentation in July
2026.

The implementation translates and specializes functions from
[susieR v0.11.92](https://github.com/stephenslab/susieR/tree/v0.11.92), including
the iterative Bayesian stepwise selection loop, single-effect regression,
prior-variance optimization, ELBO, PIP, and credible-set purity calculations.
The exact upstream tag resolves to commit
[`23606070c3025584a5e5cbd0cb6c7abb2fd4c4d4`](https://github.com/stephenslab/susieR/commit/23606070c3025584a5e5cbd0cb6c7abb2fd4c4d4).
That version is MIT-licensed; its copyright and permission notice are retained
in [`LICENSES/susieR-v0.11.92-MIT.txt`](LICENSES/susieR-v0.11.92-MIT.txt).

The implementation originated inside
[PolyFun](https://github.com/omerwe/polyfun), which is MIT-licensed. PolyFun's
copyright and permission notice are retained in
[`LICENSES/PolyFun-MIT.txt`](LICENSES/PolyFun-MIT.txt).

Scientific use should cite the SuSiE and SuSiE summary-statistics papers listed
in the README. License notices are legal attribution; scientific citations are
separate and are important for crediting the method's authors.
