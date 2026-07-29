# Contributing

Bug reports and focused pull requests are welcome.

Set up a development environment with:

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python -m build
```

Numerical changes should include a regression test and should state whether
they alter float32 results, credible-set membership, convergence, or memory
use. Changes that broaden the model beyond the specialized v0.11.92
summary-statistics path should be proposed in an issue before implementation.

For a release, keep the version synchronized in `pyproject.toml`,
`src/susie_numpy/__init__.py`, `CITATION.cff`, and `CHANGELOG.md`.

Please do not commit individual-level genetic data or data whose redistribution
terms are unclear. Small synthetic test fixtures are preferred.
