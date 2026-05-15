# Contributing

Contributions are welcome when they improve validation, reproducibility, model robustness, or documentation.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m py_compile deconvolve.py external_validation.py
```

The full deconvolution run downloads public data and may take longer than unit tests. Use synthetic tests for small pull requests.

## Pull request checklist

- `python -m pytest -q` passes.
- `python -m py_compile deconvolve.py external_validation.py` passes.
- Model architecture, HVG selection, cell-type filtering, or validation changes are documented in the README.
- Claims about accuracy include the dataset, split strategy, metric, and limitation.

## Scientific correctness

Pseudo-bulk validation is an upper bound. Do not present it as matched clinical validation. If you add or change external validation, report both positive and negative findings.

## Licence

By contributing, you agree that your contributions are licensed under the MIT licence.
