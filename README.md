# End-to-End Wine Classification MLOps Pipeline

[![CI Pipeline](https://github.com/danial-zahid/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/danial-zahid/wine-mlops-pipeline/actions/workflows/ci.yml)

An evolving MLOps project for multiclass wine cultivar classification. The current implementation loads and validates the Scikit-learn Wine dataset and creates a reproducible, stratified train/test split. Training, MLflow model registration, and model quality gates are planned components and are not implemented yet.

## Current Status

- **Implemented:** Wine dataset loading, basic schema checks, and an 80/20 stratified split with a fixed random seed.
- **Configured:** A GitHub Actions workflow for linting, training, and tests on pushes and pull requests to `main`.
- **Not implemented yet:** Model training, MLflow tracking or registry, champion-model evaluation, and automated quality-gate tests. The corresponding scripts and test files are currently empty, and `requirements.txt` does not yet declare dependencies.

## Repository Layout

```text
wine-mlops-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions workflow
├── src/
│   ├── __init__.py
│   ├── data.py                # Dataset loading, validation, and splitting
│   ├── train.py               # Planned training entry point
│   └── evaluate.py            # Planned model evaluation entry point
├── tests/
│   ├── __init__.py
│   ├── test_data.py            # Planned data pipeline tests
│   └── test_model_gate.py      # Planned model quality-gate tests
├── requirements.txt            # Dependency list (currently empty)
└── Makefile                    # Common project commands
```

## Requirements

- Python 3.10 or later
- pip

## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/danial-zahid/wine-mlops-pipeline.git
cd wine-mlops-pipeline
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the packages used by the current code and CI workflow. `requirements.txt` is currently empty, so install them directly:

```bash
python -m pip install --upgrade pip
python -m pip install flake8 pytest scikit-learn mlflow pandas numpy
```

## Use the Data Loader

The loader returns `X_train`, `X_test`, `y_train`, and `y_test`. It uses Scikit-learn's built-in Wine dataset, checks for missing feature values and the expected 13 features, then creates a stratified split with `test_size=0.20` and `random_state=42`.

```python
from src.data import load_and_validate_data

X_train, X_test, y_train, y_test = load_and_validate_data()
print(X_train.shape, X_test.shape)
```

Run the linter:

```bash
flake8 src/ tests/ --max-line-length=100
```

The training and evaluation entry points are not implemented yet. The current `pytest tests/ -v` command will not run meaningful tests until test cases are added.

## CI

The workflow in `.github/workflows/ci.yml` runs on pushes and pull requests targeting `main`. It installs Python 3.10 and the project tools, runs Flake8, invokes `src/train.py`, and runs pytest. Since training and tests are currently empty, the workflow needs those components implemented before it can validate an end-to-end pipeline.

## Planned MLOps Features

- Compare Random Forest and Gradient Boosting models using stratified cross-validation.
- Track experiments and register the best model with MLflow, including a `champion` alias.
- Evaluate the registered model on the held-out test set.
- Add quality gates for macro F1, inference latency, and valid output labels.

## License

No license is currently specified for this repository.

## Author

- **Danial Zahid**
- GitHub: [@danial-zahid](https://github.com/danial-zahid)
