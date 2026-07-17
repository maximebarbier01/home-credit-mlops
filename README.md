# Home Credit MLOps

Base project for the OpenClassrooms "Initiez-vous au MLOps" credit scoring exercise.

## Main entrypoints

- `scripts/build_home_credit_dataset.py`: data preparation, feature engineering, and final dataset EDA from the raw Home Credit tables
- `scripts/run_home_credit_experiment.py`: model training, benchmark, threshold optimization, SHAP, Excel exports, and MLflow tracking
- `scripts/mlflow_ui.py`: start the MLflow UI during the MLOps phase
- detailed French guide: `docs/mode_emploi_pipeline_ml.md`

## Project layout

```text
home-credit-mlops/
|-- configs/
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
|-- docs/
|-- scripts/
|-- src/home_credit_mlops/
|   |-- data/
|   |-- eda/
|   |-- features/
|   |-- modeling/
|   `-- reporting/
`-- tests/
```

## Build schema

The project now follows one main ML path from end to end:

1. `data/home_credit.py`
   prepares, cleans, joins, enriches, and documents the final modeling dataset.
2. `eda/diagnostics.py`
   produces the final dataset EDA and data-quality reports during the data-preparation step.
3. `features/preprocessing.py`
   defines the preprocessing used by the models.
4. `modeling/benchmark.py`
   trains and compares candidate models with cross-validation, evaluates them,
   optimizes the business decision threshold, exports diagnostics, and optionally logs everything in MLflow.
4. `modeling/interpretability.py`
   exports global feature importance and local and global SHAP explanations.
5. `reporting/excel.py`
   bundles the experiment outputs into Excel workbooks.

So the pattern stays simple:

- `scripts/` contains a few executable entrypoints
- `src/home_credit_mlops/` contains the reusable, testable, importable logic

## Important note about the environment

Poetry is installed in WSL on this machine, not in Windows PowerShell.
Run the project commands from WSL, or through `wsl bash -lc ...`.

The project is configured for Python `>=3.11,<3.13`, which matches the current WSL Python setup.

## Quick start

```bash
cd /home/maxime/projects/home-credit-mlops
poetry install
```

## Suggested workflow

1. Put the Kaggle files in `data/raw/`

2. Build the cleaned, feature-engineered dataset and the full data-preparation EDA package:

```bash
poetry run python scripts/build_home_credit_dataset.py
```

3. Run the unified experiment pipeline:

```bash
poetry run python scripts/run_home_credit_experiment.py --campaign-name dev_lightgbm_5k_cv3 --model lightgbm --sample-size 5000 --cv-folds 3 --n-jobs 1
```

Available base models are `logistic_regression`, `random_forest`, `extra_trees`,
`lightgbm`, and `xgboost`. For example, compare the two boosting implementations with:

```bash
poetry run python scripts/run_home_credit_experiment.py --campaign-name boosting_10k_cv3 --model lightgbm --model xgboost --sampling baseline --sample-size 10000 --cv-folds 3 --n-jobs 1
```

To compare imbalance handling strategies, you can benchmark the same base model with several sampling modes such as `baseline`, `smote`, `borderline_smote`, `adasyn`, and `smote_under`:

```bash
poetry run python scripts/run_home_credit_experiment.py --campaign-name dev_logreg_baseline_smote --model logistic_regression --sampling baseline --sampling smote --sample-size 5000 --cv-folds 3 --n-jobs 1
```

For WSL / VS Code stability, keep development runs focused on one base model with a sample and `--n-jobs 1`. Reserve full multi-model, multi-sampling, full-dataset benchmarks for the final phase.

4. Open the MLflow UI when you want to inspect the tracked runs:

```bash
poetry run python scripts/mlflow_ui.py
```

5. If needed, disable tracking for a quick local dry run:

```bash
poetry run python scripts/run_home_credit_experiment.py --skip-mlflow --sample-size 3000 --cv-folds 3 --n-jobs 1
```

## What the unified experiment exports

Each run under `reports/YYYYMMDD_home_credit_experiments/<timestamp>_<campaign_name>/` includes:

- a root `summary.xlsx` workbook with cross-model comparison sheets such as `campaign_overview`, `model_performance_summary`, `cv_summary`, `holdout_summary`, `decision_threshold_summary`, and `mlflow_runs`, including the `base_model` and `sampling` comparison columns
- per-folder Excel workbooks for `interpretability`, `diagnostics`, `predictions`, `cv_results`, and `threshold_optimization`
- diagnostics exported for each benchmarked candidate, with one subfolder per model or sampling variant
- interpretability exports kept for the selected best model only
- OOF and holdout prediction parquet files
- ROC, PR, and confusion-matrix diagnostics
- grouped feature importance
- SHAP global and local explanations
- campaign-level metadata and MLflow run mapping
- decision-threshold metadata
- threshold-optimization tables and plots, including business cost vs threshold
- packaged reports where CSV exports are converted to Excel tabs and then removed

## What this scaffold already covers

- reusable package structure
- feature aggregation from the main Home Credit tables
- cross-validation and hyperparameter search
- business cost with heavier false-negative penalty
- threshold optimization on out-of-fold probabilities
- SHAP-based interpretability exports
- Excel bundling of experiment artifacts
- MLflow experiment tracking and local model registry support

## Serve the versioned business decision

The final MLflow model packages the fitted pipeline together with its optimized
business threshold. Its response contains the default probability, threshold,
predicted class, and credit decision for every submitted client.

```bash
MODEL_VERSION=3  # Replace with the newly registered business-model version.

poetry run mlflow models serve \
  --model-uri "models:/home-credit-scoring/${MODEL_VERSION}" \
  --host 127.0.0.1 \
  --port 8000 \
  --env-manager local
```

The standard MLflow response uses a `predictions` envelope:

```json
{
  "predictions": [
    {
      "default_probability": 0.37,
      "business_threshold": 0.2203,
      "predicted_default": 1,
      "credit_decision": "refused"
    }
  ]
}
```
