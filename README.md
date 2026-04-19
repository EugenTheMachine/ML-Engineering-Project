# ML-Engineering-Project

## Description
Custom implementation of Lab assignments within the ML Engineering course

## Some useful cmd prompts
Before you begin working with the code locally, run this command in the PowerShell:

```cmd
$env:PYTHONPATH = (Get-Location)
```

or, alternatively:

```cmd
$env:PYTHONPATH = (Get-Location).Path
```

Then, here is the algorithm for poetry-related stuff:

```cmd
poetry lock
```

```cmd
poetry install
```

[optional]

```cmd
poetry add --group dev pre-commit@^4.5.0
```

```cmd
poetry run pre-commit install --config githooks.yml
```

```cmd
poetry run pre-commit run --all-files --config githooks.yml
```

## DVC Pipeline

This repository includes a reproducible DVC pipeline with dataset download and training stages. Model evaluation is performed as part of training, so a separate DVC evaluation stage is not required.

### Run the pipeline

```cmd
dvc repro
```

### Run individual stages

```cmd
python -m src.dataset.load_data --config src/config.yaml --output cifar10 --archive cifar-10-python.tar.gz --keep-archive --registry dataset_registry.csv
python -m src.train_eval.train --config src/config.yaml
```

## MLflow Tracking

Install MLflow (poetry updated - run install):

```bash
poetry install
```

Start MLflow UI (local file backend):

```bash
mlflow ui --backend-store-uri file:./mlruns --default-artifact-root ./mlruns
```

Then run training. You can set MLflow experiment and run name either in `src/config.yaml` (add `mlflow_experiment` and `run_name` keys) or pass them as CLI args:

```bash
python -m src.train_eval.train --config src/config.yaml --mlflow-experiment "my-experiment" --run-name "Experiment 1 - Stage 2: Model Training"
```

The training script will log config parameters, epoch-wise metrics, final test metrics, and artifacts (`best.pt`, `last.pt`, `training_history.csv`, `loss_plot.png`) to the MLflow run.

### Optional manual evaluation

```cmd
python -m src.train_eval.eval --config src/config.yaml --model experiments/trainXX/best.pt --output metrics/evaluation_metrics.json
```

### DVC remote operations

```cmd
dvc push -r localstorage
```

```cmd
dvc pull -r localstorage
```

### Update parameters

Edit `src/config.yaml` and rerun:

```cmd
dvc repro
```

The pipeline will rerun stages when configuration or code changes affect dependencies.
