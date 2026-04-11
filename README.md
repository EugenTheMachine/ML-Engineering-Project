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
