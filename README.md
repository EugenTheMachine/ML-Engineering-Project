# ML-Engineering-Project

## Description
Custom implementation of Lab assignments within the ML Engineering course

## Some useful cmd prompts
Before you begin working with the code locally, run this command in the PowerShell:

```cmd
$env:PYTHONPATH = (Get-Location)
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
