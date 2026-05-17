import logging
from pathlib import Path
from typing import List, Optional

import mlflow
from mlflow.entities import FileInfo
from mlflow.tracking import MlflowClient

from src.utils import get_cfg

logger = logging.getLogger(__name__)


def configure_mlflow(cfg: dict | None = None) -> MlflowClient:
    if cfg is None:
        cfg = get_cfg()

    tracking_uri = cfg.get("mlflow_tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri("file:./mlruns")

    logger.info("MLflow tracking URI set to %s", mlflow.get_tracking_uri())
    return MlflowClient()


def get_mlflow_client(cfg: dict | None = None) -> MlflowClient:
    return configure_mlflow(cfg)


def list_experiments(cfg: dict | None = None) -> list[dict]:
    client = get_mlflow_client(cfg)
    # Mlflow client API differs across versions. Try common methods with fallbacks.
    experiments = None
    try:
        experiments = client.list_experiments()
    except AttributeError:
        try:
            experiments = client.search_experiments()
        except Exception:
            # Fallback to mlflow module helper
            try:
                experiments = mlflow.list_experiments()
            except Exception:
                raise

    result = []
    for exp in experiments:
        # exp may be an Experiment object or ExperimentInfo
        exp_id = getattr(exp, "experiment_id", None) or getattr(
            exp, "experiment_id", None
        )
        name = getattr(exp, "name", None)
        artifact_location = getattr(exp, "artifact_location", None)
        result.append(
            {
                "experiment_id": str(exp_id),
                "name": name,
                "artifact_location": artifact_location,
            }
        )

    return result


def list_runs(experiment_id: str, cfg: dict | None = None) -> list[dict]:
    client = get_mlflow_client(cfg)
    runs = client.search_runs(
        experiment_ids=[experiment_id], order_by=["attributes.start_time desc"]
    )
    return [
        {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name or run.info.run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
        }
        for run in runs
    ]


def _find_model_artifact_path(
    run_id: str, client: MlflowClient, path: str = ""
) -> Optional[str]:
    artifacts: List[FileInfo] = client.list_artifacts(run_id, path)
    for artifact in artifacts:
        if artifact.is_dir:
            nested = _find_model_artifact_path(run_id, client, artifact.path)
            if nested:
                return nested
        elif artifact.path.endswith(".pt"):
            return artifact.path

    return None


def download_model_artifact(run_id: str, cfg: dict | None = None) -> Optional[Path]:
    client = get_mlflow_client(cfg)
    artifact_path = _find_model_artifact_path(run_id, client)
    if artifact_path is None:
        logger.warning("No PyTorch model artifact (*.pt) found for run %s", run_id)
        return None

    try:
        local_path = client.download_artifacts(run_id, artifact_path)
        local_path = Path(local_path)
        logger.info("Downloaded model artifact for run %s to %s", run_id, local_path)
        if local_path.is_dir():
            candidates = list(local_path.rglob("*.pt"))
            if candidates:
                return candidates[0]
            return None
        return local_path
    except Exception:
        logger.exception("Failed to download model artifact for run %s", run_id)
        raise


def get_run_metrics(run_id: str, cfg: dict | None = None) -> dict:
    client = get_mlflow_client(cfg)
    run = client.get_run(run_id)
    return dict(run.data.metrics)
