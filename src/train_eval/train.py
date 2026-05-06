"""
Main training script.
"""
import argparse
import logging
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, regnet_y_400mf, resnet18

import wandb
from src.dataset.dataset import get_data
from src.train_eval.eval import eval_model
from src.train_eval.utils import train_epoch
from src.utils import CFG_PATH, get_cfg


def setup_logger(log_path):
    """
    Configure logger to write both to console and file.
    """
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def init_wandb(cfg: dict, run_name: str, logger: logging.Logger) -> bool:
    """Initialize Weights & Biases for the current experiment run."""
    project_name = cfg.get("mlflow_experiment", "default")
    wandb_config = {
        k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))
    }
    try:
        wandb.init(
            project=project_name,
            name=run_name,
            config=wandb_config,
            reinit=True,
            entity="lazoskal24-national-technical-university-kharkiv-polytec",
        )
        return True
    except Exception as exc:
        logger.warning("W&B initialization failed: %s", exc)
        return False


def train_model(
    model, train_loader, val_loader, criterion, logger, out_dir, cfg, use_wandb: bool
):
    """
    Main training loop for the model.
    """

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    device = torch.device(cfg["device"])
    num_epochs = cfg["epochs"]
    patience = cfg["patience"]
    lr = cfg["lr"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    epochs_no_improve = 0

    logger.info("Training started")
    logger.info("Device: %s", device)
    logger.info("Epochs: %s | LR: %s | Patience: %s", num_epochs, lr, patience)

    for epoch in range(num_epochs):
        logger.info("Epoch %s/%s", epoch + 1, num_epochs)

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_precision, val_recall, val_f1, val_loss = eval_model(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)

        logger.info(
            "Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f | Val Precision: %.4f | Val Recall: %.4f | Val F1: %.4f",
            train_loss,
            val_loss,
            val_acc,
            val_precision,
            val_recall,
            val_f1,
        )

        # Log epoch metrics to MLflow (step = epoch)
        try:
            mlflow.log_metric("train_loss", float(train_loss), step=epoch)
            mlflow.log_metric("val_loss", float(val_loss), step=epoch)
            mlflow.log_metric("val_acc", float(val_acc), step=epoch)
            mlflow.log_metric("val_precision", float(val_precision), step=epoch)
            mlflow.log_metric("val_recall", float(val_recall), step=epoch)
            mlflow.log_metric("val_f1", float(val_f1), step=epoch)
        except Exception:
            pass

        if use_wandb:
            try:
                wandb.log(
                    {
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                        "val_acc": float(val_acc),
                        "val_precision": float(val_precision),
                        "val_recall": float(val_recall),
                        "val_f1": float(val_f1),
                    },
                    step=epoch,
                )
            except Exception:
                pass

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0

            torch.save(model.state_dict(), out_dir / "best.pt")
            logger.info("New best model saved")
        else:
            epochs_no_improve += 1
            logger.info("No improvement for %s epoch(s)", epochs_no_improve)

            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered after %s epochs", epoch + 1)
                break

        torch.save(model.state_dict(), out_dir / "last.pt")

        history_df = pd.DataFrame(history)
        history_df.to_csv(out_dir / "training_history.csv", index=False)

    if epochs_no_improve < patience:
        logger.info("Training completed after reaching max epochs: %s", num_epochs)

    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(out_dir / "loss_plot.png")

    logger.info("Training finished")
    return history


def get_next_experiment_dir(
    base_path: str = "experiments", prefix: str = "train"
) -> Path:
    root = Path(base_path)
    root.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(prefix)}(\d{{2}})$")
    existing_indices = []

    for path in root.iterdir():
        if path.is_dir():
            match = pattern.match(path.name)
            if match:
                existing_indices.append(int(match.group(1)))

    next_index = max(existing_indices, default=0) + 1
    return root / f"{prefix}{next_index:02d}"


def train(
    config_path: str = str(CFG_PATH),
    output_dir: str | None = None,
    mlflow_experiment: str | None = None,
    run_name: str | None = None,
    tracking_uri: str | None = None,
):
    """
    General wrapper for the training process.
    Sets up logging, loads data, initializes model and starts training.
    """
    cfg = get_cfg(config_path)

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = get_next_experiment_dir("experiments")
        out_dir.mkdir(parents=True, exist_ok=False)

    logger = setup_logger(out_dir / "train.log")

    model_name = cfg["model_name"]
    num_classes = cfg["num_classes"]
    batch_size = cfg["batch_size"]

    if model_name == "resnet18":
        model = resnet18(num_classes=num_classes)
    elif model_name == "regnet_y_400mf":
        model = regnet_y_400mf(num_classes=num_classes)
    elif model_name == "mobilenet_v2":
        model = mobilenet_v2(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. Supported: 'resnet18', 'regnet_y_400mf', 'mobilenet_v2'"
        )

    train_ds, val_ds, test_ds = get_data()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    criterion = CrossEntropyLoss()
    # allow overriding MLflow settings from function args
    if mlflow_experiment:
        cfg["mlflow_experiment"] = mlflow_experiment
    if run_name:
        cfg["run_name"] = run_name
    if tracking_uri:
        cfg["mlflow_tracking_uri"] = tracking_uri

    # MLflow experiment setup
    tracking_uri = cfg.get("mlflow_tracking_uri", None)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # default local mlruns folder
        mlflow.set_tracking_uri("file:./mlruns")

    experiment_name = cfg.get("mlflow_experiment", "default")
    mlflow.set_experiment(experiment_name)

    run_name = cfg.get("run_name") or f"{out_dir.name} - Stage: Model Training"

    with mlflow.start_run(run_name=run_name):
        # Log configuration parameters
        for k, v in cfg.items():
            try:
                # mlflow expects primitives; serialize others
                if isinstance(v, (int, float, str, bool)):
                    mlflow.log_param(k, v)
                else:
                    mlflow.log_param(k, str(v))
            except Exception:
                pass

        use_wandb = init_wandb(cfg, run_name, logger)

        train_history = train_model(
            model, train_loader, val_loader, criterion, logger, out_dir, cfg, use_wandb
        )

        # Load best model and evaluate on test set
        model.load_state_dict(torch.load(out_dir / "best.pt"))
        test_acc, test_precision, test_recall, test_f1, test_loss = eval_model(
            model, test_loader, criterion, torch.device(cfg["device"])
        )

        # Log test metrics
        try:
            mlflow.log_metric("test_loss", float(test_loss))
            mlflow.log_metric("test_acc", float(test_acc))
            mlflow.log_metric("test_precision", float(test_precision))
            mlflow.log_metric("test_recall", float(test_recall))
            mlflow.log_metric("test_f1", float(test_f1))
        except Exception:
            pass

        # Log artifacts: model weights and training files
        try:
            mlflow.log_artifact(str(out_dir / "best.pt"))
            mlflow.log_artifact(str(out_dir / "last.pt"))
        except Exception:
            pass

        for fname in [
            "training_history.csv",
            "loss_plot.png",
            "train.log",
            Path(config_path).name,
        ]:
            p = out_dir / fname
            if p.exists():
                try:
                    mlflow.log_artifact(str(p))
                except Exception:
                    pass

        if use_wandb:
            try:
                artifact = wandb.Artifact("model-artifact", type="model")
                for fname in [
                    "best.pt",
                    "last.pt",
                    "training_history.csv",
                    "loss_plot.png",
                    "train.log",
                    Path(config_path).name,
                ]:
                    p = out_dir / fname
                    if p.exists():
                        artifact.add_file(str(p))
                wandb.log_artifact(artifact)
            except Exception:
                pass

    # end mlflow run

    if use_wandb:
        wandb.finish()

    # copy config into experiment artifacts directory (already logged to MLflow)
    shutil.copy(config_path, out_dir / Path(config_path).name)

    logger.info("Experiment artifacts saved to: %s", out_dir)

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    return train_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CIFAR-10 model.")
    parser.add_argument(
        "--config", type=str, default=str(CFG_PATH), help="Path to the parameter file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output directory for training artifacts.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name to use (overrides config).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name to use (overrides config).",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (overrides config).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        config_path=args.config,
        output_dir=args.output,
        mlflow_experiment=args.mlflow_experiment,
        run_name=args.run_name,
        tracking_uri=args.mlflow_tracking_uri,
    )
