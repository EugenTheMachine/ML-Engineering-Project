"""
Evaluation utilities for trained CIFAR-10 models.
This module can be used as a library or as a DVC stage entrypoint.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Union

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, regnet_y_400mf, resnet18

from src.dataset.dataset import get_data
from src.train_eval.utils import inference_epoch
from src.utils import CFG_PATH, get_cfg


def setup_logger(log_path: Path):
    logger = logging.getLogger("evaluation_logger")
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


def eval_model(model, eval_loader, criterion, device):
    y_true, y_pred, loss = inference_epoch(model, eval_loader, criterion, device)
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    total_loss = loss

    return acc, precision, recall, f1, total_loss


def write_metrics(metrics: dict, output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def evaluate(config_path: str, model_path: str, output_path: str) -> dict:
    cfg = get_cfg(config_path)
    device = torch.device(cfg["device"])
    model_name = cfg["model_name"]
    num_classes = cfg["num_classes"]

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

    _, _, test_ds = get_data()
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    acc, precision, recall, f1, loss = eval_model(model, test_loader, criterion, device)

    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
    }

    write_metrics(metrics, Path(output_path))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CIFAR-10 model.")
    parser.add_argument(
        "--config", type=str, default=str(CFG_PATH), help="Path to the parameter file."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained model file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics/evaluation_metrics.json",
        help="Path to write evaluation metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log_path = Path(args.output).parent / "eval.log"
    logger = setup_logger(log_path)

    logger.info("Starting evaluation with model %s", args.model)
    metrics = evaluate(args.config, args.model, args.output)
    logger.info("Evaluation completed: %s", metrics)

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
