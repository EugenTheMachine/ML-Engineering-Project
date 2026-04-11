"""
Main training script.
"""
import argparse
import logging
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, regnet_y_400mf, resnet18

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


def train_model(model, train_loader, val_loader, criterion, logger, out_dir, cfg):
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


def train(config_path: str = str(CFG_PATH), output_dir: str | None = None):
    """
    General wrapper for the training process.
    Sets up logging, loads data, initializes model and starts training.
    """
    cfg = get_cfg(config_path)

    if output_dir:
        out_dir = Path(output_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        os.makedirs("experiments", exist_ok=True)
        subdirs = [
            int(f[5:])
            for f in os.listdir("experiments")
            if f.startswith("train") and f[5:].isdigit()
        ] + [0]
        n_past_experiments = sorted(subdirs)[-1]
        out_dir = Path("experiments") / f"train{n_past_experiments + 1:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

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
    train_history = train_model(
        model, train_loader, val_loader, criterion, logger, out_dir, cfg
    )

    model.load_state_dict(torch.load(out_dir / "best.pt"))
    test_acc, test_precision, test_recall, test_f1, test_loss = eval_model(
        model, test_loader, criterion, torch.device(cfg["device"])
    )

    shutil.copy(config_path, out_dir / Path(config_path).name)

    logger.info(
        "Test Loss: %.4f | Test Acc: %.4f | Test Precision: %.4f | Test Recall: %.4f | Test F1: %.4f",
        test_loss,
        test_acc,
        test_precision,
        test_recall,
        test_f1,
    )
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(config_path=args.config, output_dir=args.output)
