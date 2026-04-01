"""
Main training script.
"""
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

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def train_model(model, train_loader, val_loader, criterion, logger, out_dir):
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

    cfg = get_cfg(CFG_PATH)
    device = torch.device(cfg["device"])
    num_epochs = cfg["epochs"]
    patience = cfg["patience"]
    lr = cfg["lr"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    epochs_no_improve = 0

    logger.info("Training started")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {num_epochs} | LR: {lr} | Patience: {patience}")

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

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
            f"""
            Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}
                            Val Acc: {val_acc:.4f}
                            Val Precision: {val_precision:.4f}
                            Val Recall: {val_recall:.4f}
                            Val F1: {val_f1:.4f}
            """.strip()
        )

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0

            torch.save(model.state_dict(), out_dir / "best.pt")
            logger.info("New best model saved")

        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} epoch(s)")

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        torch.save(model.state_dict(), out_dir / "last.pt")

        history_df = pd.DataFrame(history)
        history_df.to_csv(out_dir / "training_history.csv", index=False)

    if epochs_no_improve < patience:
        logger.info(f"Training completed after reaching max epochs: {num_epochs}")

    # Plot losses
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(out_dir / "loss_plot.png")

    logger.info("Training finished")

    return history


def train():
    """
    General wrapper for the training process.
    Sets up logging, loads data, initializes model and starts training.
    """
    os.makedirs("experiments", exist_ok=True)

    subdirs = [int(f[5:]) for f in os.listdir("experiments") if f != ".gitkeep"] + [0]
    n_past_experiments = sorted(subdirs)[-1]
    out_dir = Path("experiments") / f"train{n_past_experiments + 1:02d}"
    os.makedirs(out_dir)

    # Setup logger
    logger = setup_logger(out_dir / "train.log")

    cfg = get_cfg(CFG_PATH)
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
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = CrossEntropyLoss()
    train_history = train_model(
        model, train_loader, val_loader, criterion, logger, out_dir
    )

    model.load_state_dict(torch.load(out_dir / "best.pt"))
    test_acc, test_precision, test_recall, test_f1, test_loss = eval_model(
        model, test_loader, criterion, torch.device(cfg["device"])
    )
    shutil.copy("src/config.yaml", out_dir / "config.yaml")

    logger.info(
        f"""
            Test Loss: {test_loss:.4f} |
                            Test Acc: {test_acc:.4f} |
                            Test Precision: {test_precision:.4f} |
                            Test Recall: {test_recall:.4f} |
                            Test F1: {test_f1:.4f}
        """.strip()
    )
    logger.info(f"Experiment artifacts saved to: {out_dir}")

    # Close logging handlers (important when running multiple experiments)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    return train_history


if __name__ == "__main__":
    train()
