"""
This module contains utility functions for training and evaluating the model.
"""

import numpy as np
import torch
from tqdm import tqdm


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Input args:
    - model: The PyTorch model to be trained.
    - train_loader: DataLoader providing the training data.
    - criterion: Loss function used for training.
    - optimizer: Optimization algorithm for updating model weights.
    - device: Device (CPU or GPU) to perform computations on.
    """

    total_loss, total = 0.0, 0
    model.train()
    model = model.to(device).float()

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device).float(), labels.to(device).long()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += 1

    epoch_loss = total_loss / total
    return epoch_loss


def inference_epoch(model, eval_loader, criterion, device):
    """
    Executes model inference during evaluation phase (validation or final testing)
    Input args:
    - model: The PyTorch model to be evaluated.
    - eval_loader: DataLoader providing the evaluation data.
    - criterion: Loss function used for evaluation.
    - device: Device (CPU or GPU) to perform computations on.
    - threshold: Classification threshold for binary prediction.
     Returns:
    - y_true: Ground truth labels.
    - y_pred: Predicted labels.
    - total_loss: Total loss over the evaluation set.
    """
    model.eval()
    model = model.to(device).float()
    y_true, y_pred = [], []
    total_loss = []
    with torch.no_grad():
        for inputs, labels in tqdm(eval_loader, desc="Evaluating"):
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            y_true += labels
            outputs = model(inputs)
            total_loss += [criterion(outputs, labels).item()]
            y_pred += outputs.argmax(dim=1)
    return torch.stack(y_true), torch.stack(y_pred), np.mean(np.array(total_loss))
