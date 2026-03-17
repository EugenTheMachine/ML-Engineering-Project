"""
This module contains the evaluation function for the trained model.
It evaluates the model on the validation or test set and computes performance metrics such as
accuracy, precision, recall, and F1 score. The function takes the trained model,
evaluation data loader, loss criterion, and device as input and returns the computed metrics.
"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.train_eval.utils import inference_epoch


def eval_model(model, eval_loader, criterion, device):
    """
    Evaluates the model on the validation or test set and computes performance metrics.
        Input args:
    - model: The PyTorch model to be evaluated.
    - eval_loader: DataLoader providing the evaluation data.
    - criterion: Loss function used for evaluation.
    - threshold: Classification threshold for binary prediction.
        Returns:
    - acc: Accuracy of the model on the evaluation set.
    - precision: Precision of the model on the evaluation set.
    - recall: Recall of the model on the evaluation set.
    - f1: F1 score of the model on the evaluation set.
    - loss: Mean loss over the evaluation set.
    """
    acc, precision, recall, f1, total_loss = [], [], [], [], []

    y_true, y_pred, loss = inference_epoch(model, eval_loader, criterion, device)
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    total_loss = loss

    return acc, precision, recall, f1, total_loss
