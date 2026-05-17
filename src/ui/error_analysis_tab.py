import logging
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from src.dataset.dataset import get_data
from src.dataset.extract_data import CIFAR10_CLASSES

from .mlflow_utils import download_model_artifact, list_experiments, list_runs
from .model_utils import build_model, load_model_weights, predict

logger = logging.getLogger(__name__)


def torch_device(cfg: dict) -> str:
    return cfg.get("device", "cpu")


def _render_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CIFAR10_CLASSES))))
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=CIFAR10_CLASSES,
            y=CIFAR10_CLASSES,
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        width=800,
        height=700,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_error_counts(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    error_counts = (
        confusion_matrix(y_true, y_pred).sum(axis=1)
        - np.diag(confusion_matrix(y_true, y_pred))
    ).tolist()
    fig = px.bar(
        x=CIFAR10_CLASSES,
        y=error_counts,
        labels={"x": "Class", "y": "Misclassification count"},
        title="Per-class misclassification counts",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_error_analysis_tab(cfg: dict) -> None:
    st.header("Error Analysis")

    experiments = list_experiments(cfg)
    if not experiments:
        st.warning(
            "No MLflow experiments found. Verify your tracking URI and run the training pipeline first."
        )
        return

    experiment_options = {exp["name"]: exp for exp in experiments}
    exp_names = list(experiment_options.keys())
    default_exp_index = (
        exp_names.index("quick-test") if "quick-test" in exp_names else 0
    )
    experiment_name = st.selectbox(
        "Select experiment", exp_names, index=default_exp_index, key="ea_experiment"
    )
    experiment = experiment_options[experiment_name]

    runs = list_runs(experiment["experiment_id"], cfg)
    if not runs:
        st.warning("No runs found for this experiment.")
        return

    run_labels = [f"{run['run_name']} ({run['run_id'][:8]})" for run in runs]
    # Prefer "Full Demo Run" if present
    default_run_index = 0
    for i, run in enumerate(runs):
        if run.get("run_name") and "Full Demo Run" in run.get("run_name"):
            default_run_index = i
            break
    run_choice = st.selectbox(
        "Select run", run_labels, index=default_run_index, key="ea_run"
    )
    selected_run = runs[run_labels.index(run_choice)]
    run_id = selected_run["run_id"]

    with st.expander("Run details", expanded=False):
        st.json(
            {
                "run_id": run_id,
                "metrics": selected_run["metrics"],
                "params": selected_run["params"],
            }
        )

    model_path = download_model_artifact(run_id, cfg)
    if model_path is None:
        st.error("Could not find a saved model artifact for the selected run.")
        return

    model = build_model(cfg)
    try:
        load_model_weights(model, Path(model_path), device=torch_device(cfg))
    except Exception as exc:
        st.error(f"Could not load model artifact: {exc}")
        return

    _, _, test_ds = get_data()
    test_loader = DataLoader(
        test_ds, batch_size=cfg.get("batch_size", 128), shuffle=False
    )

    y_true = []
    y_pred = []
    probabilities = []

    for inputs, labels in test_loader:
        probs, predicted = predict(model, inputs, device=torch_device(cfg))
        y_true.extend(labels.numpy().tolist())
        if isinstance(predicted, np.ndarray):
            y_pred.extend(predicted.tolist())
        else:
            y_pred.append(predicted)
        probabilities.extend(probs.tolist())

    y_pred = np.array(y_pred, dtype=int)
    y_true = np.array(y_true, dtype=int)
    probabilities = np.stack(probabilities, axis=0)

    if y_true.size == 0:
        st.warning("No test predictions were produced.")
        return

    _render_confusion_matrix(y_true, y_pred)
    _render_error_counts(y_true, y_pred)

    misclassified = np.where(y_true != y_pred)[0]
    if misclassified.size == 0:
        st.success("No misclassified samples found for this run.")
        return

    sort_mode = st.radio(
        "Sort misclassified examples by",
        ["Highest confidence", "Lowest confidence", "Predicted class"],
    )
    mis_confidences = probabilities[np.arange(len(y_pred)), y_pred][misclassified]
    mis_preds = y_pred[misclassified]

    if sort_mode == "Highest confidence":
        order = np.argsort(-mis_confidences)
    elif sort_mode == "Lowest confidence":
        order = np.argsort(mis_confidences)
    else:
        order = np.argsort(mis_preds)

    selected_errors = misclassified[order][: min(12, misclassified.size)]
    st.subheader("Misclassified examples")

    columns = st.columns(4)
    for index, example_idx in enumerate(selected_errors):
        raw_image = test_ds.images[example_idx]
        true_label = CIFAR10_CLASSES[y_true[example_idx]]
        predicted_label = CIFAR10_CLASSES[y_pred[example_idx]]
        confidence = probabilities[example_idx, y_pred[example_idx]]
        with columns[index % 4]:
            st.image(
                raw_image,
                caption=f"True: {true_label}\nPredicted: {predicted_label}\nConf: {confidence:.2f}",
            )

    if len(selected_errors) > 0:
        st.markdown("---")
        st.write(
            "Sorting by prediction confidence and predicted class gives a reproducible view of the most important failure modes."
        )
