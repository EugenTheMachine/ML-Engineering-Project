from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image

from src.dataset.dataset import get_data
from src.dataset.extract_data import CIFAR10_CLASSES

from .mlflow_utils import download_model_artifact, list_experiments, list_runs
from .model_utils import (
    build_model,
    generate_gradcam_overlay,
    get_class_name,
    load_model_weights,
    predict,
    prepare_image,
)


def _format_probability_chart(probabilities: np.ndarray) -> None:
    fig = px.bar(
        x=CIFAR10_CLASSES,
        y=probabilities,
        labels={"x": "Class", "y": "Probability"},
        title="Prediction probability distribution",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_prediction_tab(cfg: dict) -> None:
    st.header("Prediction & Explainability")

    experiments = list_experiments(cfg)
    if not experiments:
        st.warning("No MLflow experiments found for prediction.")
        return

    experiment_options = {exp["name"]: exp for exp in experiments}
    exp_names = list(experiment_options.keys())
    default_exp_index = (
        exp_names.index("quick-test") if "quick-test" in exp_names else 0
    )
    experiment_name = st.selectbox(
        "Select experiment", exp_names, index=default_exp_index, key="pred_experiment"
    )
    experiment = experiment_options[experiment_name]

    runs = list_runs(experiment["experiment_id"], cfg)
    if not runs:
        st.warning("No runs available for this experiment.")
        return

    run_labels = [f"{run['run_name']} ({run['run_id'][:8]})" for run in runs]
    default_run_index = 0
    for i, run in enumerate(runs):
        if run.get("run_name") and "Full Demo Run" in run.get("run_name"):
            default_run_index = i
            break
    run_choice = st.selectbox(
        "Select run", run_labels, index=default_run_index, key="pred_run"
    )
    selected_run = runs[run_labels.index(run_choice)]
    run_id = selected_run["run_id"]

    model_path = download_model_artifact(run_id, cfg)
    if model_path is None:
        st.error("A model artifact could not be located for the selected run.")
        return

    model = build_model(cfg)
    try:
        load_model_weights(model, Path(model_path), device=cfg.get("device", "cpu"))
    except Exception as exc:
        st.error(f"Unable to load model weights: {exc}")
        return

    source = st.radio("Inference source", ["Dataset sample", "Upload image"])
    raw_image = None
    sample_info = None

    if source == "Dataset sample":
        _, _, test_ds = get_data()
        sample_index = st.slider("Test sample index", 0, len(test_ds) - 1, 0)
        raw_image = test_ds.images[sample_index]
        sample_info = {
            "true_label": CIFAR10_CLASSES[int(test_ds.labels[sample_index])],
            "sample_index": int(sample_index),
        }
    else:
        image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if image_file is None:
            st.info("Upload an image to run inference.")
            return
        raw_image = np.asarray(Image.open(image_file).convert("RGB"))

    input_tensor, preview_image = prepare_image(raw_image, cfg)
    probabilities, predicted = predict(
        model, input_tensor, device=cfg.get("device", "cpu")
    )
    predicted_label = get_class_name(predicted)

    st.subheader("Prediction results")
    left, right = st.columns(2)
    with left:
        st.image(raw_image, caption="Input image", use_column_width=True)
        st.markdown(f"**Predicted label:** {predicted_label}")
        if sample_info:
            st.markdown(f"**True label:** {sample_info['true_label']}")
            st.markdown(f"**Index:** {sample_info['sample_index']}")
    with right:
        _format_probability_chart(probabilities)

    explain_target = st.selectbox(
        "Explain class",
        [f"Top prediction ({predicted_label})"] + CIFAR10_CLASSES,
        index=0,
        key="pred_explain_target",
    )
    target_class = (
        predicted
        if explain_target.startswith("Top prediction")
        else CIFAR10_CLASSES.index(explain_target)
    )

    try:
        overlay = generate_gradcam_overlay(
            model, input_tensor, preview_image, target_category=target_class
        )
        st.subheader("Grad-CAM explanation")
        st.image(
            overlay,
            caption=f"Grad-CAM overlay for class: {get_class_name(target_class)}",
            use_column_width=True,
        )
    except Exception as exc:
        st.error(f"Unable to create Grad-CAM explanation: {exc}")
