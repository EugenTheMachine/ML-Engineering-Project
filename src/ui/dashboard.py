import logging

import streamlit as st

from src.utils import get_cfg

from .dataset_tab import render_dataset_exploration_tab
from .error_analysis_tab import render_error_analysis_tab
from .mlflow_utils import configure_mlflow
from .prediction_tab import render_prediction_tab

logger = logging.getLogger(__name__)


def run_app() -> None:
    st.set_page_config(
        page_title="CIFAR-10 Model Explorer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("CIFAR-10 Interactive Model Analysis Dashboard")
    st.markdown(
        "This dashboard is built for CIFAR-10 image classification analysis. "
        "Use the tabs below to explore the dataset, inspect model failures, and generate Grad-CAM explanations."
    )

    cfg = get_cfg()
    try:
        configure_mlflow(cfg)
    except Exception as exc:
        st.error(f"Unable to configure MLflow: {exc}")
        logger.exception("Failed to configure MLflow")
        return

    tabs = st.tabs(
        ["Dataset Exploration", "Error Analysis", "Prediction & Explainability"]
    )

    with tabs[0]:
        render_dataset_exploration_tab(cfg)

    with tabs[1]:
        render_error_analysis_tab(cfg)

    with tabs[2]:
        render_prediction_tab(cfg)


if __name__ == "__main__":
    run_app()
