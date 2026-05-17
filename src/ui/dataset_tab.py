from collections import Counter

import numpy as np
import plotly.express as px
import streamlit as st

from src.dataset.dataset import get_data
from src.dataset.extract_data import CIFAR10_CLASSES


def render_dataset_exploration_tab(cfg: dict) -> None:
    st.header("Dataset Exploration")

    train_ds, val_ds, test_ds = get_data()
    dataset_sizes = {
        "Train": len(train_ds),
        "Validation": len(val_ds),
        "Test": len(test_ds),
    }

    cols = st.columns(3)
    for index, (name, size) in enumerate(dataset_sizes.items()):
        cols[index].metric(label=f"{name} samples", value=size)

    label_counts = Counter(train_ds.labels.tolist())
    class_names = CIFAR10_CLASSES
    distribution = [label_counts.get(i, 0) for i in range(len(class_names))]

    chart = px.bar(
        x=class_names,
        y=distribution,
        labels={"x": "Class", "y": "Count"},
        title="Training set class distribution",
    )
    st.plotly_chart(chart, use_container_width=True)

    st.markdown("---")
    st.subheader("Sample Inspection")

    split = st.selectbox(
        "Dataset split", ["train", "validation", "test"], index=0, key="ds_split"
    )
    datasets = {"train": train_ds, "validation": val_ds, "test": test_ds}
    labels = {
        "train": train_ds.labels,
        "validation": val_ds.labels,
        "test": test_ds.labels,
    }
    selected_dataset = datasets[split]
    selected_labels = labels[split]

    class_filter = st.selectbox(
        "Filter by class", ["All"] + class_names, key="ds_class_filter"
    )
    if class_filter != "All":
        selected_class_id = class_names.index(class_filter)
        available_indices = np.where(selected_labels == selected_class_id)[0]
    else:
        available_indices = np.arange(len(selected_dataset))

    if len(available_indices) == 0:
        st.warning("No examples found for the selected class in this split.")
        return

    sample_index = st.slider(
        "Sample index", min_value=0, max_value=len(available_indices) - 1, value=0
    )
    selected_idx = int(available_indices[sample_index])
    image = selected_dataset.images[selected_idx]
    label = selected_dataset.labels[selected_idx]

    st.image(
        image, caption=f"Label: {class_names[label]} ({label})", use_column_width=False
    )
    st.write("**Raw sample metadata**")
    st.json(
        {
            "split": split,
            "sample_index": selected_idx,
            "label_name": class_names[label],
            "label_id": int(label),
        }
    )
