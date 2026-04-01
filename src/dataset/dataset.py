"""
This module defines the CIFAR-10 dataset class and a function to load and preprocess the data.
It uses PyTorch's Dataset class, and applies augmentations using Albumentations.
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.dataset.augmentations import get_augmentation_pipeline
from src.dataset.extract_data import extract_cifar10_data
from src.utils import get_cfg


class CIFAR10Dataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images.reshape(-1, 3, 32, 32)
        self.images = np.transpose(
            self.images, (0, 2, 3, 1)
        )  # Convert to HWC format for Albumentations
        self.labels = labels.astype(np.int32)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label


def get_data() -> tuple:
    """
    Loads CIFAR-10 data and splits it into training and validation sets.

    Returns:
        tuple: (train_images, train_labels), (val_images, val_labels)
    """
    cfg = get_cfg("src/config.yaml")
    data_dir = Path(cfg["data_dir"])

    # Load training data
    images, labels = extract_cifar10_data(data_dir, subset="train")

    # keep only a fraction of the training data for training
    images, _, labels, _ = train_test_split(
        images,
        labels,
        test_size=1 - cfg["train_data_ratio"],
        random_state=cfg["seed"],
        stratify=labels,
    )

    # NOTE: add barplots showing quality metrics for 10%/20%/30%/... data used for training

    # Split into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        images,
        labels,
        test_size=cfg["val_ratio"],
        random_state=cfg["seed"],
        stratify=labels,
    )

    # Load test data
    images, labels = extract_cifar10_data(data_dir, subset="test")

    train_transforms = get_augmentation_pipeline(subset="train")
    test_transforms = get_augmentation_pipeline(subset="test")

    train_dataset = CIFAR10Dataset(
        train_images, train_labels, transform=train_transforms
    )
    val_dataset = CIFAR10Dataset(val_images, val_labels, transform=test_transforms)
    test_dataset = CIFAR10Dataset(images, labels, transform=test_transforms)

    return train_dataset, val_dataset, test_dataset
