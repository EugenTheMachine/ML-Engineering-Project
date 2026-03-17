"""
This module contains utilities for loading and processing CIFAR-10 dataset batches.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_batch(batch_path: Path) -> Dict:
    """
    Loads a single CIFAR-10 batch file.

    Args:
        batch_path (Path): Path to the batch file (pickle format).

    Returns:
        Dict: Dictionary containing 'data', 'labels', 'batch_label', and 'filenames'.

    Raises:
        FileNotFoundError: If the batch file doesn't exist.
        pickle.UnpicklingError: If the file is not a valid pickle file.
    """
    try:
        with open(batch_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        logger.info(f"Successfully loaded batch from {batch_path}")
        return batch
    except FileNotFoundError:
        logger.error(f"Batch file not found: {batch_path}")
        raise
    except pickle.UnpicklingError as e:
        logger.error(f"Error unpickling batch file {batch_path}: {e}")
        raise


def load_all_batches(
    data_dir: Path, subset: str = "train"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads all training or test batches from CIFAR-10 dataset.

    Args:
        data_dir (Path): Path to the CIFAR-10 directory.
        subset (str): Either 'train' for training batches or 'test' for test batch.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Images and labels arrays.
                                       Images shape: (N, 3072) -> reshape to (N, 3, 32, 32)
                                       Labels shape: (N,)

    Raises:
        ValueError: If subset is not 'train' or 'test'.
        FileNotFoundError: If batch files don't exist.
    """
    if subset not in ["train", "test"]:
        raise ValueError("subset must be either 'train' or 'test'")

    all_images = []
    all_labels = []

    try:
        if subset == "train":
            # Load all 5 training batches
            for i in range(1, 6):
                batch_path = data_dir / f"data_batch_{i}"
                batch = load_batch(batch_path)
                all_images.append(batch[b"data"])
                all_labels.extend(batch[b"labels"])
            logger.info("Successfully loaded all training batches")
        else:
            # Load test batch
            batch_path = data_dir / "test_batch"
            batch = load_batch(batch_path)
            all_images.append(batch[b"data"])
            all_labels = batch[b"labels"]
            logger.info("Successfully loaded test batch")

        # Concatenate all batches into single arrays
        images = np.concatenate(all_images)
        labels = np.array(all_labels)

        return images, labels

    except Exception as e:
        logger.error(f"Error loading {subset} batches: {e}")
        raise


def extract_cifar10_data(
    data_dir: Path, subset: str = "train"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads CIFAR-10 data with optional preprocessing.

    Args:
        data_dir (Path): Path to CIFAR-10 directory.
        subset (str): Either 'train' or 'test'.
        reshape (bool): Whether to reshape images to (N, 3, 32, 32).
        normalize (bool): Whether to normalize images to [0, 1] range.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Processed images and labels.

    Example:
        >>> images, labels = get_cifar10_data(Path('data/cifar-10-batches-py'))
        >>> images.shape
        (50000, 3, 32, 32)
        >>> labels.shape
        (50000,)
    """
    images, labels = load_all_batches(data_dir, subset)

    logger.info(f"CIFAR-10 {subset} data loaded with shape: {images.shape}")
    return images, labels
