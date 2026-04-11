"""
This module contains the data augmentation pipeline for training and evaluation.
It uses the torchvision.transforms.v2 library to define a series of transformations
that can be applied to the input images. For training, it uses TrivialAugmentWide
to improve model generalization. For evaluation, only normalization and conversion
to tensor are applied.
"""

import torch
from torchvision.transforms import v2


class TorchvisionWrapper:
    """
    A wrapper to make torchvision transforms compatible with the calling
    convention of albumentations (keyword arguments and dictionary output).
    This is to ensure compatibility with the existing Dataset implementation.
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, *, image):
        """
        Applies the torchvision transform.

        Args:
            image: The input image (numpy array).

        Returns:
            A dictionary {'image': transformed_tensor}.
        """
        return {"image": self.transform(image)}


def get_augmentation_pipeline(subset: str):
    """
    Creates a data augmentation pipeline using torchvision.

    For the training set, it applies TrivialAugmentWide.
    For validation/test sets, it only applies normalization.

    Args:
        subset (str): 'train' for training pipeline, otherwise evaluation pipeline.

    Returns:
        A callable transform object.
    """
    # Normalization stats for models pretrained on ImageNet
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if subset != "train":
        pipeline = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )
    else:  # subset == "train"
        pipeline = v2.Compose(
            [
                v2.ToImage(),
                v2.TrivialAugmentWide(interpolation=v2.InterpolationMode.BILINEAR),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )

    return TorchvisionWrapper(pipeline)
