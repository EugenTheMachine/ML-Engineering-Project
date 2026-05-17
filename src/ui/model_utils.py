import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision.models import mobilenet_v2, regnet_y_400mf, resnet18

from src.dataset.augmentations import get_augmentation_pipeline
from src.dataset.extract_data import CIFAR10_CLASSES
from src.utils import get_cfg

logger = logging.getLogger(__name__)


def build_model(
    cfg: dict | None = None, device: Optional[torch.device] = None
) -> torch.nn.Module:
    if cfg is None:
        cfg = get_cfg()

    model_name = cfg.get("model_name", "regnet_y_400mf")
    num_classes = int(cfg.get("num_classes", 10))

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

    if device is not None:
        model = model.to(device)
    return model


def load_model_weights(
    model: torch.nn.Module, model_path: Path, device: Optional[torch.device] = None
) -> torch.nn.Module:
    if device is None:
        cfg = get_cfg()
        device = torch.device(cfg.get("device", "cpu"))
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info("Loaded model weights from %s", model_path)
        return model
    except Exception:
        logger.exception("Failed to load model weights from %s", model_path)
        raise


def prepare_image(
    input_image: Image.Image | np.ndarray, cfg: dict | None = None
) -> Tuple[torch.Tensor, np.ndarray]:
    if cfg is None:
        cfg = get_cfg()

    if isinstance(input_image, np.ndarray):
        image = Image.fromarray(input_image.astype(np.uint8))
    else:
        image = input_image.convert("RGB")

    image = image.resize((32, 32), Image.BILINEAR)
    image_np = np.asarray(image)
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    if image_np.shape[2] == 4:
        image_np = image_np[..., :3]

    pipeline = get_augmentation_pipeline("test")
    tensor = pipeline(image=image_np)["image"]
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)

    tensor = tensor.unsqueeze(0)
    return tensor, image_np.astype(np.float32) / 255.0


def predict(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, int]:
    if device is None:
        cfg = get_cfg()
        device = torch.device(cfg.get("device", "cpu"))

    model = model.to(device)
    input_tensor = input_tensor.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        predicted = np.argmax(probabilities, axis=1)
        if probabilities.shape[0] == 1:
            return probabilities[0], int(predicted[0])
        return probabilities, predicted


def generate_gradcam_overlay(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    raw_image: np.ndarray,
    target_category: Optional[int] = None,
    use_cuda: bool = False,
) -> np.ndarray:
    def GradCAM(img, cl_sc, f_ex, classification):
        f_map = f_ex(img)
        _, N, H, W = f_map.size()
        c_score = classification(f_map)[0, cl_sc]
        grads = torch.autograd.grad(c_score, f_map)
        w = grads[0][0].mean(-1).mean(-1)
        gradcam = torch.matmul(w, f_map.view(N, H * W))
        gradcam = gradcam.view(H, W).cpu().detach().numpy()
        gradcam = np.maximum(gradcam, 0)
        return gradcam

    class Flatten(nn.Module):
        def __init__(self):
            super(Flatten, self).__init__()

        def forward(self, x):
            return x.view(x.size(0), -1)

    f_ex = nn.Sequential(*list(model.children())[:-2])
    classification = nn.Sequential(
        *(list(model.children())[-2:-1] + [Flatten()] + list(model.children())[-1:])
    )
    model = model.eval()
    tensor, _ = prepare_image(raw_image)
    pred = Variable(tensor, requires_grad=True)
    prob, class_cl = torch.topk(
        nn.Softmax(dim=1)(model(pred)), 3
    )  # Returns the k largest elements of the given input tensor along a given dimension.
    gradcam = GradCAM(pred, int(class_cl[0][0]), f_ex, classification)
    gradcam = Image.fromarray(gradcam)
    gradcam = gradcam.resize(raw_image.shape[:2], resample=Image.BILINEAR)

    plt.imshow(raw_image)
    plt.axis("off")
    plt.imshow(np.array(gradcam), alpha=0.5, cmap="jet")
    plt.savefig("result.png")

    return cv2.imread("result.png")


def get_class_name(index: int) -> str:
    return CIFAR10_CLASSES[index]
