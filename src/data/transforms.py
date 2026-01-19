"""
Transform factory for creating image preprocessing pipelines.

Supports both albumentations (for IBQ) and torchvision (for Open_MAGVIT2) backends.
"""

from typing import Callable, Literal, Optional
import numpy as np
from PIL import Image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [-1, 1] range.

    Args:
        image: Input image as numpy array with values in [0, 255]

    Returns:
        Normalized image as float32 array with values in [-1, 1]
    """
    return (image / 127.5 - 1.0).astype(np.float32)


def create_image_transforms(
    size: int,
    random_crop: bool = False,
    horizontal_flip: bool = False,
    backend: Literal["albumentations", "torchvision"] = "albumentations",
) -> Callable:
    """
    Create an image transform pipeline.

    Args:
        size: Target image size (will be square)
        random_crop: If True, use random crop; otherwise use center crop
        horizontal_flip: If True, apply random horizontal flip (only when random_crop=True)
        backend: Which transform library to use ("albumentations" or "torchvision")

    Returns:
        A callable that takes a PIL Image or numpy array and returns a transformed numpy array
    """
    if backend == "albumentations":
        return _create_albumentations_transforms(size, random_crop, horizontal_flip)
    elif backend == "torchvision":
        return _create_torchvision_transforms(size, random_crop, horizontal_flip)
    else:
        raise ValueError(f"Unknown backend: {backend}. Must be 'albumentations' or 'torchvision'")


def _create_albumentations_transforms(
    size: int,
    random_crop: bool,
    horizontal_flip: bool,
) -> Callable:
    """Create albumentations-based transform pipeline."""
    import albumentations as A

    transforms = []

    # Resize so smallest side is target size
    transforms.append(A.SmallestMaxSize(max_size=size))

    # Crop
    if random_crop:
        transforms.append(A.RandomCrop(height=size, width=size))
        if horizontal_flip:
            transforms.append(A.HorizontalFlip(p=0.5))
    else:
        transforms.append(A.CenterCrop(height=size, width=size))

    pipeline = A.Compose(transforms)

    def transform(image):
        """Transform a PIL Image or numpy array."""
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = np.array(image).astype(np.uint8)
        return pipeline(image=image)["image"]

    return transform


def _create_torchvision_transforms(
    size: int,
    random_crop: bool,
    horizontal_flip: bool,
) -> Callable:
    """Create torchvision-based transform pipeline."""
    import torchvision.transforms as T

    transforms = []

    # Resize so smallest side is target size
    transforms.append(T.Resize(size))

    # Crop
    if random_crop:
        transforms.append(T.RandomCrop((size, size)))
        if horizontal_flip:
            transforms.append(T.RandomHorizontalFlip(p=0.5))
    else:
        transforms.append(T.CenterCrop((size, size)))

    pipeline = T.Compose(transforms)

    def transform(image):
        """Transform a PIL Image or numpy array."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Apply transforms and convert to numpy
        image = pipeline(image)
        return np.array(image)

    return transform
