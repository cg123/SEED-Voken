"""
Shared data loading utilities for SEED-Voken.

This module provides base classes for HuggingFace datasets and local image folders
that can be used with both IBQ (albumentations) and Open_MAGVIT2 (torchvision) backends.
"""

from src.data.transforms import create_image_transforms, normalize_image
from src.data.huggingface import HuggingFaceImageDataset, HuggingFaceStreamingDataset
from src.data.image_folder import ImageFolderDataset

__all__ = [
    "create_image_transforms",
    "normalize_image",
    "HuggingFaceImageDataset",
    "HuggingFaceStreamingDataset",
    "ImageFolderDataset",
]
