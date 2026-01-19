"""
Base ImageFolder dataset for loading images from local directories.

Supports:
- Flat structure: root/*.jpg
- ImageFolder structure: root/class_name/*.jpg
- Split methods: subfolder (train/, val/) or percentage-based
"""

import os
import glob
import hashlib
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from src.data.transforms import create_image_transforms, normalize_image


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}


def _find_images(root: str, recursive: bool = True) -> List[str]:
    """Find all image files in a directory."""
    images = []
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            images.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
            images.extend(glob.glob(os.path.join(root, "**", f"*{ext.upper()}"), recursive=True))
    else:
        for ext in IMAGE_EXTENSIONS:
            images.extend(glob.glob(os.path.join(root, f"*{ext}")))
            images.extend(glob.glob(os.path.join(root, f"*{ext.upper()}")))
    return sorted(images)


def _detect_structure(root: str) -> Literal["flat", "imagefolder"]:
    """Detect whether the directory has flat or ImageFolder structure."""
    # Check if there are subdirectories with images
    for item in os.listdir(root):
        item_path = os.path.join(root, item)
        if os.path.isdir(item_path):
            # Check if it contains images
            subdir_images = _find_images(item_path, recursive=False)
            if subdir_images:
                return "imagefolder"
    return "flat"


def _get_imagefolder_class_mapping(root: str) -> Tuple[Dict[str, int], List[str]]:
    """Get class name to index mapping for ImageFolder structure."""
    class_dirs = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
    return class_to_idx, class_dirs


def _split_files_by_percentage(
    files: List[str],
    train_split: float,
    val_split: float,
    seed: int,
    split: str,
) -> List[str]:
    """Split files by percentage using deterministic shuffling."""
    # Create deterministic order based on hash of filenames
    # This ensures same split regardless of file system ordering
    rng = np.random.RandomState(seed)
    indices = np.arange(len(files))
    rng.shuffle(indices)

    n_total = len(files)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    if split == "train":
        selected_indices = indices[:n_train]
    elif split in ("validation", "val"):
        selected_indices = indices[n_train:n_train + n_val]
    elif split == "test":
        selected_indices = indices[n_train + n_val:]
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'validation', or 'test'")

    return [files[i] for i in sorted(selected_indices)]


class ImageFolderDataset(Dataset):
    """
    Dataset for loading images from local directories.

    Supports two directory structures:
    1. Flat: All images in root directory
    2. ImageFolder: Images organized in class subdirectories

    And two split methods:
    1. Subfolder: Separate directories for train/val/test
    2. Percentage: Split files by percentage

    Args:
        config: Configuration dictionary with the following keys:
            - data_folder (str): Root directory (required)
            - split (str): train/validation/test (default: "train")
            - split_method (str): "subfolder" or "percentage" (default: "subfolder")
            - train_split (float): Fraction for training (default: 0.8)
            - val_split (float): Fraction for validation (default: 0.1)
            - size (int): Target image size (default: 256)
            - random_crop (bool): Use random crop instead of center crop
            - horizontal_flip (bool): Apply random horizontal flip
            - seed (int): Random seed for percentage splits (default: 42)
        transform_backend: Which transform library to use ("albumentations" or "torchvision")
    """

    def __init__(
        self,
        config: Optional[Union[Dict, OmegaConf]] = None,
        transform_backend: Literal["albumentations", "torchvision"] = "albumentations",
    ):
        self.config = config or {}
        if not isinstance(self.config, dict):
            self.config = OmegaConf.to_container(self.config)

        # Required parameters
        self.data_folder = self.config.get("data_folder")
        if not self.data_folder:
            raise ValueError("data_folder is required")
        if not os.path.exists(self.data_folder):
            raise ValueError(f"data_folder does not exist: {self.data_folder}")

        # Optional parameters
        self.split = self.config.get("split", "train")
        self.split_method = self.config.get("split_method", "subfolder")
        self.train_split = self.config.get("train_split", 0.8)
        self.val_split = self.config.get("val_split", 0.1)
        self.size = self.config.get("size", 256)
        self.random_crop = self.config.get("random_crop", self.split == "train")
        self.horizontal_flip = self.config.get("horizontal_flip", self.random_crop)
        self.seed = self.config.get("seed", 42)

        # Load files based on split method
        self.image_paths: List[str] = []
        self.class_labels: Optional[List[int]] = None
        self.class_names: Optional[List[str]] = None
        self.class_to_idx: Optional[Dict[str, int]] = None

        if self.split_method == "subfolder":
            self._load_subfolder_split()
        elif self.split_method == "percentage":
            self._load_percentage_split()
        else:
            raise ValueError(f"Unknown split_method: {self.split_method}")

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.data_folder} for split '{self.split}'")

        # Create transform
        self.transform = create_image_transforms(
            size=self.size,
            random_crop=self.random_crop,
            horizontal_flip=self.horizontal_flip,
            backend=transform_backend,
        )

    def _load_subfolder_split(self):
        """Load images from split-specific subdirectory."""
        # Normalize split name for directory lookup
        split_dir_names = {
            "train": ["train", "training"],
            "validation": ["val", "validation", "valid"],
            "val": ["val", "validation", "valid"],
            "test": ["test", "testing"],
        }

        # Find the split directory
        split_root = None
        for name in split_dir_names.get(self.split, [self.split]):
            potential_path = os.path.join(self.data_folder, name)
            if os.path.exists(potential_path):
                split_root = potential_path
                break

        if split_root is None:
            # No split subdirectory found, use root directly
            split_root = self.data_folder

        # Detect structure and load
        structure = _detect_structure(split_root)

        if structure == "imagefolder":
            self._load_imagefolder_structure(split_root)
        else:
            self._load_flat_structure(split_root)

    def _load_percentage_split(self):
        """Load images and split by percentage."""
        # Detect structure
        structure = _detect_structure(self.data_folder)

        if structure == "imagefolder":
            # For ImageFolder, split within each class to maintain balance
            self.class_to_idx, self.class_names = _get_imagefolder_class_mapping(self.data_folder)
            all_paths = []
            all_labels = []

            for cls_name, cls_idx in self.class_to_idx.items():
                cls_dir = os.path.join(self.data_folder, cls_name)
                cls_images = _find_images(cls_dir, recursive=True)
                cls_images = _split_files_by_percentage(
                    cls_images, self.train_split, self.val_split, self.seed, self.split
                )
                all_paths.extend(cls_images)
                all_labels.extend([cls_idx] * len(cls_images))

            self.image_paths = all_paths
            self.class_labels = all_labels
        else:
            # Flat structure
            all_images = _find_images(self.data_folder, recursive=True)
            self.image_paths = _split_files_by_percentage(
                all_images, self.train_split, self.val_split, self.seed, self.split
            )

    def _load_imagefolder_structure(self, root: str):
        """Load images from ImageFolder structure."""
        self.class_to_idx, self.class_names = _get_imagefolder_class_mapping(root)

        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = os.path.join(root, cls_name)
            cls_images = _find_images(cls_dir, recursive=True)
            if self.class_labels is None:
                self.class_labels = []
            self.image_paths.extend(cls_images)
            self.class_labels.extend([cls_idx] * len(cls_images))

    def _load_flat_structure(self, root: str):
        """Load images from flat structure."""
        self.image_paths = _find_images(root, recursive=True)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.image_paths[idx]

        # Load image
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply transforms
        image = self.transform(image)
        image = normalize_image(image)

        result = {"image": image, "file_path_": image_path}

        # Add label if available
        if self.class_labels is not None:
            result["class_label"] = self.class_labels[idx]
            if self.class_names is not None:
                result["class_name"] = self.class_names[self.class_labels[idx]]

        return result
