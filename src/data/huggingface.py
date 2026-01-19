"""
Base HuggingFace dataset classes for loading image datasets from HuggingFace Hub.

Provides both map-style and streaming datasets for different use cases:
- HuggingFaceImageDataset: For small/medium datasets that fit in memory
- HuggingFaceStreamingDataset: For large datasets that need streaming
"""

from typing import Any, Callable, Dict, Literal, Optional, Union
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from omegaconf import OmegaConf

from src.data.transforms import create_image_transforms, normalize_image


# Common image column names in HuggingFace datasets
IMAGE_COLUMN_NAMES = ["image", "img", "pixel_values", "input_image"]
# Common label column names in HuggingFace datasets
LABEL_COLUMN_NAMES = ["label", "labels", "class", "class_label", "target"]


def _detect_column(columns: list, candidates: list, column_type: str) -> Optional[str]:
    """Auto-detect a column name from a list of candidates."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _get_pil_image(item: Any) -> Image.Image:
    """Extract PIL Image from various HuggingFace dataset formats."""
    if isinstance(item, Image.Image):
        return item
    elif isinstance(item, dict):
        # Some datasets wrap images in dicts with 'bytes' or 'path' keys
        if "bytes" in item:
            import io
            return Image.open(io.BytesIO(item["bytes"]))
        elif "path" in item:
            return Image.open(item["path"])
    elif isinstance(item, np.ndarray):
        return Image.fromarray(item)
    elif isinstance(item, str):
        # Path to image
        return Image.open(item)
    raise ValueError(f"Cannot convert {type(item)} to PIL Image")


class HuggingFaceImageDataset(Dataset):
    """
    Map-style dataset for loading images from HuggingFace Hub.

    Suitable for small to medium sized datasets that can be downloaded
    and stored locally.

    Args:
        config: Configuration dictionary with the following keys:
            - dataset_name (str): HuggingFace dataset name or "parquet" for local/S3 parquet files
            - dataset_config (str, optional): Dataset configuration name
            - data_files (dict/str, optional): Parquet file paths (e.g., {"train": "s3://bucket/*.parquet"})
            - split (str): Dataset split (default: "train")
            - image_column (str, optional): Column name for images (auto-detected if not specified)
            - label_column (str, optional): Column name for labels (auto-detected if not specified)
            - size (int): Target image size (default: 256)
            - random_crop (bool): Use random crop instead of center crop (default: True for train)
            - horizontal_flip (bool): Apply random horizontal flip (default: same as random_crop)
            - token (str, optional): HuggingFace token for gated datasets
            - cache_dir (str, optional): Custom cache directory for downloads
            - trust_remote_code (bool): Whether to trust remote code (default: False)
        transform_backend: Which transform library to use ("albumentations" or "torchvision")
    """

    def __init__(
        self,
        config: Optional[Union[Dict, OmegaConf]] = None,
        transform_backend: Literal["albumentations", "torchvision"] = "albumentations",
    ):
        from datasets import load_dataset

        self.config = config or {}
        if not isinstance(self.config, dict):
            self.config = OmegaConf.to_container(self.config)

        # Required parameters
        self.dataset_name = self.config.get("dataset_name")
        if not self.dataset_name:
            raise ValueError("dataset_name is required")

        # Optional parameters
        self.dataset_config = self.config.get("dataset_config")
        self.data_files = self.config.get("data_files")
        self.split = self.config.get("split", "train")
        self.image_column = self.config.get("image_column")
        self.label_column = self.config.get("label_column")
        self.size = self.config.get("size", 256)
        self.random_crop = self.config.get("random_crop", self.split == "train")
        self.horizontal_flip = self.config.get("horizontal_flip", self.random_crop)
        self.token = self.config.get("token")
        self.cache_dir = self.config.get("cache_dir")
        self.trust_remote_code = self.config.get("trust_remote_code", False)

        # Load dataset
        load_kwargs = {
            "path": self.dataset_name,
            "split": self.split,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.dataset_config:
            load_kwargs["name"] = self.dataset_config
        if self.data_files:
            load_kwargs["data_files"] = self.data_files
        if self.token:
            load_kwargs["token"] = self.token
        if self.cache_dir:
            load_kwargs["cache_dir"] = self.cache_dir

        self.dataset = load_dataset(**load_kwargs)

        # Auto-detect columns if not specified
        columns = self.dataset.column_names
        if not self.image_column:
            self.image_column = _detect_column(columns, IMAGE_COLUMN_NAMES, "image")
            if not self.image_column:
                raise ValueError(
                    f"Could not auto-detect image column. "
                    f"Available columns: {columns}. "
                    f"Please specify image_column in config."
                )
        if not self.label_column:
            self.label_column = _detect_column(columns, LABEL_COLUMN_NAMES, "label")
            # Label column is optional

        # Create transform
        self.transform = create_image_transforms(
            size=self.size,
            random_crop=self.random_crop,
            horizontal_flip=self.horizontal_flip,
            backend=transform_backend,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]

        # Get image
        raw_image = item[self.image_column]
        pil_image = _get_pil_image(raw_image)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Apply transforms
        image = self.transform(pil_image)
        image = normalize_image(image)

        result = {"image": image}

        # Add label if available
        if self.label_column and self.label_column in item:
            result["class_label"] = item[self.label_column]

        return result


class HuggingFaceStreamingDataset(IterableDataset):
    """
    Streaming dataset for loading large image datasets from HuggingFace Hub.

    Suitable for very large datasets (like ImageNet-1k) where downloading
    the entire dataset is impractical.

    Args:
        config: Configuration dictionary (same as HuggingFaceImageDataset)
        transform_backend: Which transform library to use ("albumentations" or "torchvision")
    """

    def __init__(
        self,
        config: Optional[Union[Dict, OmegaConf]] = None,
        transform_backend: Literal["albumentations", "torchvision"] = "albumentations",
    ):
        from datasets import load_dataset

        self.config = config or {}
        if not isinstance(self.config, dict):
            self.config = OmegaConf.to_container(self.config)

        # Required parameters
        self.dataset_name = self.config.get("dataset_name")
        if not self.dataset_name:
            raise ValueError("dataset_name is required")

        # Optional parameters
        self.dataset_config = self.config.get("dataset_config")
        self.data_files = self.config.get("data_files")
        self.split = self.config.get("split", "train")
        self.image_column = self.config.get("image_column")
        self.label_column = self.config.get("label_column")
        self.size = self.config.get("size", 256)
        self.random_crop = self.config.get("random_crop", self.split == "train")
        self.horizontal_flip = self.config.get("horizontal_flip", self.random_crop)
        self.token = self.config.get("token")
        self.cache_dir = self.config.get("cache_dir")
        self.trust_remote_code = self.config.get("trust_remote_code", False)

        # Load dataset in streaming mode
        load_kwargs = {
            "path": self.dataset_name,
            "split": self.split,
            "streaming": True,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.dataset_config:
            load_kwargs["name"] = self.dataset_config
        if self.data_files:
            load_kwargs["data_files"] = self.data_files
        if self.token:
            load_kwargs["token"] = self.token
        if self.cache_dir:
            load_kwargs["cache_dir"] = self.cache_dir

        self.dataset = load_dataset(**load_kwargs)

        # Get column names from features
        columns = list(self.dataset.features.keys()) if hasattr(self.dataset, 'features') else []

        # Auto-detect columns if not specified
        if not self.image_column:
            self.image_column = _detect_column(columns, IMAGE_COLUMN_NAMES, "image")
            if not self.image_column:
                # For streaming, we may not have features info upfront
                # Default to "image" and let it fail later if wrong
                self.image_column = "image"
        if not self.label_column:
            self.label_column = _detect_column(columns, LABEL_COLUMN_NAMES, "label")

        # Create transform
        self.transform = create_image_transforms(
            size=self.size,
            random_crop=self.random_crop,
            horizontal_flip=self.horizontal_flip,
            backend=transform_backend,
        )

    def __iter__(self):
        for item in self.dataset:
            # Get image
            raw_image = item[self.image_column]
            pil_image = _get_pil_image(raw_image)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Apply transforms
            image = self.transform(pil_image)
            image = normalize_image(image)

            result = {"image": image}

            # Add label if available
            if self.label_column and self.label_column in item:
                result["class_label"] = item[self.label_column]

            yield result
