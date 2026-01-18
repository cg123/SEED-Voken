"""
IBQ wrappers for HuggingFace datasets.

Uses albumentations for image transforms to match the existing IBQ codebase.
"""

from typing import Dict, Optional, Union
from omegaconf import OmegaConf

from src.data.huggingface import (
    HuggingFaceImageDataset as BaseHuggingFaceImageDataset,
    HuggingFaceStreamingDataset as BaseHuggingFaceStreamingDataset,
)


class HuggingFaceImageDataset(BaseHuggingFaceImageDataset):
    """
    HuggingFace image dataset with albumentations transforms.

    See src.data.huggingface.HuggingFaceImageDataset for full documentation.
    """

    def __init__(self, config: Optional[Union[Dict, OmegaConf]] = None):
        super().__init__(config=config, transform_backend="albumentations")


class HuggingFaceStreamingDataset(BaseHuggingFaceStreamingDataset):
    """
    HuggingFace streaming dataset with albumentations transforms.

    See src.data.huggingface.HuggingFaceStreamingDataset for full documentation.
    """

    def __init__(self, config: Optional[Union[Dict, OmegaConf]] = None):
        super().__init__(config=config, transform_backend="albumentations")


class HuggingFaceDatasetTrain(HuggingFaceImageDataset):
    """
    HuggingFace training dataset.

    Convenience class that sets default split to "train" with random crop and flip.
    """

    def __init__(self, config: Optional[Union[Dict, OmegaConf]] = None):
        config = dict(config) if config else {}
        config.setdefault("split", "train")
        config.setdefault("random_crop", True)
        config.setdefault("horizontal_flip", True)
        super().__init__(config=config)


class HuggingFaceDatasetValidation(HuggingFaceImageDataset):
    """
    HuggingFace validation dataset.

    Convenience class that sets default split to "validation" with center crop.
    """

    def __init__(self, config: Optional[Union[Dict, OmegaConf]] = None):
        config = dict(config) if config else {}
        config.setdefault("split", "validation")
        config.setdefault("random_crop", False)
        config.setdefault("horizontal_flip", False)
        super().__init__(config=config)


class HuggingFaceDatasetTest(HuggingFaceImageDataset):
    """
    HuggingFace test dataset.

    Convenience class that sets default split to "test" with center crop.
    """

    def __init__(self, config: Optional[Union[Dict, OmegaConf]] = None):
        config = dict(config) if config else {}
        config.setdefault("split", "test")
        config.setdefault("random_crop", False)
        config.setdefault("horizontal_flip", False)
        super().__init__(config=config)


# Streaming variants
class HuggingFaceStreamingDatasetTrain(HuggingFaceStreamingDataset):
    """
    HuggingFace streaming training dataset.

    Convenience class for streaming large datasets with training augmentations.
    """

    def __init__(self, config: Optional[Union[Dict, OmegaConf]] = None):
        config = dict(config) if config else {}
        config.setdefault("split", "train")
        config.setdefault("random_crop", True)
        config.setdefault("horizontal_flip", True)
        super().__init__(config=config)


class HuggingFaceStreamingDatasetValidation(HuggingFaceStreamingDataset):
    """
    HuggingFace streaming validation dataset.

    Convenience class for streaming large datasets with validation preprocessing.
    """

    def __init__(self, config: Optional[Union[Dict, OmegaConf]] = None):
        config = dict(config) if config else {}
        config.setdefault("split", "validation")
        config.setdefault("random_crop", False)
        config.setdefault("horizontal_flip", False)
        super().__init__(config=config)
