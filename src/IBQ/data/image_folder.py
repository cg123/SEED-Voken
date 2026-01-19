"""
IBQ wrappers for ImageFolder datasets.

Uses albumentations for image transforms to match the existing IBQ codebase.
"""

from typing import Dict, Optional, Union
from omegaconf import OmegaConf

from src.data.image_folder import ImageFolderDataset as BaseImageFolderDataset


class ImageFolderDataset(BaseImageFolderDataset):
    """
    ImageFolder dataset with albumentations transforms.

    See src.data.image_folder.ImageFolderDataset for full documentation.
    """

    def __init__(self, config: Optional[Union[Dict, OmegaConf]] = None):
        super().__init__(config=config, transform_backend="albumentations")


class ImageFolderTrain(ImageFolderDataset):
    """
    ImageFolder training dataset.

    Convenience class that sets default split to "train" with random crop and flip.
    """

    def __init__(self, config: Optional[Union[Dict, OmegaConf]] = None):
        config = dict(config) if config else {}
        config.setdefault("split", "train")
        config.setdefault("random_crop", True)
        config.setdefault("horizontal_flip", True)
        super().__init__(config=config)


class ImageFolderValidation(ImageFolderDataset):
    """
    ImageFolder validation dataset.

    Convenience class that sets default split to "validation" with center crop.
    """

    def __init__(self, config: Optional[Union[Dict, OmegaConf]] = None):
        config = dict(config) if config else {}
        config.setdefault("split", "validation")
        config.setdefault("random_crop", False)
        config.setdefault("horizontal_flip", False)
        super().__init__(config=config)


class ImageFolderTest(ImageFolderDataset):
    """
    ImageFolder test dataset.

    Convenience class that sets default split to "test" with center crop.
    """

    def __init__(self, config: Optional[Union[Dict, OmegaConf]] = None):
        config = dict(config) if config else {}
        config.setdefault("split", "test")
        config.setdefault("random_crop", False)
        config.setdefault("horizontal_flip", False)
        super().__init__(config=config)
