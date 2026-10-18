"""MNIST data loading and preprocessing utilities.

This module handles all aspects of MNIST dataset management including:
- Dataset downloading and loading
- Preprocessing and normalization
- Train/test splits
- Data augmentation
- PyTorch DataLoader creation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from .schemas import DataAugmentationConfig, DatasetConfig

logger = logging.getLogger(__name__)


class CIFAR10Dataset(Dataset):
    """Custom CIFAR10 dataset with optional preprocessing.

    Wraps torchvision CIFAR10 dataset with additional preprocessing capabilities
    and shape validation for different model types.
    """

    def __init__(
        self,
        root: Path,
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        flatten: bool = False,
        download: bool = True,
    ):
        """Initialize CIFAR10 dataset.

        Args:
            root: Root directory for dataset storage
            train: Whether to load training or test set
            transform: Optional torchvision transforms
            flatten: Whether to flatten images to 1D (for linear models)
            download: Whether to download dataset if not found
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.dataset = datasets.CIFAR10(
            root=str(self.root),
            train=train,
            download=download,
            transform=transform,
        )

        self.flatten = flatten

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, label) tensors
        """
        image, label = self.dataset[idx]

        # Flatten for linear models if requested
        if self.flatten:
            image = image.view(-1)

        return image, torch.tensor(label, dtype=torch.long)

    def get_sample_shape(self) -> Tuple[int, ...]:
        """Get the shape of a single sample (excluding batch dimension).

        Returns:
            Shape tuple of a single sample
        """
        sample_image, _ = self[0]
        return tuple(sample_image.shape)


def create_transforms(
    config: DatasetConfig,
    resize: int = 224,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create train and test transforms.

    Args:
        config: Data configuration containing dataset information
        augmentation_config: Optional data augmentation configuration (unused)
        Normalize: Whether to normalize images
        resize: Target size for BICUBIC resizing (default: 224)

    Returns:
        Tuple of (train_transforms, test_transforms)
    """
    # Base transforms (always applied)
    base_transforms = []
    # To tensor first
    base_transforms.append(transforms.ToTensor())

    if config.dataset == "cifar10":
        # BICUBIC resizing for CIFAR-10 (32x32 -> 224x224)
        base_transforms.append(
            transforms.Resize(
                size=(resize, resize),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        )
        # Normalize with standard ImageNet stats
        base_transforms.append(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )

    # Both train and test use the same transforms (no augmentation)
    train_transforms = transforms.Compose(base_transforms)
    test_transforms = transforms.Compose(base_transforms)

    logger.info(f"Created transforms with {len(base_transforms)} operations")

    return train_transforms, test_transforms


def create_datasets(
    config: DatasetConfig,
    augmentation_config: Optional[DataAugmentationConfig] = None,
    data_dir: Path = Path("data"),
    *,
    flatten: bool = False,
) -> Tuple[Dataset, Dataset]:
    """Create train and test datasets.

    Args:
        config: Data configuration
        augmentation_config: Optional augmentation configuration
        data_dir: Directory to store raw data

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create transforms
    train_transforms, test_transforms = create_transforms(config=config, resize=224)

    if config.dataset == "cifar10":
        DatasetClass = CIFAR10Dataset
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    # Load full training set
    train_dataset = DatasetClass(
        root=data_dir,
        train=True,
        transform=train_transforms,
        flatten=flatten,
        download=config.download,
    )

    # Load test set
    test_dataset = DatasetClass(
        root=data_dir,
        train=False,
        transform=test_transforms,
        flatten=flatten,
        download=config.download,
    )

    logger.info(
        f"Dataset sizes - Train: {len(train_dataset)}, Test: {len(test_dataset)}"
    )

    return train_dataset, test_dataset


def create_dataloaders(
    datasets: Tuple[Dataset, Dataset], config: DatasetConfig
) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders for all dataset splits.

    Args:
        datasets: Tuple of (train, test) datasets
        config: Data configuration

    Returns:
        Dictionary mapping split names to DataLoaders
    """
    train_dataset, test_dataset = datasets

    dataloaders = {}

    # Training dataloader (with shuffling)
    dataloaders["train"] = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    # Test dataloader (no shuffling)
    dataloaders["test"] = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    logger.info(f"Created {len(dataloaders)} dataloaders")
    return dataloaders


def get_dataset_stats(
    datasets: Tuple[Dataset, Dataset],
) -> Dict[str, Dict[str, float]]:
    """Compute basic statistics for each dataset split.

    Args:
        datasets: Tuple of (train, test) datasets

    Returns:
        Dictionary with statistics for each split
    """
    train_dataset, test_dataset = datasets
    stats = {}

    def compute_stats(dataset: Dataset, name: str) -> Dict[str, float]:
        """Compute statistics for a single dataset."""
        if len(dataset) == 0:
            return {}

        # Sample a few batches to compute statistics
        loader = DataLoader(dataset, batch_size=min(1000, len(dataset)), shuffle=False)
        images, labels = next(iter(loader))

        return {
            "num_samples": len(dataset),
            "image_mean": float(images.mean()),
            "image_std": float(images.std()),
            "image_min": float(images.min()),
            "image_max": float(images.max()),
            "num_classes": len(torch.unique(labels)),
            "label_distribution": dict(
                zip(torch.unique(labels).tolist(), torch.bincount(labels).tolist())
            ),
        }

    stats["train"] = compute_stats(train_dataset, "train")
    stats["test"] = compute_stats(test_dataset, "test")

    return stats


def extract_features_for_dataset(dataset, model, device, batch_size=128):
    """Extract features for all images in a dataset using the given model."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_features = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model(images)  # Should return [batch_size, feature_dim]
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
    features = torch.cat(all_features)
    labels = torch.cat(all_labels)
    return features, labels


def prepare_data(
    config: DatasetConfig,
    augmentation_config: Optional[DataAugmentationConfig] = None,
    data_dir: Path = Path("data/raw"),
    *,
    flatten: bool = False,
    model: Optional[callable] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, DataLoader], Dict[str, Dict[str, float]]]:
    """Complete data preparation pipeline.

    This is the main function that orchestrates the entire data loading process.

    Args:
        config: Data configuration
        augmentation_config: Optional augmentation configuration
        data_dir: Directory for raw data storage

    Returns:
        Tuple of (dataloaders_dict, dataset_stats)
    """
    logger.info("Starting data preparation pipeline")

    # Create datasets
    datasets = create_datasets(
        config=config,
        augmentation_config=augmentation_config,
        data_dir=data_dir,
        flatten=flatten,
    )

    if model is not None and device is not None:
        # Extract and cache features for train/test datasets
        import hashlib
        import os
        import pickle

        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        cache_key = hashlib.md5(
            (str(config.model_dump()) + str(type(model))).encode()
        ).hexdigest()
        cache_path = cache_dir / f"features_{config.dataset}_{cache_key}.pkl"

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                feature_datasets = pickle.load(f)
            logger.info(f"Loaded cached features from {cache_path}")
        else:
            splits = ["train", "test"]
            feature_datasets = {}
            for split, dataset in zip(splits, datasets):
                if dataset is not None:
                    features, labels = extract_features_for_dataset(
                        dataset, model, device, batch_size=config.batch_size
                    )
                    feature_datasets[split] = torch.utils.data.TensorDataset(
                        features, labels
                    )
                else:
                    feature_datasets[split] = None
            with open(cache_path, "wb") as f:
                pickle.dump(feature_datasets, f)
            logger.info(f"Saved extracted features to cache: {cache_path}")

        # Create dataloaders from feature datasets
        dataloaders = create_dataloaders(
            (
                feature_datasets["train"],
                feature_datasets["test"],
            ),
            config,
        )
        # Compute statistics on features (optional, here we keep original stats)
        stats = get_dataset_stats(datasets)
    else:
        # Fallback to original image dataloaders
        dataloaders = create_dataloaders(datasets, config)
        stats = get_dataset_stats(datasets)

    logger.info("Data preparation completed successfully")
    return dataloaders, stats


def validate_data_shape(
    dataloaders: Dict[str, DataLoader], expected_shape: Tuple[int, ...]
) -> bool:
    """Validate that data has expected shape for the model.

    Args:
        dataloaders: Dictionary of dataloaders
        expected_shape: Expected shape (excluding batch dimension)

    Returns:
        True if shapes match, False otherwise
    """
    try:
        # Get a sample from training data
        train_loader = dataloaders["train"]
        batch_images, batch_labels = next(iter(train_loader))

        # Check image shape (excluding batch dimension)
        actual_shape = tuple(batch_images.shape[1:])

        if actual_shape != expected_shape:
            logger.error(
                f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
            )
            return False

        logger.info(f"Data shape validation passed: {actual_shape}")
        return True

    except Exception as e:
        logger.error(f"Data shape validation failed: {e}")
        return False


# Utility f
__all__ = [
    "MNISTDataset",
    "create_transforms",
    "create_datasets",
    "create_dataloaders",
    "prepare_data",
    "validate_data_shape",
    "get_dataset_stats",
]
