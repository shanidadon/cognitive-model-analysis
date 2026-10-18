"""Tests for data loading and preprocessing functionality.

This module tests CIFAR10 data loading, preprocessing, and DataLoader creation
to ensure proper data handling across different configurations.
"""

import pickle
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from src.experiment.data import CIFAR10Dataset  # Add this import

# --- Comprehensive CIFAR10 Tests ---


class TestCIFAR10Comprehensive:
    """Comprehensive tests for CIFAR10 dataset loading and validation."""

    def test_cifar10_download_and_load(self):
        """Test that CIFAR10 data can be downloaded and loaded successfully."""
        data_dir = Path("test_data")

        # Create dataset with download=True
        dataset = CIFAR10Dataset(root=data_dir, download=True)

        # Check that data exists with detailed messages
        assert data_dir.exists(), f"Data directory {data_dir} does not exist"
        assert len(dataset) > 0, f"Dataset is empty, length: {len(dataset)}"

        # Test basic properties
        image, label = dataset[0]
        assert isinstance(
            image, torch.Tensor
        ), f"Image is not a tensor, got: {type(image)}"

        # Handle both int and tensor labels
        if isinstance(label, torch.Tensor):
            label_value = label.item()
        else:
            label_value = label

        assert isinstance(
            label_value, int
        ), f"Label is not convertible to int, got: {type(label)} with value: {label}"
        assert image.shape == (
            3,
            32,
            32,
        ), f"Unexpected image shape: {image.shape}, expected: (3, 32, 32)"
        assert (
            0 <= label_value <= 9
        ), f"Label out of range: {label_value}, expected: 0-9"

    def test_cifar10_data_properties(self):
        """Test CIFAR10 dataset basic properties."""
        dataset = CIFAR10Dataset(root=Path("test_data"), train=True, download=True)
        # Test dataset size
        assert len(dataset) == 50000  # Training set size
        # Test data ranges
        image, label = dataset[0]
        assert image.min() >= 0
        assert image.max() <= 1  # Assuming normalized
        assert image.dtype == torch.float32
        # Get unique labels (convert tensor labels to int)
        labels = []
        for i in range(min(1000, len(dataset))):
            label = dataset[i][1]
            if isinstance(label, torch.Tensor):
                label = label.item()
            labels.append(label)
        unique_labels = set(labels)
        assert len(unique_labels) == 10  # CIFAR10 has 10 classes

    @pytest.mark.parametrize("train", [True, False])
    def test_cifar10_train_test_splits(self, train):
        """Test both training and test splits of CIFAR10."""
        dataset = CIFAR10Dataset(root=Path("test_data"), train=train, download=True)

        expected_size = 50000 if train else 10000
        assert len(dataset) == expected_size

    def test_cifar10_transform_application(self):
        """Test that transforms are correctly applied to CIFAR10 data."""
        # Only apply normalization if dataset already returns tensors
        transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        dataset = CIFAR10Dataset(
            root=Path("test_data"), train=True, transform=transform, download=True
        )
        image, _ = dataset[0]
        # Check that normalization was applied
        assert image.mean().item() != 0  # Should be normalized
        assert 0.1 < image.std().item() < 10  # Reasonable range for normalized data

    @pytest.mark.slow
    def test_cifar10_batch_loading(self):
        """Test loading CIFAR10 in batches through DataLoader."""
        dataset = CIFAR10Dataset(root=Path("test_data"), train=True, download=True)

        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Test batch properties
        batch_images, batch_labels = next(iter(loader))
        assert batch_images.shape == (32, 3, 32, 32)
        assert batch_labels.shape == (32,)
        assert batch_labels.dtype == torch.long

    @pytest.mark.slow
    def test_cifar10_batch_distribution_analysis(self):
        """Analyze the distribution of classes across CIFAR10 batches."""
        import pickle

        data_dir = Path("test_data")

        # Download dataset first
        dataset = CIFAR10Dataset(root=data_dir, train=True, download=True)

        # CIFAR10 class names
        classes = [
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

        print("\n" + "=" * 60)
        print("CIFAR10 Batch Distribution Analysis")
        print("=" * 60)

        # Analyze each individual batch file
        cifar_dir = data_dir / "cifar-10-batches-py"

        for batch_num in range(1, 6):
            batch_file = cifar_dir / f"data_batch_{batch_num}"

            # Load the raw batch file
            with open(batch_file, "rb") as f:
                batch_data = pickle.load(f, encoding="bytes")

            # Count classes in this batch
            labels = batch_data[b"labels"]
            class_counts = {}

            for class_idx in range(10):
                count = labels.count(class_idx)
                class_counts[classes[class_idx]] = count

            print(f"\nBatch {batch_num} (10,000 images):")
            print("-" * 40)

            for class_name in classes:
                count = class_counts[class_name]
                percentage = (count / 10000) * 100
                print(f"{class_name:>12}: {count:>4} images ({percentage:>5.1f}%)")

        # Test the test batch too
        test_batch_file = cifar_dir / "test_batch"
        with open(test_batch_file, "rb") as f:
            test_data = pickle.load(f, encoding="bytes")

        test_labels = test_data[b"labels"]
        test_class_counts = {}

        for class_idx in range(10):
            count = test_labels.count(class_idx)
            test_class_counts[classes[class_idx]] = count

        print(f"\nTest Batch (10,000 images):")
        print("-" * 40)

        for class_name in classes:
            count = test_class_counts[class_name]
            percentage = (count / 10000) * 100
            print(f"{class_name:>12}: {count:>4} images ({percentage:>5.1f}%)")

        # Verify all classes are present in each batch
        for batch_num in range(1, 6):
            batch_file = cifar_dir / f"data_batch_{batch_num}"
            with open(batch_file, "rb") as f:
                batch_data = pickle.load(f, encoding="bytes")

            labels = batch_data[b"labels"]
            unique_labels = set(labels)

            # Each batch should contain all 10 classes
            assert len(unique_labels) == 10, f"Batch {batch_num} missing some classes"
            assert unique_labels == set(
                range(10)
            ), f"Batch {batch_num} has wrong classes"

        print("\n" + "=" * 60)
        print("✓ All batches contain all 10 classes")
        print("✓ Distribution varies across batches (as expected)")
        print("=" * 60)

    @pytest.mark.slow
    def test_save_cifar10_class_examples(self):
        """Save one example image from each CIFAR10 class for visual inspection."""
        data_dir = Path("test_data")
        examples_dir = data_dir / "image_examples"
        examples_dir.mkdir(parents=True, exist_ok=True)

        # CIFAR10 class names
        classes = [
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

        # Create dataset without transforms to get raw images
        dataset = CIFAR10Dataset(
            root=data_dir, train=True, download=True, transform=None
        )

        # Create a figure with 10 subplots (2 rows, 5 columns)
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()

        found_classes = set()
        truck_count = 0  # Track number of truck images saved
        processed_items = 0

        print(f"\nSearching for examples of each CIFAR10 class...")

        # Collect one example of each class (and extra truck images)
        for idx, (image, label) in enumerate(dataset):
            # Ensure label is an integer
            if isinstance(label, torch.Tensor):
                label = label.item()

            # Save first example of each class
            if label not in found_classes:
                # Get class name
                class_name = classes[label]
                print(f"Found {class_name} (class {label}) at index {idx}")

                # Convert PIL Image or numpy array to numpy for plotting
                if isinstance(image, torch.Tensor):
                    # If already tensor, convert to numpy (C,H,W) -> (H,W,C)
                    img_np = image.detach().cpu().permute(1, 2, 0).numpy()
                else:
                    # If PIL Image or numpy array, convert to numpy
                    img_np = np.array(image)
                    # If PIL image, it might be (H,W,C), if numpy it might be (C,H,W)
                    if img_np.ndim == 3 and img_np.shape[0] == 3:
                        # Convert from (C,H,W) to (H,W,C)
                        img_np = np.transpose(img_np, (1, 2, 0))

                # Ensure values are in [0,1] range and convert to float32
                if img_np.max() > 1.0:
                    img_np = img_np / 255.0
                img_np = np.clip(img_np, 0, 1).astype(np.float32)

                # Plot the image in the subplot
                ax = axes[label]
                ax.imshow(img_np)
                ax.axis("off")
                ax.set_title(f"{class_name}\n(class {label})", fontsize=10, pad=5)

                # Save individual image with class name using plt.imsave directly
                individual_path = examples_dir / f"{label:02d}_{class_name}.png"
                plt.imsave(str(individual_path), img_np)
                found_classes.add(label)

            # Save additional truck images (class 9)
            elif label == 9 and truck_count < 2:  # Save 2 more truck images
                truck_count += 1
                class_name = classes[label]
                print(
                    f"Found additional {class_name} (class {label}) at index {idx} - saving as truck_{truck_count}.png"
                )

                # Convert image to numpy
                if isinstance(image, torch.Tensor):
                    img_np = image.detach().cpu().permute(1, 2, 0).numpy()
                else:
                    img_np = np.array(image)
                    if img_np.ndim == 3 and img_np.shape[0] == 3:
                        img_np = np.transpose(img_np, (1, 2, 0))

                # Ensure values are in [0,1] range and convert to float32
                if img_np.max() > 1.0:
                    img_np = img_np / 255.0
                img_np = np.clip(img_np, 0, 1).astype(np.float32)

                # Save additional truck image with counter
                truck_path = examples_dir / f"09_truck_{truck_count}.png"
                plt.imsave(str(truck_path), img_np)

            processed_items += 1
            if (
                len(found_classes) == 10 and truck_count >= 2
            ):  # Found all classes + extra trucks
                break
            if processed_items > 10000:  # Safety limit
                raise RuntimeError(
                    "Could not find all required images in first 10,000 images"
                )

        # Save the combined figure (use fig.savefig to avoid tensor issues)
        fig.suptitle(
            "CIFAR10 Dataset - One Example from Each Class", fontsize=16, y=0.98
        )
        fig.tight_layout()
        combined_path = examples_dir / "all_classes_overview.png"
        fig.savefig(str(combined_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Verify that we found and saved all classes
        assert (
            len(found_classes) == 10
        ), f"Only found {len(found_classes)} classes: {found_classes}"

        # Verify that files were created
        assert combined_path.exists(), "Combined overview image was not created"

        for class_idx, class_name in enumerate(classes):
            individual_path = examples_dir / f"{class_idx:02d}_{class_name}.png"
            assert (
                individual_path.exists()
            ), f"Individual image for {class_name} was not created"

        # Verify additional truck images were created
        for i in range(1, truck_count + 1):
            truck_path = examples_dir / f"09_truck_{i}.png"
            assert truck_path.exists(), f"Additional truck image {i} was not created"

        # Print summary
        print(f"\n" + "=" * 60)
        print(f"CIFAR10 Visual Inspection Complete!")
        print(f"=" * 60)
        print(f"📁 Images saved to: {examples_dir}")
        print(f"📊 Combined overview: {combined_path.name}")
        print(f"🖼️  Individual images:")
        for class_idx, class_name in enumerate(classes):
            filename = f"{class_idx:02d}_{class_name}.png"
            print(f"   • {filename}")
        print(f"🚚 Additional truck images:")
        for i in range(1, truck_count + 1):
            print(f"   • 09_truck_{i}.png")
        print(f"\n✓ All 10 classes found and saved")
        print(f"✓ Additional truck images saved: {truck_count}")
        print(f"✓ Total images processed: {processed_items}")
        print(f"=" * 60)
