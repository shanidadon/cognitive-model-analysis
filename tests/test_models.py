"""Tests for machine learning models.

This module tests all model implementations including architecture,
forward passes, parameter counting, and factory functions.
"""

import os
import pickle
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.experiment.models import (
    BaseModel,
)


class TestImageRepresentation:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model_name",
        [
            "DINOv2_ViT_Large",
            # Add more model names here as needed
        ],
    )
    def test_single_image_representation_shape(self, model_name):
        image_dir = "test_data/image_examples"
        image_files = [f for f in os.listdir(image_dir) if not f.startswith(".")]
        image_path = os.path.join(image_dir, image_files[0])
        image = Image.open(image_path).convert("RGB")
        from src.experiment.models import model_factory

        model = model_factory(model_name, device="cuda")
        # Force all model params and buffers to CUDA
        for param in model.parameters():
            param.data = param.data.cuda()
        for buf in model.buffers():
            buf.data = buf.data.cuda()
        model.eval()
        tensor_image = model.transform(image)
        if tensor_image.dim() == 3:
            tensor_image = tensor_image.unsqueeze(0)
        device = next(model.parameters()).device
        tensor_image = tensor_image.to(device)
        with torch.no_grad():
            rep = model.forward(tensor_image)
        # Typical DINOv2 CLS shape: [1, 1024] or token shape [1, 257, 1024]
        assert rep.shape[0] == 1
        assert rep.shape[-1] == 1024

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model_name",
        [
            "DINOv2_ViT_Large",
            # Add more model names here as needed
        ],
    )
    def test_single_image_bicubic_vs_default_transform_difference(self, model_name):
        image_dir = "test_data/image_examples"
        image_files = [f for f in os.listdir(image_dir) if not f.startswith(".")]
        image_path = os.path.join(image_dir, image_files[0])
        image = Image.open(image_path).convert("RGB")
        from src.experiment.models import model_factory

        model = model_factory(model_name, device="cuda")
        for param in model.parameters():
            param.data = param.data.cuda()
        for buf in model.buffers():
            buf.data = buf.data.cuda()
        model.eval()
        bicubic_transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True
                ),
                transforms.ToTensor(),
            ]
        )
        tensor_bicubic = bicubic_transform(image).unsqueeze(0)
        device = next(model.parameters()).device
        tensor_bicubic = tensor_bicubic.to(device)
        with torch.no_grad():
            rep_bicubic = model.forward(tensor_bicubic)
        tensor_default = model.transform(image)
        if tensor_default.dim() == 3:
            tensor_default = tensor_default.unsqueeze(0)
        tensor_default = tensor_default.to(device)
        with torch.no_grad():
            rep_default = model.forward(tensor_default)
        assert not torch.allclose(rep_bicubic, rep_default)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model_name",
        [
            "DINOv2_ViT_Large",
            # Add more model names here as needed
        ],
    )
    def test_visual_comparison_bicubic_vs_default(self, model_name):
        """Print first 10 values of representations for visual comparison."""
        image_dir = "test_data/image_examples"
        image_files = [f for f in os.listdir(image_dir) if not f.startswith(".")]
        image_path = os.path.join(image_dir, image_files[0])
        image = Image.open(image_path).convert("RGB")
        from src.experiment.models import model_factory

        model = model_factory(model_name, device="cuda")
        for param in model.parameters():
            param.data = param.data.cuda()
        for buf in model.buffers():
            buf.data = buf.data.cuda()
        model.eval()

        # BICUBIC transform
        bicubic_transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True
                ),
                transforms.ToTensor(),
            ]
        )
        tensor_bicubic = bicubic_transform(image).unsqueeze(0)
        device = next(model.parameters()).device
        tensor_bicubic = tensor_bicubic.to(device)
        with torch.no_grad():
            rep_bicubic = model.forward(tensor_bicubic)

        # Default transform
        tensor_default = model.transform(image)
        if tensor_default.dim() == 3:
            tensor_default = tensor_default.unsqueeze(0)
        tensor_default = tensor_default.to(device)
        with torch.no_grad():
            rep_default = model.forward(tensor_default)

        # Print first 10 values for visual comparison
        print("\n" + "=" * 60)
        print(f"Model: {model_name}")
        print(f"Image: {image_files[0]}")
        print("=" * 60)
        print(f"\nBICUBIC Transform - First 10 values:")
        print(rep_bicubic.flatten()[:10].cpu().numpy())
        print(f"\nDefault Transform - First 10 values:")
        print(rep_default.flatten()[:10].cpu().numpy())
        print(f"\nDifference (BICUBIC - Default):")
        print((rep_bicubic.flatten()[:10] - rep_default.flatten()[:10]).cpu().numpy())
        print("=" * 60 + "\n")

        # This test always passes, it's just for visualization
        assert True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model_name",
        [
            "DINOv2_ViT_Large",
            # Add more model names here as needed
        ],
    )
    def test_cosine_similarity_across_categories(self, model_name):
        """Test cosine similarity between truck and frog images with different transforms."""
        image_dir = "test_data/image_examples"
        truck_path = os.path.join(image_dir, "02_bird.png")
        frog_path = os.path.join(image_dir, "07_horse.png")

        # Check that the images exist
        assert os.path.exists(truck_path), f"Truck image not found at {truck_path}"
        assert os.path.exists(frog_path), f"Frog image not found at {frog_path}"

        truck_img = Image.open(truck_path).convert("RGB")
        frog_img = Image.open(frog_path).convert("RGB")

        # Initialize model
        from src.experiment.models import model_factory

        model = model_factory(model_name, device="cuda")
        for param in model.parameters():
            param.data = param.data.cuda()
        for buf in model.buffers():
            buf.data = buf.data.cuda()
        model.eval()

        device = next(model.parameters()).device

        # BICUBIC transform
        bicubic_transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True
                ),
                transforms.ToTensor(),
            ]
        )

        # Get representations with bicubic transform
        truck_bicubic = bicubic_transform(truck_img).unsqueeze(0).to(device)
        frog_bicubic = bicubic_transform(frog_img).unsqueeze(0).to(device)

        # Save the bicubic transformed images for debugging
        save_image(truck_bicubic, "test_data/truck_bicubic.png")
        save_image(frog_bicubic, "test_data/frog_bicubic.png")

        with torch.no_grad():
            truck_rep_bicubic = model.forward(truck_bicubic)
            frog_rep_bicubic = model.forward(frog_bicubic)

        # Get representations with default transform
        truck_default = model.transform(truck_img)
        if truck_default.dim() == 3:
            truck_default = truck_default.unsqueeze(0)
        truck_default = truck_default.to(device)

        frog_default = model.transform(frog_img)
        if frog_default.dim() == 3:
            frog_default = frog_default.unsqueeze(0)
        frog_default = frog_default.to(device)

        # BICUBIC transform
        regular_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        # Get representations with bicubic transform
        truck_regular = regular_transform(truck_img).unsqueeze(0).to(device)
        frog_regular = regular_transform(frog_img).unsqueeze(0).to(device)

        # Save the regular transformed images for debugging
        save_image(truck_regular, "test_data/truck_regular.png")
        save_image(frog_regular, "test_data/frog_regular.png")

        with torch.no_grad():
            truck_rep_default = model.forward(truck_default)
            frog_rep_default = model.forward(frog_default)

        # Flatten representations for cosine similarity
        truck_rep_bicubic_flat = truck_rep_bicubic.flatten()
        frog_rep_bicubic_flat = frog_rep_bicubic.flatten()
        truck_rep_default_flat = truck_rep_default.flatten()
        frog_rep_default_flat = frog_rep_default.flatten()

        # Compute cosine similarities
        cos_sim_bicubic = torch.nn.functional.cosine_similarity(
            truck_rep_bicubic_flat.unsqueeze(0), frog_rep_bicubic_flat.unsqueeze(0)
        ).item()

        cos_sim_default = torch.nn.functional.cosine_similarity(
            truck_rep_default_flat.unsqueeze(0), frog_rep_default_flat.unsqueeze(0)
        ).item()

        # Print results for analysis
        print("\n" + "=" * 60)
        print(f"Model: {model_name}")
        print(f"Cross-category similarity (truck vs frog)")
        print("=" * 60)
        print(f"BICUBIC transform cosine similarity: {cos_sim_bicubic:.6f}")
        print(f"Default transform cosine similarity: {cos_sim_default:.6f}")
        print(
            f"Difference (BICUBIC - Default): {cos_sim_bicubic - cos_sim_default:.6f}"
        )
        print("=" * 60 + "\n")

        # Assertions: similarity should be relatively low (different categories)
        # and finite (no NaN/Inf)
        assert torch.isfinite(
            torch.tensor(cos_sim_bicubic)
        ), "BICUBIC similarity is not finite"
        assert torch.isfinite(
            torch.tensor(cos_sim_default)
        ), "Default similarity is not finite"
        assert (
            -1.0 <= cos_sim_bicubic <= 1.0
        ), "BICUBIC similarity out of valid range [-1, 1]"
        assert (
            -1.0 <= cos_sim_default <= 1.0
        ), "Default similarity out of valid range [-1, 1]"

        # Different categories should have lower similarity than same category
        # This is a sanity check - typically < 0.8 for different categories
        assert (
            cos_sim_bicubic < 0.95
        ), "Unexpectedly high similarity between different categories (BICUBIC)"
        assert (
            cos_sim_default < 0.95
        ), "Unexpectedly high similarity between different categories (Default)"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model_name",
        [
            "DINOv2_ViT_Large",
            # Add more model names here as needed
        ],
    )
    def test_cosine_similarity_same_category(self, model_name):
        """Test cosine similarity between two different truck images (same category)."""
        image_dir = "test_data/image_examples"

        # Load two different truck images
        truck1_path = os.path.join(image_dir, "09_truck_1.png")
        truck2_path = os.path.join(image_dir, "09_truck_2.png")

        # Check that the images exist
        assert os.path.exists(truck1_path), f"Truck image 1 not found at {truck1_path}"
        assert os.path.exists(truck2_path), f"Truck image 2 not found at {truck2_path}"

        truck1_img = Image.open(truck1_path).convert("RGB")
        truck2_img = Image.open(truck2_path).convert("RGB")

        # Initialize model
        from src.experiment.models import model_factory

        model = model_factory(model_name, device="cuda")
        for param in model.parameters():
            param.data = param.data.cuda()
        for buf in model.buffers():
            buf.data = buf.data.cuda()
        model.eval()

        device = next(model.parameters()).device

        # BICUBIC transform
        bicubic_transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True
                ),
                transforms.ToTensor(),
            ]
        )

        # Get representations with bicubic transform
        truck1_bicubic = bicubic_transform(truck1_img).unsqueeze(0).to(device)
        truck2_bicubic = bicubic_transform(truck2_img).unsqueeze(0).to(device)

        with torch.no_grad():
            truck1_rep_bicubic = model.forward(truck1_bicubic)
            truck2_rep_bicubic = model.forward(truck2_bicubic)

        # Get representations with default transform
        truck1_default = model.transform(truck1_img)
        if truck1_default.dim() == 3:
            truck1_default = truck1_default.unsqueeze(0)
        truck1_default = truck1_default.to(device)

        truck2_default = model.transform(truck2_img)
        if truck2_default.dim() == 3:
            truck2_default = truck2_default.unsqueeze(0)
        truck2_default = truck2_default.to(device)

        with torch.no_grad():
            truck1_rep_default = model.forward(truck1_default)
            truck2_rep_default = model.forward(truck2_default)

        # Flatten representations for cosine similarity
        truck1_rep_bicubic_flat = truck1_rep_bicubic.flatten()
        truck2_rep_bicubic_flat = truck2_rep_bicubic.flatten()
        truck1_rep_default_flat = truck1_rep_default.flatten()
        truck2_rep_default_flat = truck2_rep_default.flatten()

        # Compute cosine similarities
        cos_sim_bicubic = torch.nn.functional.cosine_similarity(
            truck1_rep_bicubic_flat.unsqueeze(0), truck2_rep_bicubic_flat.unsqueeze(0)
        ).item()

        cos_sim_default = torch.nn.functional.cosine_similarity(
            truck1_rep_default_flat.unsqueeze(0), truck2_rep_default_flat.unsqueeze(0)
        ).item()

        # Print results for analysis
        print("\n" + "=" * 60)
        print(f"Model: {model_name}")
        print(f"Same-category similarity (truck_1 vs truck_2)")
        print("=" * 60)
        print(f"BICUBIC transform cosine similarity: {cos_sim_bicubic:.6f}")
        print(f"Default transform cosine similarity: {cos_sim_default:.6f}")
        print(
            f"Difference (BICUBIC - Default): {cos_sim_bicubic - cos_sim_default:.6f}"
        )
        print("=" * 60 + "\n")

        # Assertions: similarity should be relatively high (same category)
        # and finite (no NaN/Inf)
        assert torch.isfinite(
            torch.tensor(cos_sim_bicubic)
        ), "BICUBIC similarity is not finite"
        assert torch.isfinite(
            torch.tensor(cos_sim_default)
        ), "Default similarity is not finite"
        assert (
            -1.0 <= cos_sim_bicubic <= 1.0
        ), "BICUBIC similarity out of valid range [-1, 1]"
        assert (
            -1.0 <= cos_sim_default <= 1.0
        ), "Default similarity out of valid range [-1, 1]"

        # Same category images should have higher similarity than different categories
        # Typically > 0.5 or even > 0.7 for same category in vision models
        assert (
            cos_sim_bicubic > 0.3
        ), f"Unexpectedly low similarity for same category (BICUBIC): {cos_sim_bicubic:.6f}"
        assert (
            cos_sim_default > 0.3
        ), f"Unexpectedly low similarity for same category (Default): {cos_sim_default:.6f}"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model_name",
        [
            "DINOv2_ViT_Large",
            # Add more model names here as needed
        ],
    )
    def test_batch_representation_shape(self, model_name):
        batch_dir = "test_data/cifar-10-batches-py"
        batch_name = "data_batch_1"
        batch_path = os.path.join(batch_dir, batch_name)
        limit_n = 16  # keep small for test
        with open(batch_path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
        data = entry["data"]
        images = []
        for row in data[:limit_n]:
            img = np.reshape(row, (3, 32, 32)).transpose(1, 2, 0)
            images.append(Image.fromarray(img))
        from src.experiment.models import model_factory

        model = model_factory(model_name, device="cuda")
        for param in model.parameters():
            param.data = param.data.cuda()
        for buf in model.buffers():
            buf.data = buf.data.cuda()
        model.eval()
        image_tensors = []
        device = next(model.parameters()).device
        for img in images:
            t = model.transform(img)
            if t.dim() == 3:
                t = t.unsqueeze(0)
            image_tensors.append(t)
        image_batch = torch.cat(image_tensors, dim=0).to(device)
        with torch.no_grad():
            batch_rep = model.forward(image_batch)
        assert batch_rep.shape[0] == limit_n
        assert batch_rep.shape[-1] == 1024

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model_name",
        [
            "DINOv2_ViT_Large",
            # Add more model names here as needed
        ],
    )
    def test_batch_bicubic_vs_default_transform_difference(self, model_name):
        batch_dir = "test_data/cifar-10-batches-py"
        batch_name = "data_batch_1"
        batch_path = os.path.join(batch_dir, batch_name)
        limit_n = 8  # keep small for test
        with open(batch_path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
        data = entry["data"]
        images = []
        for row in data[:limit_n]:
            img = np.reshape(row, (3, 32, 32)).transpose(1, 2, 0)
            images.append(Image.fromarray(img))
        from src.experiment.models import model_factory

        model = model_factory(model_name, device="cuda")
        for param in model.parameters():
            param.data = param.data.cuda()
        for buf in model.buffers():
            buf.data = buf.data.cuda()
        model.eval()
        bicubic_transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True
                ),
                transforms.ToTensor(),
            ]
        )
        device = next(model.parameters()).device
        bicubic_tensors = [
            bicubic_transform(img).unsqueeze(0).to(device) for img in images
        ]
        bicubic_batch = torch.cat(bicubic_tensors, dim=0)
        with torch.no_grad():
            rep_bicubic = model.forward(bicubic_batch)
        default_tensors = []
        for img in images:
            t = model.transform(img)
            if t.dim() == 3:
                t = t.unsqueeze(0)
            default_tensors.append(t.to(device))
        default_batch = torch.cat(default_tensors, dim=0)
        with torch.no_grad():
            rep_default = model.forward(default_batch)
        assert not torch.allclose(rep_bicubic, rep_default)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model_name",
        [
            "DINOv2_ViT_Large",
            # Add more model names here as needed
        ],
    )
    def test_single_image_representation_no_nan_or_inf(self, model_name):
        image_dir = "test_data/image_examples"
        image_files = [f for f in os.listdir(image_dir) if not f.startswith(".")]
        image_path = os.path.join(image_dir, image_files[0])
        image = Image.open(image_path).convert("RGB")
        from src.experiment.models import model_factory

        model = model_factory(model_name, device="cuda")
        for param in model.parameters():
            param.data = param.data.cuda()
        for buf in model.buffers():
            buf.data = buf.data.cuda()
        model.eval()
        tensor_image = model.transform(image)
        if tensor_image.dim() == 3:
            tensor_image = tensor_image.unsqueeze(0)
        device = next(model.parameters()).device
        tensor_image = tensor_image.to(device)
        with torch.no_grad():
            rep = model.forward(tensor_image)
        assert not torch.isnan(rep).any(), "Representation contains NaN values"
        assert torch.isfinite(rep).all(), "Representation contains Inf values"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model_name",
        [
            "DINOv2_ViT_Large",
            # Add more model names here as needed
        ],
    )
    def test_batch_representation_no_nan_or_inf(self, model_name):
        batch_dir = "test_data/cifar-10-batches-py"
        batch_name = "data_batch_1"
        batch_path = os.path.join(batch_dir, batch_name)
        limit_n = 16  # keep small for test
        with open(batch_path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
        data = entry["data"]
        images = []
        for row in data[:limit_n]:
            img = np.reshape(row, (3, 32, 32)).transpose(1, 2, 0)
            images.append(Image.fromarray(img))
        from src.experiment.models import model_factory

        model = model_factory(model_name, device="cuda")
        for param in model.parameters():
            param.data = param.data.cuda()
        for buf in model.buffers():
            buf.data = buf.data.cuda()
        model.eval()
        image_tensors = []
        device = next(model.parameters()).device
        for img in images:
            t = model.transform(img)
            if t.dim() == 3:
                t = t.unsqueeze(0)
            image_tensors.append(t)
        image_batch = torch.cat(image_tensors, dim=0).to(device)
        with torch.no_grad():
            batch_rep = model.forward(image_batch)
        assert not torch.isnan(
            batch_rep
        ).any(), "Batch representation contains NaN values"
        assert torch.isfinite(
            batch_rep
        ).all(), "Batch representation contains Inf values"


class TestBaseModel:
    """Test BaseModel functionality."""

    def test_base_model_interface(self):
        """Test that BaseModel provides expected interface."""

        class TestModel(BaseModel):
            def forward(self, x):
                return x

        model = TestModel()

        # Should have required methods
        assert hasattr(model, "get_num_parameters")
        assert hasattr(model, "get_model_info")
        assert callable(model.forward)

    def test_parameter_counting(self):
        """Test parameter counting in BaseModel."""

        class SimpleModel(BaseModel):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        param_count = model.get_num_parameters()

        # Linear layer: (10 * 5) + 5 = 55 parameters
        assert param_count == 55

    def test_model_info(self):
        """Test model info dictionary."""

        class InfoModel(BaseModel):
            def __init__(self):
                super().__init__()
                self._input_shape = (784,)
                self.linear = nn.Linear(784, 10)

            def forward(self, x):
                return self.linear(x)

        model = InfoModel()
        info = model.get_model_info()

        assert "model_class" in info
        assert "num_parameters" in info
        assert "input_shape" in info
        assert info["model_class"] == "InfoModel"
        assert info["input_shape"] == (784,)


def save_image(tensor, path):
    """Utility to save a tensor as an image for debugging."""
    grid = torchvision.utils.make_grid(tensor.cpu())
    ndarr = grid.mul(255).byte().numpy().transpose(1, 2, 0)
    img = Image.fromarray(ndarr)
    img.save(path)
