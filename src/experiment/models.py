"""Machine learning models for MNIST classification.

This module implements three types of models:


All models follow consistent interfaces and integrate with the Pydantic
configuration system for reproducible experiments.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from .schemas import Model, ModelConfig

logger = logging.getLogger(__name__)

from bml_vision import (
    Clip_ViT_Large,
    CORnet_RT,
    DINOv2_ViT_Large,
    Fair_Clip_ViT_Large,
    Fair_SimCLR_ViT_Large,
    Fair_SLIP_ViT_Large,
    Google_ViT_Large,
    Inception_V3,
)

# ---- Base Model Interface ----


def model_factory(model_name, device="cuda"):
    """
    Creates and returns a model instance from the BML Vision Toolkit.

    Args:
        model_name (str): Name of the model to create
        device (str): Device to place the model on

    Returns:
        model: An instance of a BML Vision model
    """
    model_mapping = {
        "Google_ViT_Large_224": Google_ViT_Large,
        "FAIR_SLIP_ViT_Large_CLIP_25": Fair_Clip_ViT_Large,
        "FAIR_SLIP_ViT_Large_SimCLR_25": Fair_SimCLR_ViT_Large,
        "FAIR_SLIP_ViT_Large_SLIP_25": Fair_SLIP_ViT_Large,
        "DINOv2_ViT_Large": DINOv2_ViT_Large,
        "CORnet_RT": CORnet_RT,
        "Inception_V3": Inception_V3,
        "Clip_ViT_Large": Clip_ViT_Large,
        
        
    }

    if model_name not in model_mapping:
        raise ValueError(f"Unknown model name: {model_name}")

    # Create model instance
    model_class = model_mapping[model_name]
    model = model_class()
    model.to(device)
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    return model


class BaseModel(nn.Module):
    """Base class for all models with common functionality."""

    def __init__(self):
        super().__init__()
        self._input_shape = None
        self._num_parameters = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError

    def get_num_parameters(self) -> int:
        """Count total number of trainable parameters."""
        if self._num_parameters is None:
            self._num_parameters = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        return self._num_parameters

    def get_model_info(self) -> Dict[str, any]:
        """Get model information for logging/debugging."""
        return {
            "model_class": self.__class__.__name__,
            "num_parameters": self.get_num_parameters(),
            "input_shape": self._input_shape,
        }

    """Convolutional neural network for image classification.

    Features:
    - Multiple convolutional layers with configurable channels
    - Batch normalization (optional)
    - Dropout for regularization
    - Global average pooling before classification
    """

    def __init__(self, config: CNNModel):
        super().__init__()
        self.config = config

        # Dynamic input shape based on channels
        if config.input_channels == 1:
            self._input_shape = (config.input_channels, 28, 28)  # MNIST
        elif config.input_channels == 3:
            self._input_shape = (config.input_channels, 32, 32)  # CIFAR-10
        else:
            raise ValueError(f"Unsupported input_channels: {config.input_channels}")

        # Build convolutional layers
        conv_layers = []
        in_channels = config.input_channels

        for out_channels in config.channels:
            layer_group = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size,
                    padding=config.kernel_size // 2,  # Same padding
                    bias=not config.batch_norm,
                )
            ]

            if config.batch_norm:
                layer_group.append(nn.BatchNorm2d(out_channels))

            layer_group.extend(
                [
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout2d(config.dropout),
                ]
            )

            conv_layers.extend(layer_group)
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Global average pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(config.channels[-1], config.num_classes)

        # Initialize weights
        self._init_weights()

        logger.info(
            f"Created CNNClassifier: channels={config.channels}, kernel_size={config.kernel_size}"
        )

    def _init_weights(self):
        """Initialize CNN weights using appropriate schemes."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # Ensure correct input shape
        if x.dim() == 3:  # Add batch dimension if missing
            x = x.unsqueeze(0)

        # Convolutional feature extraction
        features = self.conv_layers(x)

        # Global pooling and classification
        pooled = self.global_pool(features)
        flattened = pooled.view(pooled.size(0), -1)

        return self.classifier(flattened)

    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract feature maps from intermediate layers.

        Useful for visualization and analysis.

        Args:
            x: Input tensor

        Returns:
            List of feature map tensors from each conv layer
        """
        feature_maps = []

        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            if isinstance(layer, nn.ReLU):  # Save after activation
                feature_maps.append(x.detach().clone())

        return feature_maps


class BMLVisionWrapper(BaseModel):
    """Wrapper for BML Vision Toolkit models."""
    
    def __init__(self, config: Model):
        super().__init__()
        self.config = config
        
        # Determine device 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create the BML Vision model using existing factory
        self.model = model_factory(config.name, device)
        
        # Set input shape based on model
        if "ViT" in config.name:
            self._input_shape = (3, 224, 224)
        elif config.name == "Inception_V3":
            self._input_shape = (3, 299, 299)  
        else:
            self._input_shape = (3, 224, 224)
        
        logger.info(f"Created BMLVisionWrapper: model_name={config.name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through BML Vision model."""
        return self.model(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the model (if supported)."""
        if hasattr(self.model, 'get_features'):
            return self.model.get_features(x)
        else:
            # For models without explicit feature extraction,
            # return the forward pass (assuming it returns features)
            return self.forward(x)


# ---- Model Factory Functions ----


def create_model(config: ModelConfig) -> BaseModel:
    """Factory function to create models from configuration.

    Args:
        config: Model configuration (discriminated union)

    Returns:
        Instantiated model

    Raises:
        ValueError: If model type is not supported
    """
    if isinstance(config, Model):
        # All models in the Model class are BML Vision models
        return BMLVisionWrapper(config)
    else:
        raise ValueError(f"Unsupported model type: {type(config)}")


def count_model_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters in a model with breakdown.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
    }


def get_model_summary(model: BaseModel, input_shape: Tuple[int, ...]) -> str:
    """Generate a model summary string.

    Args:
        model: Model to summarize
        input_shape: Input shape (without batch dimension)

    Returns:
        Formatted summary string
    """
    param_counts = count_model_parameters(model)
    model_info = model.get_model_info()

    summary = f"""
Model Summary:
--------------
Class: {model_info['model_class']}
Input Shape: {input_shape}
Parameters:
  - Total: {param_counts['total_parameters']:,}
  - Trainable: {param_counts['trainable_parameters']:,}
  - Non-trainable: {param_counts['non_trainable_parameters']:,}

Architecture:
{model}
"""
    return summary


def validate_model_config(config: ModelConfig, input_shape: Tuple[int, ...]) -> bool:
    """Validate model configuration against input data shape.

    Args:
        config: Model configuration
        input_shape: Expected input shape

    Returns:
        True if configuration is valid
    """
    try:
        if isinstance(config, Model):
            # BML Vision models expect 3D input (C, H, W)
            if len(input_shape) != 3:
                logger.error(
                    f"BML Vision models expect 3D input (C, H, W), got {len(input_shape)}D"
                )
                return False
            
            # Most BML Vision models expect 3-channel RGB input
            if input_shape[0] != 3:
                logger.error(
                    f"BML Vision models expect 3-channel RGB input, got {input_shape[0]} channels"
                )
                return False
            
            # Validate input size based on model type
            if "ViT" in config.name and input_shape[1:] != (224, 224):
                logger.warning(
                    f"ViT models typically expect 224x224 input, got {input_shape[1:]}"
                )
            elif config.name == "Inception_V3" and input_shape[1:] != (299, 299):
                logger.warning(
                    f"Inception_V3 expects 299x299 input, got {input_shape[1:]}"
                )

        else:
            # Invalid config type
            logger.error(f"Invalid model config type: {type(config)}")
            return False

        logger.info(
            f"Model configuration validation passed for {config.name}"
        )
        return True

    except Exception as e:
        logger.error(f"Model configuration validation failed: {e}")
        return False


# ---- Training Utilities ----


def create_optimizer(model: nn.Module, config) -> Optimizer:
    """Create optimizer from training configuration.

    Args:
        model: Model to optimize
        config: Training configuration with optimizer settings

    Returns:
        Configured optimizer
    """
    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def create_scheduler(optimizer: Optimizer, config):
    """Create learning rate scheduler from configuration.

    Args:
        optimizer: Optimizer to schedule
        config: Training configuration with scheduler settings

    Returns:
        Configured scheduler or None
    """
    if not config.scheduler:
        return None

    scheduler_config = config.scheduler

    if scheduler_config.name == "reduce_lr_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_config.factor,
            patience=scheduler_config.patience,
            min_lr=scheduler_config.min_lr,
        )
    elif scheduler_config.name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.patience,  # Reuse patience as step_size
            gamma=scheduler_config.factor,
        )
    elif scheduler_config.name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,  # Default, should be set based on total epochs
            eta_min=scheduler_config.min_lr,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_config.name}")


__all__ = [
    "BaseModel",
    "BMLVisionWrapper",
    "create_model",
    "model_factory",
    "count_model_parameters",
    "get_model_summary",
    "validate_model_config",
    "create_optimizer",
    "create_scheduler",
]
