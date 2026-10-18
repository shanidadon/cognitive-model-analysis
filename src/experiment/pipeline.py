"""Main experiment pipeline with ExCa orchestration.

This module contains the core Experiment class that orchestrates the entire
machine learning pipeline, from data loading to model training and evaluation.
Uses ExCa for caching expensive operations and ensuring reproducibility.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import prepare_data, validate_data_shape
from .infra import ExperimentInfra, create_infra
from .models import create_optimizer, create_scheduler, model_factory
from .schemas import ExperimentConfig

logger = logging.getLogger(__name__)

class ExperimentRidge:
    """Ridge regression experiment for feature-based classification.

    Uses DINOv2 features for extraction and scikit-learn's RidgeClassifierCV
    for training a ridge regression classifier.
    """

    def __init__(
        self, config: ExperimentConfig, infra: Optional[ExperimentInfra] = None
    ):
        """Initialize ridge regression experiment.

        Args:
            config: Validated experiment configuration
            infra: Optional pre-configured infrastructure
        """
        self.config = config
        self.infra = infra or create_infra(
            cache_dir=config.experiment.cache_dir,
            experiment_name=config.experiment.name,
        )

        # Set up device
        self.device = torch.device(
            config.experiment.device
            if config.experiment.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize model state
        self.model = None
        self.ridge_model = None
        self.best_alpha = None

        logger.info(f"Initialized ExperimentRidge: {config.experiment.name}")
        logger.info(f"Using device: {self.device}")

    def load_model(self) -> nn.Module:
        """Load DINOv2 model for feature extraction.

        Returns:
            Loaded DINOv2 model
        """
        from .models import model_factory

        model_name = self.config.model.name
        logger.info(f"Loading model: {model_name}")

        model = model_factory(model_name, device=self.device)

        if model is None:
            raise RuntimeError(f"model_factory returned None for model: {model_name}")

        self.model = model
        logger.info(f"Model loaded successfully. Type: {type(model)}")
        return model

    def extract_representations(
        self, dataloader: DataLoader, split_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract DINOv2 representations for a dataset split.

        Args:
            dataloader: DataLoader for the dataset split
            split_name: Name of the split (train/test)

        Returns:
            Tuple of (X, y) where:
            - X: DINOv2 feature vectors [N, 1024]
            - y: Class labels [N]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        @self.infra.cached_stage(f"extract_features")
        def _extract_features(
            _cache_context: Dict[str, Any] = None,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Extract features using DINOv2."""
            X_batch_list = []  # Accumulate feature batches
            y_batch_list = []  # Accumulate label batches

            logger.info(f"Extracting representations for {split_name} split...")
            self.model.eval()

            with torch.no_grad():
                for batch_images, batch_labels in tqdm(
                    dataloader, desc=f"Extracting {split_name} features"
                ):
                    batch_images = batch_images.to(self.device)

                    # Extract features - Forward pass through DINOv2
                    # Output shape: [batch_size, 1024]
                    features = self.model(batch_images)

                    # Handle different output formats
                    if isinstance(features, tuple):
                        features = features[0]

                    # Flatten if needed (e.g., [B, D, 1, 1] -> [B, D])
                    if features.dim() > 2:
                        features = features.view(features.size(0), -1)

                    X_batch_list.append(features.cpu().numpy())
                    y_batch_list.append(batch_labels.cpu().numpy())

            # Concatenate all batches
            X = np.concatenate(X_batch_list, axis=0)  # Shape: [N, 1024]
            y = np.concatenate(y_batch_list, axis=0)  # Shape: [N]

            logger.info(
                f"Extracted {split_name} features: X.shape={X.shape}, "
                f"y.shape={y.shape}, dtype={X.dtype}"
            )
            return X, y

        # Cache key includes dataset config, split, and model
        cache_context = {
            "data_config": self.config.data.model_dump(),
            "split": split_name,
            "model_name": self.config.model.name,
            "model_variant": getattr(self.config.model, "variant", "default"),
        }

        return _extract_features(_cache_context=cache_context)

    def prepare_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Prepare data by extracting features from all splits.

        Returns:
            Dictionary mapping split names to (X, y) tuples
        """
        from .data import create_dataloaders, create_datasets

        # 1. Create datasets (handles CIFAR-10 download, transforms, etc.)
        datasets = create_datasets(self.config.data)

        # 2. Create dataloaders from datasets
        dataloaders = create_dataloaders(datasets, self.config.data)

        # 3. Extract features for each split
        data_dict = {}
        for split_name, dataloader in dataloaders.items():
            X, y = self.extract_representations(dataloader, split_name)
            data_dict[split_name] = (X, y)

        return data_dict

    def train(self) -> None:
        """Train ridge regression model on extracted features."""
        from sklearn.linear_model import RidgeClassifierCV

        logger.info("Preparing data for ridge training...")
        data = self.prepare_data()
        X_train, y_train = data["train"]

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Number of classes: {len(np.unique(y_train))}")

        # Create RidgeClassifierCV with cross-validation
        alphas = np.logspace(-3, 3, 7)  # Alpha values: [0.001, 0.01, ..., 1000]
        self.ridge_model = RidgeClassifierCV(alphas=alphas, cv=5)

        logger.info(f"Training RidgeClassifierCV with alphas: {alphas}")
        self.ridge_model.fit(X_train, y_train)
        self.best_alpha = self.ridge_model.alpha_

        logger.info(f"Ridge model trained. Best alpha: {self.best_alpha}")

    def evaluate(self) -> Dict[str, float]:
        """Evaluate ridge model on test set.

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating ridge model...")
        data = self.prepare_data()
        X_test, y_test = data["test"]

        # Make predictions
        y_pred = self.ridge_model.predict(X_test)

        # Compute metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        test_precision = precision_score(
            y_test, y_pred, average="macro", zero_division=0
        )
        test_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

        metrics = {
            "test_accuracy": test_accuracy,
            "test_f1": test_f1,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "best_alpha": float(self.best_alpha),
        }

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def run(self) -> Dict[str, Any]:
        """Run complete ridge regression experiment pipeline.

        Orchestrates model loading, data preparation, training, and evaluation.

        Returns:
            Dictionary containing experiment results and metrics
        """
        logger.info(f"Starting ExperimentRidge: {self.config.experiment.name}")
        start_time = time.time()

        try:
            # Load DINOv2 model
            logger.info("Loading DINOv2 model...")
            self.load_model()

            # Train ridge classifier
            logger.info("Training ridge classifier...")
            self.train()

            # Evaluate on test set
            logger.info("Evaluating model...")
            metrics = self.evaluate()

            # Compute elapsed time
            elapsed_time = time.time() - start_time

            # Prepare results dictionary
            results = {
                "experiment_name": self.config.experiment.name,
                "model_name": self.config.model.name,
                "test_metrics": metrics,
                "experiment_time": elapsed_time,
                "results_path": "results",
            }

            logger.info(
                f"Experiment completed in {elapsed_time:.2f} seconds. "
                f"Test Accuracy: {metrics.get('test_accuracy', 0):.4f}"
            )
            return results

        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            raise

    @classmethod
    def from_config_file(cls, config_path: Path) -> "ExperimentRidge":
        """Factory method to create experiment from config file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            ExperimentRidge instance
        """
        from .schemas import load_config

        config = load_config(config_path)
        return cls(config)


__all__ = [
    "ExperimentRidge",
]
