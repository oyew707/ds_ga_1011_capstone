"""
-------------------------------------------------------
[Program Description]
-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "10/27/24"
-------------------------------------------------------
"""

# Imports
from src import evaluation, logger, model
from tqdm import tqdm
import os
import torch
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
from accelerate import Accelerator
from torch.utils.data import DataLoader
from src.dataset import TextDataset, BaseActivationExtractor

# Constants
log = logger.getlogger(__name__, 'debug')
torch.backends.mps.enabled = False


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

@dataclass
class TrainingConfig:
    """
    -------------------------------------------------------
    Configuration for training settings
    -------------------------------------------------------
    Parameters:
        batch_size - Size of batch for training (positive integer)
        num_epochs - Number of training epochs (positive integer)
        mixed_precision - Mixed precision type ('no', 'fp16', 'bf16') (str)
        learning_rate - Learning rate for optimization (float > 0)
    -------------------------------------------------------
    """
    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 1e-3
    mixed_precision: str = 'fp16'
    run_path: str = 'runs/'


class MonosemanticityTrainer:
    """
    -------------------------------------------------------
    Distributed trainer for sparse autoencoder using Accelerate
    -------------------------------------------------------
    Parameters:
        model - Sparse autoencoder model (SparseAutoencoder)
        optimizer - PyTorch optimizer (torch.optim.Optimizer)
        extractor - Activation extractor for model (BaseActivationExtractor)
        train_config - Training configuration (TrainingConfig)
    -------------------------------------------------------
    """

    def __init__(
            self,
            model: model.SparseAutoencoder,
            optimizer: torch.optim.Optimizer,
            extractor: BaseActivationExtractor,
            train_config: TrainingConfig
    ):
        self.train_config = train_config
        self.run_path = os.path.join(os.getcwd(), train_config.run_path)
        if not os.path.isdir(self.run_path):
            os.makedirs(self.run_path)

        self.accelerator = Accelerator(
            mixed_precision=train_config.mixed_precision,
            cpu = get_device() != 'cuda'
        )
        self.activation_extractor = extractor

        # Prepare model, optimizer for distributed training
        self.model, self.optimizer = self.accelerator.prepare(model, optimizer)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        -------------------------------------------------------
        Trains the model for one epoch
        -------------------------------------------------------
        Parameters:
            dataloader - DataLoader containing training text data (DataLoader)
        Returns:
            epoch_metrics - Dictionary of averaged metrics for the epoch (Dict[str, float])
        -------------------------------------------------------
        """
        log.debug("Training epoch")
        self.model.train()
        accumulated_losses = defaultdict(list)

        with self.accelerator.accumulate(self.model):
            for batch in dataloader:
                # Extract activations in the main process
                texts = batch[0]
                x = self.activation_extractor.extract_activations(texts)['activations']

                # Forward pass
                x_hat, h = self.model(x)
                losses = evaluation.loss_function(x, x_hat, h, self.train_config.learning_rate)

                # Update metrics
                for k, v in losses.items():
                    accumulated_losses[k].append(v)

                # Backward pass
                self.accelerator.backward(losses['loss'].mean())

                self.optimizer.step()
                self.optimizer.zero_grad()

        # Average metrics
        epoch_metrics = {k: torch.cat(v).mean().item() for k, v in accumulated_losses.items()}

        return epoch_metrics

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        -------------------------------------------------------
        Evaluates the model on validation/test data
        -------------------------------------------------------
        Parameters:
            dataloader - DataLoader containing evaluation data (DataLoader)
        Returns:
            eval_metrics - Dictionary of averaged metrics (Dict[str, float])
        -------------------------------------------------------
        """
        self.model.eval()
        total_metrics = defaultdict(float)
        num_batches = len(dataloader)

        with torch.no_grad():
            for batch in dataloader:
                # Extract activations in the main process
                texts = batch[0]
                x = self.activation_extractor.extract_activations(texts)['activations']

                # Forward pass
                x_hat, h = self.model(x)
                losses = evaluation.loss_function(x, x_hat, h, self.train_config.learning_rate)

                for k, v in losses.items():
                    total_metrics[k] += v.item()

        eval_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        return eval_metrics

    def train(
            self,
            train_dataloader: DataLoader,
            val_dataloader: Optional[DataLoader] = None
    ) -> Dict[str, list]:
        """
        -------------------------------------------------------
        Trains the model for specified number of epochs
        -------------------------------------------------------
        Parameters:
            train_dataloader - DataLoader for training data (DataLoader)
            val_dataloader - Optional DataLoader for validation data (DataLoader or None)
        Returns:
            history - Dictionary of training metrics history (Dict[str, list])
        -------------------------------------------------------
        """
        # Prepare dataloaders for distributed training
        train_dataloader = self.accelerator.prepare(train_dataloader)
        if val_dataloader is not None:
            val_dataloader = self.accelerator.prepare(val_dataloader)

        history = defaultdict(list)
        best_loss = float('inf')
        model_path = os.path.join(self.run_path, 'model.pkl')
        log.info('Saving model to: ' + model_path)
        for epoch in tqdm(range(self.train_config.num_epochs)):
            log.info(f"\nEpoch {epoch + 1}/{self.train_config.num_epochs}")

            # Train epoch
            train_metrics = self.train_epoch(train_dataloader)
            for k, v in train_metrics.items():
                history[f'train_{k}'].append(v)

            log.info(
                f"\nEpoch {epoch + 1}/{self.train_config.num_epochs}"
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"MSE: {train_metrics['mse_loss']:.4f}, "
                f"L1: {train_metrics['l1_regularization']:.4f}"
            )
            # Validation
            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                for k, v in val_metrics.items():
                    history[f'val_{k}'].append(v)

                log.info(
                    f"\nEpoch {epoch + 1}/{self.train_config.num_epochs}"
                    f"Validation - Loss: {val_metrics['loss']:.4f}, "
                    f"MSE: {val_metrics['mse_loss']:.4f}, "
                    f"L1: {val_metrics['l1_regularization']:.4f}"
                )

            # Save checkpoint
            val_loss = val_metrics['loss'] if val_dataloader is not None else train_metrics['loss']
            if val_loss < best_loss:
                torch.save(self.model.state_dict(), model_path)

        return dict(history)
