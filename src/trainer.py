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
        l1_coefficient_start - Initial Coefficient for L1 regularization on the hidden layer (float > 0)
        l1_coefficient_end - Final Coefficient for L1 regularization on the hidden layer (float > l1_coefficient_start)
        warmup_epochs - Number of epochs for L1 regularization warmup (positive integer)
        schedule_type - Type of scheduling for L1 regularization ('cosine', 'linear') (str)
        text_column - Column containing text data (str)
    -------------------------------------------------------
    """
    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 1e-3
    mixed_precision: str = 'fp16'
    run_path: str = 'runs/'
    l1_coefficient_start: float = 0.01
    l1_coefficient_end: float = 1
    warmup_epochs: int = 5  # Number of epochs for warmup
    schedule_type: str = 'cosine'  # Type of scheduling
    text_column: str = "text"


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
        # Input validation
        assert train_config.l1_coefficient_end >= train_config.l1_coefficient_start, "Max L1 coefficient must be >= initial coefficient"
        assert train_config.schedule_type in ['cosine', 'linear', 'exponential'], "Schedule type must be 'cosine' or 'linear'"
        assert train_config.num_epochs > train_config.warmup_epochs > 0, "Warmup epochs must be > 0"

        self.train_config = train_config
        self.l1_coefficient = train_config.l1_coefficient_start
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

    def _update_l1_coefficient(self, epoch: int) -> None:
        """
        -------------------------------------------------------
        Updates the L1 regularization coefficient based on the schedule type
        -------------------------------------------------------
        Parameters:
            epoch - Current epoch number (int)
        """
        if epoch >= self.train_config.warmup_epochs:
            self.l1_coefficient = self.train_config.l1_coefficient_end
            log.debug(f'Setting to the max; Warmup peroid is over, L1 coefficient: {self.l1_coefficient}')
            return

        progress = epoch / self.train_config.warmup_epochs
        diff = self.train_config.l1_coefficient_end - self.train_config.l1_coefficient_start

        if self.train_config.schedule_type == 'linear':
            self.l1_coefficient = self.train_config.l1_coefficient_start + progress * (diff)
            log.debug(f'Linear schedule; Update L1 coefficient: {self.l1_coefficient}')
        elif self.train_config.schedule_type == 'cosine':
            self.l1_coefficient = self.train_config.l1_coefficient_start + 0.5 * diff * (1 - torch.cos(progress * torch.pi))
            log.debug(f'Cosine schedule; Update L1 coefficient: {self.l1_coefficient}')
        elif self.train_config.schedule_type == 'exponential':
            self.l1_coefficient = self.train_config.l1_coefficient_start * (self.train_config.l1_coefficient_end / self.train_config.l1_coefficient_start) ** progress
            log.debug(f'Exponential schedule; Update L1 coefficient: {self.l1_coefficient}')
        else:
            raise NotImplementedError(f'Schedule type {self.train_config.schedule_type} not implemented')


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
                texts = batch[self.train_config.text_column]
                x = self.activation_extractor.extract_activations(texts)['activations']


                # Forward pass
                x_hat, h = self.model(x)
                losses = evaluation.loss_function(x, x_hat, h, self.l1_coefficient)

                # Update metrics
                for k, v in losses.items():
                    accumulated_losses[k].append(v)

                # Backward pass
                self.accelerator.backward(losses['loss'].mean())
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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
        total_metrics = defaultdict(list)
        num_batches = len(dataloader)

        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                # Extract activations in the main process
                texts = batch[self.train_config.text_column]
                x = self.activation_extractor.extract_activations(texts)['activations']

                # Forward pass
                x_hat, h = self.model(x)
                losses = evaluation.loss_function(x, x_hat, h, self.train_config.l1_coefficient_end)

                for k, v in losses.items():
                    total_metrics[k].append(v)

                if idx % 10 == 0:
                    log.debug(f"Input shape: {x.shape} Reconstructed shape: {x_hat.shape}")
                    log.debug(f"Input range: [{x.min().item():.3f}, {x.max().item():.3f}] "
                              f"Reconstructed range: [{x_hat.min().item():.3f}, {x_hat.max().item():.3f}]")
                    log.debug(f"Input mean/std: {x.mean().item():.3f}/{x.std().item():.3f} "
                              f"Reconstructed mean/std: {x_hat.mean().item():.3f}/{x_hat.std().item():.3f}")

        eval_metrics = {k: torch.cat(v).mean().item() for k, v in total_metrics.items()}
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

            # Update L1 coefficient
            self._update_l1_coefficient(epoch)

            # Save checkpoint
            val_loss = val_metrics['loss'] if val_dataloader is not None else train_metrics['loss']
            if val_loss < best_loss:
                torch.save(self.model.state_dict(), model_path)

        return dict(history)
