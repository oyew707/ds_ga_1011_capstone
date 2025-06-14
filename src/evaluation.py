"""
-------------------------------------------------------
Contains functions for evaluating model performance
-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "10/27/24"
-------------------------------------------------------
"""

# Imports
import warnings
from typing import Tuple, List, Dict, Callable
from torch.nn import functional as F
from torch import Tensor, sum, abs, log as torch_log, quantile
from src.logger import getlogger
import matplotlib.pyplot as plt
import os

# Constants
warnings.filterwarnings('ignore')
log = getlogger(__name__, 'info')


def mse_loss(x: Tensor, x_hat: Tensor) -> Tensor:
    """
    -------------------------------------------------------
    Calculates the mean squared error loss between the original and reconstructed input tensors
    loss = (1/|X|)∑||x - x̂||²
    -------------------------------------------------------
    Parameters:
        x - Original input tensor of shape (batch_size, input_dim) (torch.Tensor)
        x_hat - Reconstructed input tensor of shape (batch_size, input_dim) (torch.Tensor)
    Returns:
        loss - Mean squared error loss (batch_size,) (torch.Tensor)
    -------------------------------------------------------
    """
    res = F.mse_loss(x_hat, x, reduction='none').mean(dim=1)
    log.debug(f'MSE Loss: {res[:5]}')
    return res


def l1_regularization(h: Tensor, lambda_l1: float) -> Tensor:
    """
    -------------------------------------------------------
    Calculates the L1 regularization term on hidden activations
    λ||h||₁
    -------------------------------------------------------
    Parameters:
        h - Hidden layer activations of shape (batch_size, hidden_dim) (torch.Tensor)
    Returns:
        l1_regularization - L1 regularization term (torch.Tensor)
    -------------------------------------------------------
    """
    assert lambda_l1 > 0, "L1 regularization coefficient must be > 0"
    reg = abs(h).sum() * lambda_l1
    log.debug(f'L1 Regularization: {reg}')
    return reg


def loss_function(x: Tensor, x_hat: Tensor, h: Tensor, l1_coefficient: float) -> dict[
    str, Tensor]:
    """
    -------------------------------------------------------
    Calculates combined loss: MSE reconstruction + L1 regularization
    -------------------------------------------------------
    Parameters:
        x - Original input tensor of shape (batch_size, input_dim) (torch.Tensor)
        x_hat - Reconstructed input tensor of shape (batch_size, input_dim) (torch.Tensor)
        h - Hidden layer activations of shape (batch_size, hidden_dim) (torch.Tensor)
        l1_coefficient - L1 regularization coefficient (float)
    Returns:
        total_loss - {'loss': Total loss (MSE + λ||h||₁)
                    'mse_loss': Mean squared reconstruction error
                    'l1_regularization': L1 regularization term (torch.Tensor)}
    -------------------------------------------------------
    """
    mse = mse_loss(x, x_hat)
    l1 = l1_regularization(h, l1_coefficient)
    # Combined loss as specified in equation (1) from the proposal
    # L = (1/|X|)∑||x - x̂||² + λ||h||₁
    total_loss = mse + l1
    log.debug(f'Total Loss: {total_loss[:5]}')
    return {
            'loss': total_loss,
            'mse_loss': mse,
            'l1_regularization': l1.unsqueeze(0)
        }


def classify_monosemantic_features(
        specificity_scores: Tensor,
        threshold: float = 1.0
) -> Tuple[Tensor, Tensor]:
    """
    -------------------------------------------------------
    Classifies features as monosemantic based on specificity scores
    -------------------------------------------------------
    Parameters:
        specificity_scores - Tensor of activation specificity scores (torch.Tensor)
        threshold - Minimum score to be considered monosemantic (float > 0)
    Returns:
        monosemantic_mask - Boolean mask identifying monosemantic features (torch.Tensor)
        fraction_monosemantic - Fraction of features classified as monosemantic (float)
    -------------------------------------------------------
    """
    monosemantic_mask = specificity_scores >= threshold
    fraction_monosemantic = monosemantic_mask.float().mean()

    return monosemantic_mask, fraction_monosemantic


def calculate_activation_specificity(
        feature_activations: Tensor,
        token_concept_probs: Tensor,
        background_probs: Tensor,
        activation_threshold: float = 0.75
) -> List:
    """
    -------------------------------------------------------
    Calculates activation specificity scores for features
    -------------------------------------------------------
    Parameters:
        feature_activations - Feature activation matrix of shape (n_samples, n_features) (torch.Tensor)
        token_concept_probs - P(s|Concept) probabilities of shape (n_samples,) (torch.Tensor)
        background_probs - P(s) background probabilities of shape (n_samples,) (torch.Tensor)
        activation_threshold - Percentile threshold for activation consideration (float between 0 and 1)
    Returns:
        specificity_scores - Tensor of specificity scores for each feature (torch.Tensor)
    -------------------------------------------------------
    """
    log.debug('Calculating activation specificity score')
    likelihood_ratios = torch_log(token_concept_probs / background_probs)

    thresholds = quantile(
        feature_activations,
        1 - activation_threshold,
        dim=0,
        keepdim=True
    )

    active_masks = feature_activations >= thresholds

    specificity_scores = []
    for feature_idx in range(feature_activations.shape[1]):
        active_mask = active_masks[:, feature_idx]
        if active_mask.sum() == 0:
            specificity_scores.append(0.0)
            log.debug(f'Feature {feature_idx} has no active samples')
            continue

        mean_likelihood_ratio = likelihood_ratios[active_mask].mean()
        specificity_scores.append(mean_likelihood_ratio.item())

    return specificity_scores


def plot_training_metrics(metrics: Dict[str, List[float]], run_path: str) -> None:
    """
    -------------------------------------------------------
    Plots training metrics history
    -------------------------------------------------------
    Parameters:
        metrics - Dictionary of training metrics history (Dict[str, List[float]])
        run_path - Path to save the plot (str)
    -------------------------------------------------------
    """
    # Create the run directory if it doesn't exist
    os.makedirs(run_path, exist_ok=True)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each metric
    for metric_name, values in metrics.items():
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, 'o-', linewidth=2, markersize=4, label=metric_name)

    # Customize the plot
    plt.title('Training Metrics', fontsize=14, pad=10)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the plot
    plot_path = os.path.join(run_path, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
