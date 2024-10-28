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

# Constants
warnings.filterwarnings('ignore')
log = getlogger(__name__, 'debug')


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
        l1_regularization - L1 regularization term (batch_size,) (torch.Tensor)
    -------------------------------------------------------
    """
    assert lambda_l1 > 0, "L1 regularization coefficient must be > 0"
    reg = abs(h).sum(dim=1) * lambda_l1
    log.debug(f'L1 Regularization: {reg[:5]}')
    return reg


def loss_function(x: Tensor, x_hat: Tensor, h: Tensor, l1_coefficient: float) -> dict[
    str, Callable[[Tensor, Tensor], Tensor] | Tensor]:
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
            'l1_regularization': l1
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
