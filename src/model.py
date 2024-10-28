"""
-------------------------------------------------------
This module contains the implementation of the Sparse Autoencoder model
that learns a dictionary of features from the input data.
-------------------------------------------------------
Author:  einsteinoyewole
=Email:   eo2233@nyu.edu
__updated__ = "10/27/24"
-------------------------------------------------------
"""


# Imports
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple, Dict
from dataclasses import dataclass

# Constants

@dataclass
class AutoencoderConfig:
    """
    -------------------------------------------------------
    Configuration class for the Sparse Autoencoder
    -------------------------------------------------------
    Parameters:
        input_dim - Dimension of input features (positive integer)
        hidden_dim - Dimension of hidden layer, must be >= input_dim (positive integer)
        l1_coefficient - Coefficient for L1 regularization on the hidden layer (float > 0)
    -------------------------------------------------------
    """
    input_dim: int
    hidden_dim: int  # Must be >= input_dim (overcomplete)
    l1_coefficient: float = 0.1

class SparseAutoencoder(nn.Module):
    """
    -------------------------------------------------------
    Overcomplete sparse autoencoder for extracting monosemantic features
    -------------------------------------------------------
    Parameters:
        config - Configuration object for the autoencoder (AutoencoderConfig)
    """
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        assert config.hidden_dim >= config.input_dim, "Hidden dim must be >= input_dim for overcompleteness"

        self.encoder = nn.Linear(config.input_dim, config.hidden_dim)
        self.decoder = nn.Linear(config.hidden_dim, config.input_dim)
        self.config = config

    def encode(self, x: Tensor) -> Tensor:
        """
         -------------------------------------------------------
         Encodes input data with pre-encoder bias adjustment
         Pre-encoder bias allows for better reconstruction of the input
         -------------------------------------------------------
         Parameters:
             x - Input tensor of shape (batch_size, input_dim) (torch.Tensor)
         Returns:
             encoded - Encoded representation of shape (batch_size, hidden_dim) (torch.Tensor)
         -------------------------------------------------------
         """
        x_bar = x - self.decoder.bias
        return F.relu(self.encoder(x_bar))

    def decode(self, h: Tensor) -> Tensor:
        """
        -------------------------------------------------------
        Decodes hidden representations back to input space
        -------------------------------------------------------
        Parameters:
            h - Hidden representation tensor of shape (batch_size, hidden_dim) (torch.Tensor)
        Returns:
            decoded - Reconstructed input of shape (batch_size, input_dim) (torch.Tensor)
        -------------------------------------------------------
        """
        return self.decoder(h)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        -------------------------------------------------------
        Forward pass through the autoencoder
        -------------------------------------------------------
        Parameters:
            x - Input tensor of shape (batch_size, input_dim) (torch.Tensor)
        Returns:
            x_hat - Reconstructed input of shape (batch_size, input_dim) (torch.Tensor)
            h - Hidden representation of shape (batch_size, hidden_dim) (torch.Tensor)
        -------------------------------------------------------
        """
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h
