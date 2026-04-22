"""
CNN Architecture for Learning 1D Euclidean Action (Double Well)

A lightweight 1D convolutional network that maps a discretized Euclidean
path q(τ) to a scalar action S[q]. Key design choices:

- Conv1d layers with same-padding preserve the τ-grid resolution.
- GlobalSum pools over the time axis, making S[q] extensive (additive
  over τ), consistent with the definition S = Σ_i L_i.
- SiLU activations throughout (smooth, works well for physics surrogates).
- No translation equivariance is enforced: Dirichlet BC pins the endpoints,
  breaking the symmetry that would otherwise justify circular padding.
"""

import torch
import torch.nn as nn


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class GlobalSum(nn.Module):
    """Sum over the time axis, making the output extensive.
    
    (B, C, N_tau) → (B, C)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=2)


def create_1d_cnn(n_channels: int = 32, kernel_sizes: list[int] = [3, 5, 7]) -> nn.Sequential:
    """
    Build the 1D action CNN.

    Architecture
    ------------
    - Stack of Conv1d(in_c → n_channels, ks, same-padding) + SiLU for each
      kernel size in kernel_sizes.
    - Pointwise Conv1d(n_channels → n_channels) + SiLU mixer.
    - Pointwise Conv1d(n_channels → 1) projects to a scalar density.
    - GlobalSum integrates the density over τ → scalar action S[q].

    Parameters
    ----------
    n_channels : int
        Number of feature channels in the convolutional layers.
    kernel_sizes : list of int
        Kernel sizes for the initial feature-extraction stack.
        Using multiple sizes captures interactions at different τ-scales.

    Returns
    -------
    nn.Sequential
        The assembled model. Input shape: (B, 1, N_tau).
        Output shape: (B, 1).
    """
    layers = []
    in_c = 1
    for ks in kernel_sizes:
        pad = ks // 2
        layers.append(nn.Conv1d(in_c, n_channels, ks, padding=pad))
        layers.append(nn.SiLU())
        in_c = n_channels

    layers.append(nn.Conv1d(n_channels, n_channels, kernel_size=1))
    layers.append(nn.SiLU())
    layers.append(nn.Conv1d(n_channels, 1, kernel_size=1))
    layers.append(GlobalSum())

    return nn.Sequential(*layers)