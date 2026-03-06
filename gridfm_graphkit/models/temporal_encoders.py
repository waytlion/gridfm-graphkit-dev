"""
Temporal Convolutional Network (TCN) for the ST-GNN architecture.

Implements a causal dilated TCN that maps spatial latent sequences
[B, N_nodes, W, D] to a single target-horizon embedding [B, N_nodes, D].

Dynamic depth: the number of TemporalBlocks is computed automatically
so that the receptive field strictly covers the historical window W.

Classes:
    Chomp1d       - enforces strict causality by removing future padding
    TemporalBlock - residual block with two dilated causal convolutions
    TCN           - stacks TemporalBlocks with exponential dilation
"""

import math
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Remove trailing padding to enforce strict causality.

    After a causal Conv1d with padding = (kernel_size - 1) * dilation,
    the output has extra elements at the end. Chomp1d slices them off
    so output length == input length.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Single residual block: two dilated causal convolutions + skip connection.

    Conv1d -> Chomp1d -> ReLU -> Dropout  (x2, sequentially)
    Plus a 1x1 conv downsample if n_inputs != n_outputs.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        # First dilated causal conv
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        # Second dilated causal conv
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2,
        )

        # Residual connection (1x1 conv if dimensions differ)
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for the ST-GNN.

    Dynamically computes the number of layers so that the receptive field
    strictly covers the historical window W:
        receptive_field = (kernel_size - 1) * (2^L - 1) + 1 >= W
        -> L = ceil(log2((W - 1) / (kernel_size - 1) + 1))

    Hidden dimension is kept constant at input_dim across all layers.

    Input:  [B, N_nodes, W, D_in]   (unfolded spatial latent sequence)
    Output: [B, N_nodes, D_in]      (terminal state vector at t)
    """

    def __init__(
        self,
        input_dim: int,
        window_size: int,
        convs_per_block: int =2,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.window_size = window_size
        self.convs_per_block = convs_per_block

        # Dynamic depth calculation
        num_layers = math.ceil(
            math.log2((window_size - 1) / (self.convs_per_block * (kernel_size - 1)) + 1)
        )
        num_layers = max(1, num_layers)  # at least 1 layer

        num_channels = [input_dim] * num_layers

        # Build TemporalBlock stack
        layers = []
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            layers.append(TemporalBlock(
                n_inputs=in_channels,
                n_outputs=out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=padding,
                dropout=dropout,
            ))

        self.network = nn.Sequential(*layers)

        # Actual receptive field for logging / debugging
        self.receptive_field = (kernel_size - 1) * (2 ** num_layers - 1) + 1

    def forward(self, h_4d: torch.Tensor) -> torch.Tensor:
        """
        h_4d: [B, N_nodes, W, D_in] (unfolded spatial latent sequence)

        Returns: [B, N_nodes, D_in] (terminal state = forecast embedding)
        """
        B, N_nodes, W, D_in = h_4d.shape

        # Merge batch and node dims, permute to Conv1d format [*, D, W]
        x = h_4d.reshape(B * N_nodes, W, D_in).permute(0, 2, 1)  # [B*N, D, W]

        # Causal temporal convolutions
        y = self.network(x)  # [B*N, D_out, W]

        # Extract terminal state (last timestep)
        z = y[:, :, -1]  # [B*N, D_out]

        # Restore batch and node dims
        return z.view(B, N_nodes, -1)  # [B, N_nodes, D_out]
