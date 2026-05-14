"""
Temporal encoders: (TCN/Transformer) for the ST-GNN architecture.
    - Plug-in components for modelling temporal structure in node embeddings
    - Both encoders:
        - input [B, N, W, D]
        - exchange/aggregate information along the W (time) axis
        - return [B, N, D, W] (same content, permuted).

Classes:
    - TCN:
        TemporalBlock - residual block with two dilated causal convolutions
        TCN           - stacks TemporalBlocks with exponential dilation
                      - number of TemporalBlocks is computed automatically so that the receptive field >= the input window W.
                      - returns a full temporal sequence preserving the window dimension: [B, N_nodes, D, W] (features-first).

    - Transformer:
        SinusoidalPositionalEncoding
        TemporalTransformerEncoder
"""

import math
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """
    - enforces causality for TCN by removing the "right side" (==future) padding, so output length == input length.    
    """
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Single residual block: two dilated causal convolutions + skip connection.

    Conv1d -> Chomp1d -> ReLU -> Dropout  (x2, sequentially)
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

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        return self.relu(out + x)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for the ST-GNN
        - dynamically computes the number of layers so that the receptive field
        covers the historical window W
        - Hidden dim is constant across all layers == (D_out == D_in)
    
    Forward():
        Input: [B, N_nodes, W, D_in] (spatial latent sequence)
        Return: [B, N_nodes, D_in, W] (temporal sequence == convolution over time (W)) 
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

        # build TemporalBlock
        layers = []
        for i in range(num_layers):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(
                n_inputs=input_dim,
                n_outputs=input_dim,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=padding,
                dropout=dropout,
            ))
        self.network = nn.Sequential(*layers)

        #  receptive field param used for logging/debugging
        self.receptive_field = 2*((kernel_size - 1) * (2 ** num_layers - 1)) + 1

    def forward(self, h_4d: torch.Tensor) -> torch.Tensor:
        B, N_nodes, W, D_in = h_4d.shape
        x = h_4d.reshape(B * N_nodes, W, D_in).permute(0, 2, 1)  # permute to Conv1d format [B*N, D, W]
        # Causal temporal convolutions
        y = self.network(x)  # [B*N, D_in, W]
        return y.view(B, N_nodes, D_in, W)  #Note: D_in == D_out in our use case
        
        #*Old implementation below: Return only last dim (could actually restore this in future i think, cause it yielded same performance i believe)
        # return only last terminal state of TCN output -> Restore batch and node dims
        #return z.view(B, N_nodes, -1)  # [B, N_nodes, D_in]

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for TemporalTransformerEncoder
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [1, max_len, d_model] — broadcastable over batch dim
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, W, D] → x + PE[:, :W, :]"""
        return x + self.pe[:, : x.size(1), :]


class TemporalTransformerEncoder(nn.Module):
    """
    Transformer encoder over the temporal (W) dimension.

    Architecture:
        - Sinusoidal positional encoding (fixed, generalises across W)
        - Pre-LN TransformerEncoderLayer (norm_first=True)
        - Bidirectional: all W steps are historical
    
    Forward():
        Input: [B, N_nodes, W, D_in] (spatial latent sequence)
        Return: [B, N_nodes, D_in, W] (temporal sequence == convolution over time (W)) 
    """

    def __init__(
        self,
        input_dim: int,
        window_size: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.pos_enc = SinusoidalPositionalEncoding(input_dim, max_len=window_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True, 
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, h_4d: torch.Tensor) -> torch.Tensor:
        B, N_nodes, W, D = h_4d.shape
        x = h_4d.reshape(B * N_nodes, W, D)
        x = self.pos_enc(x)
        y = self.encoder(x)  # [B*N, W, D]
        return y.reshape(B, N_nodes, W, D).permute(0, 1, 3, 2).contiguous()
