from __future__ import annotations
import math
import json
import numpy as np

from .base_backbone import BaseBackbone
from ..builder import BACKBONES

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

@BACKBONES.register_module()
class MTMamba(nn.Module):
    def __init__(self, d_input=900, d_model=900, n_layer=1, d_state=16, d_conv=4, expand=2, dt_rank='auto', conv_bias=True, bias=False):
        super().__init__()

        self.d_input = d_input  # Input size
        self.d_model = d_model  # The hidden layer dimensions of the model
        self.n_layer = n_layer  # The number of layers in the model
        self.d_state = d_state  # Dimension of state space
        self.d_conv = d_conv    # Dimension of the convolution kernel
        self.expand = expand    # Expansion Factor
        self.dt_rank = dt_rank  # The rank of the input depends on the step size Δ
        self.conv_bias = conv_bias  # Whether the convolutional layer uses a bias term
        self.bias = bias    # Whether other layers (such as linear layers) use bias terms
        # Calculate the inner dimension, the expanded dimension
        self.d_inner = int(self.expand * self.d_model)
        # If dt_rank is not specified, it is automatically calculated
        if self.dt_rank == 'auto':
            # Automatically calculate the rank of Δ based on the hidden layer dimension
            self.dt_rank = math.ceil(self.d_model / 16)

        self.position = PositionalEncoding(self.d_input, self.d_model)

        # Create an embedding layer
        self.embedding = nn.Linear(self.d_input, self.d_model)
        # Create a module list containing multiple residual blocks
        self.layers = nn.ModuleList([ResidualBlock(self.d_model, self.d_inner, self.d_state, self.d_conv, self.dt_rank, self.conv_bias, self.bias) for _ in range(self.n_layer)])
        # Create a RMSNorm module for normalization operations
        self.norm_f = RMSNorm(self.d_model)
        # Create a linear layer for the final output
        self.lm_head = nn.Linear(self.d_model, self.d_input, bias=False)
        # Tie the output weights of the linear layer to the weights of the embedding layer
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.

    def forward(self, input):
        x = self.embedding(input)
        # Positional encoding
        x = self.position(x)
        # Traverse all residual blocks
        for layer in self.layers:
            x = layer(x)
        # Normalization
        x = self.norm_f(x)

        logits = x.view(x.size(0), -1)    # If don't do lm_head, just output it directly

        return (logits, )

# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, window, d_model):
        super().__init__()

        self.register_buffer('d_model', torch.tensor(d_model, dtype=torch.float64))

        pe = torch.zeros(window, d_model)
        for pos in range(window):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))

            for i in range(1, d_model, 2):
                pe[pos, i] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(self.pe.shape)
        return x * torch.sqrt(self.d_model) + self.pe[:, :x.size(1)]

class ResidualBlock(nn.Module):
    def __init__(self, d_model, d_inner, d_state, d_conv, dt_rank, conv_bias, bias):
        super().__init__()
        self.mixer = MambaBlock(d_model, d_inner, d_state, d_conv, dt_rank, conv_bias, bias)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        # Apply normalization and MambaBlock
        # then make a residual connection with the input x
        output = self.mixer(self.norm(x)) + x

        return output


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_inner, d_state, d_conv, dt_rank, conv_bias, bias):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        # Create a repeating sequence for initializing the matrix A of the state-space model
        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_inner)
        # Save the logarithm of matrix A as a trainable parameter
        self.A_log = nn.Parameter(torch.log(A))
        # Initialize the matrix D to all 1 trainable parameters
        self.D = nn.Parameter(torch.ones(d_inner))
        # Output linear transformation layer
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)

    def forward(self, x):
        # Get the dimension of input x
        # batchsize, seq_len, dim
        (b, l, d) = x.shape
        # Apply a linear transformation of the input
        # shape (b, l, 2 * d_in)
        x_and_res = self.in_proj(x)
        # The transformed output is divided into two parts x and res
        # The obtained x is divided into two parts
        # one part x is used for subsequent transformation to generate the required parameters
        # and res is used for the residual part
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        # Get x_back for calculating the backward
        x_back = x.flip([2])
        # Adjust the shape of x
        x = rearrange(x, 'b l d_in -> b d_in l')
        x_back = rearrange(x_back, 'b l d_in -> b d_in l')
        # Apply depthwise convolution and then take the first l outputs
        x = self.conv1d(x)[:, :, :l]
        x_back = self.conv1d(x_back)[:, :, :l]
        # Reshape x again
        x = rearrange(x, 'b d_in l -> b l d_in')
        x_back = rearrange(x_back, 'b d_in l -> b l d_in')
        # Apply SiLU activation function
        x = F.silu(x)
        x_back = F.silu(x_back)
        # Running the State Space Model
        y = self.ssm(x)
        y_back = self.ssm(x_back)
        y_back = y_back.flip([2])
        # Multiply the SiLU activation result of res by y
        y = y * F.silu(res)
        y_back = y_back * F.silu(res)
        y_all = y + y_back
        # Apply output linear transformation
        output = self.out_proj(y_all)

        return output

    def ssm(self, x):
        #  Get the dimension of A_log
        #  A is assigned the following values ​​during initialization:
        #  A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        #  self.A_log = nn.Parameter(torch.log(A))
        # （args.d_inner, args.d_state）
        (d_in, n) = self.A_log.shape

        # Calculate ∆ A B C D, these are state space parameters
        # Calculate the matrix A
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        # Take the value of D
        D = self.D.float()
        # Apply the projective transformation of x
        # ( b,l,d_in) -> (b, l, dt_rank + 2*n)
        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        # Split delta, B, C
        # delta: (b, l, dt_rank). B, C: (b, l, n)
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        # Apply dt_proj and calculate delta
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        # Apply the selective scanning algorithm
        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        # Get the dimension of input u
        (b, l, d_in) = u.shape
        # Get the number of columns of matrix A
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH)
        # - B is using a simplified Euler discretization
        # Calculate the discretized A
        # Dot-product delta and A, broadcast A along the last dimension of delta
        # and then perform element-wise multiplication
        # A:(d_in, n),delta:(b, l, d_in)
        # A broadcast extension->(b,l,d_in,n)
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        # delta, B, and u
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # Perform selective scan
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        # Initialize the output list ys
        ys = []
        for i in range(l):
            # Update status x
            # deltaA:((b,l,d_in, n)
            # deltaB_u:( b,l,d_in,n)
            # x:
            x = deltaA[:, i] * x + deltaB_u[:, i]
            # Calculate the output y
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            # Add the output y to the list ys
            ys.append(y)
        # Stack the list ys into a tensor y
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        # Multiply the input u by D and add to the output y
        y = y + u * D

        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Calculate the normalized output
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output
