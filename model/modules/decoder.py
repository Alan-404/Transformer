import torch
import torch.nn as nn
import math
from model.utils.block import DecoderBlock
from model.utils.position import PositionalEncoding

from typing import Optional

class Decoder(nn.Module):
    def __init__(self, n_tokens: int, n_layers: int, d_model: int, n_heads: int, activation: str = 'relu', dropout_p: float = 0.) -> None:
        super().__init__()
        self.sqrt_dim = math.sqrt(d_model)

        self.embedding = nn.Embedding(n_tokens, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderBlock(d_model, n_heads, activation, dropout_p) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, n_tokens)

        # Init Weights and Bias
        self.embedding.weight.data.uniform_(-1, 1)
        
    def forward(self, x: torch.Tensor, features: torch.Tensor,  look_ahead_mask: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None)-> torch.Tensor:
        x = self.embedding(x) / self.sqrt_dim
        x = x + self.pe(x.size(1))
        for layer in self.layers:
            x = layer(x, features, look_ahead_mask, padding_mask)

        x = self.linear(x)
        return x