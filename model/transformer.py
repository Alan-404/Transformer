import torch
import torch.nn as nn

from model.modules.encoder import Encoder
from model.modules.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, n_tokens: int, n_layers: int, d_model: int, n_heads: int, activation: str = 'relu', dropout_p: float = 0.) -> None:
        super().__init__()
        self.encoder = Encoder(n_tokens, n_layers, d_model, n_heads, activation, dropout_p)
        self.decoder = Decoder(n_tokens, n_layers, d_model, n_heads, activation, dropout_p)

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_lengths: torch.Tensor, y_lengths: torch.Tensor):
        pass