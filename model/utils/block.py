import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils.attention import MultiHeadAttention
from model.utils.activation import activations
from typing import Optional

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, activation: str = 'relu', dropout_p: float = 0.) -> None:
        super().__init__()
        self.dropout_p = dropout_p

        # Main Layers
        self.attention = MultiHeadAttention(d_model, n_heads, dropout_p)
        self.ffn = FFN(d_model, activation)

        # Norm Layers
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Sub-layer 1
        attention = self.attention(x, x, x, mask)
        attention = self.norm_1(F.dropout(attention + x, p=self.dropout_p, training=self.training))

        # Sub-layer 2
        ffn_out = self.ffn(attention)
        ffn_out = self.norm_2(F.dropout(ffn_out + attention, p=self.dropout_p, training=self.training))

        return ffn_out
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, activation: str = 'relu', dropout_p: float = 0.) -> None:
        super().__init__()
        self.dropout_p = dropout_p

        # Main Layers
        self.local_attention = MultiHeadAttention(d_model, n_heads, dropout_p)
        self.global_attention = MultiHeadAttention(d_model, n_heads, dropout_p)
        self.ffn = FFN(d_model, activation)

        # Norm
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, features: torch.Tensor, look_ahead_mask: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Sub - layer 1
        local_attention = self.local_attention(x, x, x, look_ahead_mask)
        local_attention = self.norm_1(F.dropout(local_attention + x, p=self.dropout_p, training=self.training))

        # Sub - layer 2
        global_attention = self.global_attention(local_attention, features, features, padding_mask)
        global_attention = self.norm_2(F.dropout(global_attention + local_attention, p=self.dropout_p, training=self.training))

        # Sub - layer 3
        ffn_out = self.ffn(global_attention)
        ffn_out = self.norm_3(F.dropout(ffn_out + global_attention, p=self.dropout_p, training=self.training))

        return ffn_out

class FFN(nn.Module):
    def __init__(self, d_model: int, activation: str = 'relu', n_expands: int = 4) -> None:
        super().__init__()
        activation = activation.lower()
        assert activation in activations.keys()
        hidden_dim = d_model * n_expands

        self.hidden_layer = nn.Linear(d_model, hidden_dim)
        self.activation = activations[activation]
        self.final_layer = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.final_layer(x)
        return x