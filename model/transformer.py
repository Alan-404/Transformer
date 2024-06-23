import torch
import torch.nn as nn

from model.modules.encoder import Encoder
from model.modules.decoder import Decoder

from model.utils.masking import generate_look_ahead_mask, generate_padding_mask

from typing import Optional

class Transformer(nn.Module):
    def __init__(self, 
                 n_tokens: int, 
                 n_layers: int = 6, 
                 d_model: int = 512, 
                 n_heads: int = 8, 
                 activation: str = 'relu', 
                 dropout_p: float = 0.) -> None:
        super().__init__()
        self.encoder = Encoder(n_tokens, n_layers, d_model, n_heads, activation, dropout_p)
        self.decoder = Decoder(n_tokens, n_layers, d_model, n_heads, activation, dropout_p)

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_lengths: Optional[torch.Tensor] = None, y_lengths: Optional[torch.Tensor] = None):
        batch_size, x_ctx = x.size()
        y_ctx = y.size(1)

        if x_lengths is not None:
            padding_mask = generate_padding_mask(x_lengths).unsqueeze(1).unsqueeze(1)
        else:
            padding_mask = torch.ones((batch_size, 1, 1, x_ctx), dtype=torch.bool, device=x.device)
        
        if y_lengths is not None:
            look_ahead_mask = generate_look_ahead_mask(y_lengths).unsqueeze(1)
        else:
            look_ahead_mask = torch.tril(torch.ones((y_ctx, y_ctx), dtype=torch.bool, device=y.device)).unsqueeze(0).repeat((batch_size, 1, 1, 1))

        x = self.encoder(x, padding_mask)
        y = self.decoder(y, x, look_ahead_mask, padding_mask)

        return y