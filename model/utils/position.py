import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

        self.angles = nn.Parameter(
            1.0 / 10000.0 ** (torch.arange(0, d_model, 2) / d_model), 
            requires_grad=False
        ).unsqueeze(0) # (1, d_model/2)

    def forward(self, length: int) -> torch.Tensor:
        pe = torch.zeros((length, self.d_model), dtype=self.angles.dtype, device=self.angles.device)
        pos = torch.arange(length, dtype=self.angles.dtype, device=self.angles.device).unsqueeze(1) # (length, 1)

        pos_angles = torch.matmul(pos, self.angles) # (length, d_model/2)

        pe[:, 0::2] = torch.sin(pos_angles)
        pe[:, 1::2] = torch.cos(pos_angles)

        return pe.unsqueeze(0) # Open for Batch ==> [1, length, d_model]