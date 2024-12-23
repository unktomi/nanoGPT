import torch
import torch.nn as nn
from blame_linear import BlameLinear

class BlameMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = BlameLinear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)    # This layer now uses blame tracking
        x = self.gelu(x)
        x = self.c_proj(x)  # Keep this as regular Linear for now
        x = self.dropout(x)
        return x