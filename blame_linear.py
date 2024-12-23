import torch
import torch.nn as nn
from torch.nn import functional as F
from blamed_parameter import BlamedParameter

class BlameLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        # Wrap parameters with blame tracking
        self.blamed_weight = BlamedParameter(self.weight)
        if bias:
            self.blamed_bias = BlamedParameter(self.bias)

    def forward(self, x):
        # Track activations during forward pass
        self.blamed_weight.update(x)
        out = F.linear(x, self.weight)
        if self.bias is not None:
            self.blamed_bias.update(torch.ones_like(out))
            out = out + self.bias
        return out