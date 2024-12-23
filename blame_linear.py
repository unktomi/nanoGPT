import torch
import torch.nn as nn
from torch.nn import functional as F
from blamed_parameter import BlamedParameter

class BlameLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        # Replace standard parameters with blamed ones
        self.weight = BlamedParameter(self.weight.data)
        if bias:
            self.bias = BlamedParameter(self.bias.data)

    def forward(self, x):
        # Track activations during forward pass
        self.weight.update(x)
        out = F.linear(x, self.weight)
        if self.bias is not None:
            self.bias.update(torch.ones_like(out))
            out = out + self.bias
        return out