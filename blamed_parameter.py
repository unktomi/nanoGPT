import torch
import torch.nn as nn

class BlamedParameter(nn.Parameter):
    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad=requires_grad)
        self.blame = torch.zeros_like(data)
        self.blame_decay = 0.99
        self._step = 0
        
    @classmethod
    def from_param(cls, param):
        if isinstance(param, BlamedParameter):
            return param
        return cls(param.data, param.requires_grad)
        
    def update_blame(self, warmup_steps=None):
        if self.grad is None:
            return
            
        grad_mag = self.grad.abs()
        if warmup_steps and self._step < warmup_steps:
            warmup_factor = self._step / warmup_steps
            grad_mag = grad_mag * warmup_factor
            
        self.blame = self.blame_decay * self.blame + (1 - self.blame_decay) * grad_mag
        self._step += 1
    
    def get_effective_blame(self, warmup_steps=None):
        if warmup_steps and self._step < warmup_steps:
            warmup_factor = self._step / warmup_steps
            return warmup_factor * self.blame + (1 - warmup_factor) * torch.ones_like(self.blame)
        return self.blame
        
    @property
    def low_blame_mask(self, threshold=1e-6):
        return self.blame < threshold
