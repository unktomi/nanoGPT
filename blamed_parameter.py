"""
Implementation of BlamedParameter wrapper for tracking gradient magnitudes in neural network training.
"""
import torch
import torch.nn as nn

class BlamedParameter(nn.Parameter):
    """
    A wrapper around nn.Parameter that tracks gradient magnitudes ("blame") during training.
    
    The blame score represents how much each parameter influences the loss,
    based on a running average of gradient magnitudes. Parameters with consistently
    high blame scores receive proportionally larger updates.
    """
    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad=requires_grad)
        self.blame = torch.zeros_like(data)
        self.blame_decay = 0.99
        self._step = 0
        
    @classmethod
    def from_param(cls, param):
        """Create a BlamedParameter from an existing Parameter."""
        if isinstance(param, BlamedParameter):
            return param
        return cls(param.data, param.requires_grad)
        
    def update_blame(self, warmup_steps=None):
        """Update blame based on current gradient magnitude."""
        if self.grad is None:
            return
        
        # Update running average of gradient magnitudes
        grad_mag = self.grad.abs()
        if warmup_steps and self._step < warmup_steps:
            # During warmup, gradually increase blame weighting
            warmup_factor = self._step / warmup_steps
            grad_mag = grad_mag * warmup_factor
            
        self.blame = self.blame_decay * self.blame + (1 - self.blame_decay) * grad_mag
        self._step += 1
    
    def get_effective_blame(self, warmup_steps=None):
        """Get the effective blame for parameter updates."""
        if warmup_steps and self._step < warmup_steps:
            # During warmup, blend between uniform weighting and blame-based weighting
            warmup_factor = self._step / warmup_steps
            return warmup_factor * self.blame + (1 - warmup_factor) * torch.ones_like(self.blame)
        return self.blame
        
    @property
    def low_blame_mask(self, threshold=1e-6):
        """Return a boolean mask identifying consistently low-blame parameters."""
        return self.blame < threshold
