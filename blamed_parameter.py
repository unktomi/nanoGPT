import torch
import torch.nn as nn

class BlamedParameter:
    """Wrapper around nn.Parameter that adds blame tracking"""
    
    def __init__(self, parameter):
        self.param = parameter if isinstance(parameter, nn.Parameter) else nn.Parameter(parameter)
        self.activation = torch.zeros_like(self.param.data)
        self.blame = torch.zeros_like(self.param.data)
        self.decay = 0.99
        
    def update(self, current_activation):
        # Update activation (reward)
        self.activation = self.decay * self.activation + (1 - self.decay) * current_activation.abs()
        
        # Update blame (penalty) if we have gradients
        if self.param.grad is not None:
            self.blame = self.decay * self.blame + (1 - self.decay) * self.param.grad.abs()

    def get_update_weight(self):
        # Simple reward/penalty ratio 
        return self.blame / (self.activation + 1e-8)
        
    @property
    def data(self):
        return self.param.data
        
    @property
    def grad(self):
        return self.param.grad