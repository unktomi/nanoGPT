import torch
import torch.nn as nn

class BlamedParameter(torch.nn.Parameter):
    def __new__(cls, data, requires_grad=True):
        # Create the Parameter properly
        instance = super().__new__(cls, data, requires_grad=requires_grad)
        
        # Add our tracking tensors
        instance.activation = torch.zeros_like(data)
        instance.blame = torch.zeros_like(data)
        instance.decay = 0.99
        
        return instance

    def update(self, current_activation):
        # Update activation (reward)
        self.activation = self.decay * self.activation + (1 - self.decay) * current_activation.abs()
        
        # Update blame (penalty) if we have gradients
        if self.grad is not None:
            self.blame = self.decay * self.blame + (1 - self.decay) * self.grad.abs()

    def get_update_weight(self):
        # Simple reward/penalty ratio 
        return self.blame / (self.activation + 1e-8)