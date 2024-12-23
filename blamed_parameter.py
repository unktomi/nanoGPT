import torch
import torch.nn as nn

class BlamedParameter(nn.Parameter):
    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad=requires_grad)
        self.activation = torch.zeros_like(data)  # reward
        self.blame = torch.zeros_like(data)       # penalty
        self.decay = 0.99

    def update(self, current_activation):
        # Update activation (reward)
        self.activation = self.decay * self.activation + (1 - self.decay) * current_activation.abs()
        
        # Update blame (penalty) if we have gradients
        if self.grad is not None:
            self.blame = self.decay * self.blame + (1 - self.decay) * self.grad.abs()

    def get_update_weight(self):
        # Simple reward/penalty ratio 
        return self.blame / (self.activation + 1e-8)