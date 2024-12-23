import torch
from torch.optim import Optimizer

class BlameOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3):
        super().__init__([p for p in params if p.requires_grad], {'lr': lr})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Get reward/penalty weighting
                update_weight = p.get_update_weight()
                
                # Apply weighted update
                p.data.add_(p.grad, alpha=-group['lr'] * update_weight)