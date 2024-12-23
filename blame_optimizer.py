import torch
from torch.optim import Optimizer

class BlameOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3):
        # Extract actual parameters from blame wrappers
        blame_params = [(p.param if hasattr(p, 'param') else p) for p in params]
        super().__init__(blame_params, {'lr': lr})
        self.blame_wrappers = [p for p in params if hasattr(p, 'param')]

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p, blame_wrapper in zip(group['params'], self.blame_wrappers):
                if p.grad is None:
                    continue
                    
                # Get reward/penalty weighting
                update_weight = blame_wrapper.get_update_weight()
                
                # Apply weighted update
                p.data.add_(p.grad, alpha=-group['lr'] * update_weight)