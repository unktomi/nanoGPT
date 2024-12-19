import math
import torch
from torch.optim import Optimizer
from blamed_parameter import BlamedParameter

class BlameAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, warmup_steps=None, blame_decay=0.99):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps,
                      weight_decay=weight_decay, warmup_steps=warmup_steps)
        
        # Convert parameters to BlamedParameters if they aren't already
        params = list(params)
        for idx, p in enumerate(params):
            if not isinstance(p, BlamedParameter):
                params[idx] = BlamedParameter.from_param(p)
                
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                params_with_grad.append(p)
                grads.append(p.grad)
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                state['step'] += 1
                state_steps.append(state['step'])
                
                # Update blame tracking
                p.update_blame(group['warmup_steps'])
                
            # Perform optimization step
            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step_t = state_steps[i]
                
                # Get effective blame for this parameter
                effective_blame = param.get_effective_blame(group['warmup_steps'])
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                step_size = group['lr']
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step_t
                bias_correction2 = 1 - beta2 ** step_t
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Weight decay
                if group['weight_decay'] != 0:
                    param.data.mul_(1 - group['lr'] * group['weight_decay'])
                    
                # Update with blame weighting
                param.addcdiv_(exp_avg, denom, value=-step_size * effective_blame)
                
        return loss
