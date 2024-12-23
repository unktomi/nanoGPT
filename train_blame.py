import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig
from blame_gpt import BlameGPT
from blamed_parameter import BlamedParameter

def get_blame_stats(model):
    stats = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module.weight, BlamedParameter):
            w = module.weight
            stats[f"{name}_act_mean"] = w.activation.mean().item()
            stats[f"{name}_blame_mean"] = w.blame.mean().item()
            stats[f"{name}_update_weight"] = w.get_update_weight().mean().item()
            stats[f"{name}_act_max"] = w.activation.max().item()
            stats[f"{name}_blame_max"] = w.blame.max().item()
    return stats

# Load the config
out_dir = 'out-blame'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# Initialize wandb if available
try:
    import wandb
    wandb_log = True
    wandb_project = 'blame-gpt'
    wandb_run_name = 'blame-test'
except ImportError:
    wandb_log = False

# Load model
model_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024,
                bias=False, vocab_size=50304, dropout=0.0)

model = BlameGPT(GPTConfig(**model_args))
model.to('cuda')

# Optimizer
optimizers = model.configure_optimizers(
    weight_decay=1e-1,
    learning_rate=6e-4,
    betas=(0.9, 0.95),
    device_type='cuda'
)

# Training loop
iter_num = 0
best_val_loss = float('inf')

while True:
    # Determine learning rate
    lr = 6e-4  # Fixed for simplicity in test
    for opt in optimizers:
        for param_group in opt.param_groups:
            param_group['lr'] = lr
    
    # Evaluate loss and collect blame stats
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        blame_stats = get_blame_stats(model)
        
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print("Blame tracking stats:")
        for k, v in blame_stats.items():
            print(f"  {k}: {v:.4f}")
            
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                **blame_stats
            })
        
        # Save checkpoint
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizers': [opt.state_dict() for opt in optimizers],
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    # Forward backward update
    for micro_step in range(5):  # Simplified accumulation steps
        with torch.cuda.amp.autocast():
            logits, loss = model(X, Y)
            loss = loss / 5  # gradient accumulation
        
        X, Y = get_batch('train')  # get next batch
        loss.backward()
    
    # Clip gradients and step optimizers
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for opt in optimizers:
        opt.step()
        opt.zero_grad()
    
    # Increment and check for completion
    iter_num += 1
    if iter_num > 600000:  # max iters
        break