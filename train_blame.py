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

# First, determine device
if torch.backends.mps.is_available():
    device = 'mps'
    print("Using MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = 'cuda'
    print("Using CUDA")
else:
    device = 'cpu'
    print("Using CPU")

# Smaller test configuration for initial runs
model_args = dict(
    n_layer=4,      # Start with fewer layers
    n_head=4,       # Fewer attention heads
    n_embd=128,     # Smaller embedding dimension
    block_size=64,  # Smaller context window
    bias=False,
    vocab_size=50304,
    dropout=0.0
)

# Initialize model and move to device
model = BlameGPT(GPTConfig(**model_args))
model.to(device)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Training hyperparameters
batch_size = 8
max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
weight_decay = 0.1

# Get optimizers
optimizers = model.configure_optimizers(
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    betas=(0.9, 0.95),
    device_type=device
)

def get_batch(split):
    # Download the data first if needed using prepare.py
    data_dir = os.path.join('data', 'openwebtext')
    
    if not os.path.exists(data_dir):
        print("Data not found. Please run prepare.py first!")
        exit(1)
        
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - model_args['block_size'], (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+model_args['block_size']]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+model_args['block_size']]).astype(np.int64)) for i in ix])
    
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    eval_iters = 20  # Reduced for testing
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_blame_stats(model):
    stats = {}
    for name, module in model.named_modules():
        if hasattr(module, 'blamed_weight'):
            w = module.blamed_weight
            stats[f"{name}_act_mean"] = w.activation.mean().item()
            stats[f"{name}_blame_mean"] = w.blame.mean().item()
            stats[f"{name}_update_weight"] = w.get_update_weight().mean().item()
    return stats

# Training loop
print("Starting training...")
X, Y = get_batch('train')

for iter in range(max_iters):
    # Every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        blame_stats = get_blame_stats(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print("Blame stats:")
        for k, v in blame_stats.items():
            print(f"  {k}: {v:.4f}")

    # Forward pass
    logits, loss = model(X, Y)
    
    # Backward pass
    for opt in optimizers:
        opt.zero_grad(True)
    loss.backward()
    
    # Update with both optimizers
    for opt in optimizers:
        opt.step()
    
    # Get new batch
    X, Y = get_batch('train')

print('Training finished')