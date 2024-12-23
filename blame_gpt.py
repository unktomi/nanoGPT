import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from blame_mlp import BlameMLP
from blamed_parameter import BlamedParameter
from model import LayerNorm, CausalSelfAttention

class BlameBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = BlameMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class BlameGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([BlameBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def get_blamed_parameters(self):
        blamed_params = []
        for module in self.modules():
            if hasattr(module, 'blamed_weight'):
                blamed_params.append(module.blamed_weight)
            if hasattr(module, 'blamed_bias') and module.blamed_bias is not None:
                blamed_params.append(module.blamed_bias)
        return blamed_params

    def get_regular_parameters(self):
        # Get all parameters that aren't handled by blamed parameters
        blamed_modules = set(id(m) for m in self.modules() if hasattr(m, 'blamed_weight'))
        regular_params = []
        for name, param in self.named_parameters():
            if not any(id(m) == id(param) for m in blamed_modules):
                regular_params.append(param)
        return regular_params

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Get blamed and regular parameters
        blamed_params = self.get_blamed_parameters()
        regular_params = self.get_regular_parameters()

        print(f'Number of blamed parameters: {len(blamed_params)}')
        print(f'Number of regular parameters: {len(regular_params)}')

        from blame_optimizer import BlameOptimizer
        optimizers = [
            BlameOptimizer(blamed_params, lr=learning_rate),
            torch.optim.AdamW(regular_params, lr=learning_rate, betas=betas)
        ]

        return optimizers