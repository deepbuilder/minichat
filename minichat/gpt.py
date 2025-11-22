import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    sequence_len: int = 512
    n_layers: int = 12
    vocab_size: int = 65_536
    emb_dim: int = 128
    n_heads: int = 8

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

# (softmax(qT * k)/sqrt(d_k))*v

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.head_dim = config.emb_dim // config.n_heads
        self.c_q = nn.Linear(config.emb_dim, self.n_heads * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.emb_dim, self.n_heads * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.emb_dim, self.n_heads * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.emb_dim, config.emb_dim, bias=False)
    
    def forward(self, x):
        B,T,C = x.shape
        q = self.c_q(x).view(B, T, self.n_heads, self.head_dim) # B, T, H, D
        k = self.c_k(x).view(B, T, self.n_heads, self.head_dim) # B, T, H, D
        v = self.c_v(x).view(B, T, self.n_heads, self.head_dim) # B, T, H, D
        q, k, v = q.transpose(1, 2), k.transpose(1,2), v.transpose(1,2)
        q, k, v = norm(q), norm(k), norm(v)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # B, H, T, D
        y = y.transpose(1,2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.emb_dim, 4*config.emb_dim)
        self.c_proj = nn.Linear(4*config.emb_dim, config.emb_dim)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.emb_dim),
            "wpe": nn.Embedding(config.sequence_len, config.emb_dim),  # Added positional embeddings
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layers)])
        })
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        
        # Initialize weights
        self.init_weights()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def init_weights(self):
        self.apply(self._init_weights)
        # Apply special scaling to residual projections (GPT-2 style)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
    
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()
        assert T <= self.config.sequence_len, f"Cannot forward sequence of length {T}, max is {self.config.sequence_len}"
        
        # Token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, emb_dim)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, emb_dim)
        x = tok_emb + pos_emb  # (B, T, emb_dim)
        
        # Apply initial norm
        x = norm(x)
        
        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # Final norm
        x = norm(x)
        
        if targets is not None:
            # Training mode: compute loss
            logits = self.lm_head(x)
            logits = logits.float()               
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # Inference mode: return logits
            logits = self.lm_head(x)
            return logits
    
    def generate(self, tokens, max_tokens):
        device = self.device()
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Crop ids to the last sequence_len tokens if it gets too long
                ids_cond = ids if ids.size(1) <= self.config.sequence_len else ids[:, -self.config.sequence_len:]
                
                logits = self.forward(ids_cond)  # B, T, vocab_size
                logits = logits[:, -1, :]  # Take the last time step
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1)
                ids = torch.cat((ids, next_ids), dim=1)
                token = next_ids.item()
                yield token
    
    def estimate_flops_per_token(self):
        # Rough estimate of FLOPs per token during training
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel() + self.transformer.wpe.weight.numel()
        l, h, q, t = self.config.n_layers, self.config.n_heads, self.config.emb_dim // self.config.n_heads, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token
                