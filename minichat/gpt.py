import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from functools import partial


@dataclass
class GPTConfig:
    sequence_len: int = 512
    n_layers: int = 12
    vocab_size: int = 65_536
    emb_dim: int = 128
    n_heads: int = 8
    n_kv_heads: int = 8

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_pos_emb(x, sin, cos):
    assert x.dim() == 4  # B, H, T, D
    B, H, T, D = x.shape
    d = D // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos
    out = torch.cat([y1, y2], dim=-1)
    out = out.to(x.dtype)
    return out

# (softmax(qT * k)/sqrt(d_k))*v

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.emb_dim // config.n_heads
        assert self.head_dim * config.n_heads == config.emb_dim, "emb_dim must be divisible by n_heads"
        assert self.n_kv_heads <= self.n_heads and self.n_heads % self.n_kv_heads == 0, "n_kv_heads must be less than or equal to n_heads and n_heads must be divisible by n_kv_heads"
        self.c_q = nn.Linear(config.emb_dim, self.n_heads * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.emb_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.emb_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.emb_dim, config.emb_dim, bias=False)
    
    def forward(self, x, cos_sin, kv_cache=None):
        B,T,C = x.shape
        q = self.c_q(x).view(B, T, self.n_heads, self.head_dim) # B, T, H, D
        k = self.c_k(x).view(B, T, self.n_kv_heads, self.head_dim) # B, T, H, D
        v = self.c_v(x).view(B, T, self.n_kv_heads, self.head_dim) # B, T, H, D

        # Appy rotary embeddings
        cos, sin = cos_sin
        q, k = apply_rotary_pos_emb(q, sin, cos), apply_rotary_pos_emb(k, sin, cos)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1,2), v.transpose(1,2)

        # Apply KV cache if provided
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        
        Tq, Tk = q.size(2), k.size(2)

        enable_gqa = self.n_heads != self.n_kv_heads
        if kv_cache is None or Tq == Tk:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa) # B, H, T, D
        elif Tq ==1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa) # B, H, 1, D
        else:
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False, enable_gqa=enable_gqa) # B, H, T, D
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
    
    def forward(self, x, cos_sin, kv_cache=None):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.emb_dim),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layers)])
        })
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.emb_dim // config.n_heads
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, device='cuda')
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

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
        torch.nn.init.zeros_(self.lm_head.weight)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        head_dim = self.config.emb_dim // self.config.n_heads
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, device='cuda')
        self.cos.copy_(cos)
        self.sin.copy_(sin)
        self.transformer.wte.to(dtype=torch.bfloat16)
    
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device='cuda'):
        channel_range = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
        cos, sin = freqs.cos(), freqs.sin()  # (seq_len, head_dim/2)
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # (1, seq_len, 1, head_dim/2)
        return cos, sin
    
    def get_device(self):
        return next(self.parameters()).device
    
    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.emb_dim
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        unembedding_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(unembedding_params), "Parameter count mismatch in optimizer setup"
        dmodel_lr_scale = model_dim/768 ** -0.5
        adam_groups = [
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=unembedding_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=matrix_params, lr=matrix_lr * dmodel_lr_scale)
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = partial(torch.optim.AdamW, fused=True)
        optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']
        return optimizer

    
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        assert T <= self.config.sequence_len, f"Cannot forward sequence of length {T}, max is {self.config.sequence_len}"
        assert idx.device == self.cos.device, f"Rotary embeddings are on {self.cos.device}, but input is on {idx.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"

        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]  # (1, T, 1, head_dim/2)

        x = self.transformer.wte(idx)  # token embeddings of shape (B, T, emb_dim)
        
        # Apply initial norm
        x = norm(x)
        
        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        
        # Final norm
        x = norm(x)
        softcap = 15
        if targets is not None:
            # Training mode: compute loss
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)
            logits = logits.float()              
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # Inference mode: return logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)
            logits = logits.float()              
            return logits
    
    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list), "Input tokens should be a list of token IDs"
        device = self.device()
        rng = None
        if temperature > 0.0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        
        for _ in range(max_tokens):
            # Crop ids to the last sequence_len tokens if it gets too long
            ids_cond = ids if ids.size(1) <= self.config.sequence_len else ids[:, -self.config.sequence_len:]
            
            logits = self.forward(ids_cond)  # B, T, vocab_size
            logits = logits[:, -1, :]  # Take the last time step
            if top_k is not None:
                topk_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_values[:, [-1]]] = -float('Inf')
            if temperature > 0.0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
    
    def estimate_flops_per_token(self):
        # Rough estimate of FLOPs per token during training
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layers, self.config.n_heads, self.config.emb_dim // self.config.n_heads, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token
                