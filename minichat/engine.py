import torch
import torch.nn.functional as F

class KVCache:
    def __init__(self, num_layers, num_heads, head_dim, batch_size, seq_len=1024, growth_size=1024):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.growth_size = growth_size
        self.seq_len = seq_len
        self.pos = 0
        self.kv_cache = None
        self.kv_shape = (self.num_layers, 2, self.batch_size, self.num_heads, self.seq_len, self.head_dim)
    
    def get_pos(self):
        return self.pos

    def reset(self):
        self.pos = 0

    def insert_kv(self, layer_idx, key, value):
        if self.kv_cache is None:
            kv_shape = (self.num_layers, 2, self.batch_size, self.num_heads, self.seq_len, self.head_dim)
            self.kv_cache = torch.empty(kv_shape, dtype=key.dtype, device=key.device)

        B, H, T_add, D = key.size()
        t0, t1 = self.pos, self.pos + T_add

        if t1 > self.seq_len:
            t_needed = t1 + self.growth_size
            t_needed = t_needed + (t_needed + 1023) & ~1023  # Align to 1024
            append_shape = (self.num_layers, 2, self.batch_size, self.num_heads, t_needed - self.seq_len, self.head_dim)
            append_cache = torch.empty(append_shape, dtype=key.dtype, device=key.device)
            self.kv_cache = torch.cat([self.kv_cache, append_cache], dim=4)
            self.seq_len = self.kv_cache.shape[4]

        self.kv_cache[layer_idx, 0, :, :, t0:t1, :] = key
        self.kv_cache[layer_idx, 1, :, :, t0:t1, :] = value

        key_view = self.kv_cache[layer_idx, 0, :, :, :t1, :]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1, :]

        if layer_idx == self.num_layers-1:
            self.pos = t1

        return key_view, value_view

    def prefill(self, other):
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill with a None KV cache"
        
        other_kv_cache = other.kv_cache
        other_pos = other.pos

        other_num_layers, other_kv, other_batch_size, other_num_heads, other_seq_len, other_head_dim = other_kv_cache.shape

        assert other_num_layers == self.num_layers, "Number of layers must match"
        assert other_num_heads == self.num_heads, "Number of heads must match"
        assert other_head_dim == self.head_dim, "Head dimension must match"
        assert other_batch_size == 1 or other_batch_size == self.batch_size, "Other batch size must be 1 or equal to current batch size"
        assert self.seq_len >= other_seq_len, "Other sequence length must be less than or equal to current sequence length"

        self.kv_cache = torch.empty(self.kv_shape, dtype=other.kv_cache.dtype, device=other.kv_cache.device)

        self.kv_cache[:, :, :, :, :other_pos, :] = other_kv_cache
        self.pos = other_pos

@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    assert temperature >= 0.0, "Temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        values, ids = torch.topk(logits, top_k)
        values = values / temperature
        probs = F.softmax(values, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return ids.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

class RowState:
    def __init__(self):
        self.completed = False

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    @torch.inference_mode()
    def generate(self, tokens, max_tokens=None, num_samples=1, temperature=1.0, top_k=None, seed=42):
        device = self.model.get_device()
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            num_layers=self.model.config.n_layers,
            num_heads=self.model.config.n_heads,
            head_dim=self.model.config.emb_dim // self.model.config.n_heads,
        )

        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # Add batch dimension

        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :]
        next_ids = sample_next_token(logits, torch.Generator(device=device), temperature, top_k)

        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len

        # Replicate KV cache for batch
        batch_size = num_samples
        kv_cache_decode = KVCache(
            batch_size=batch_size,
            seq_len=kv_length_hint,
            num_layers=self.model.config.n_layers,
            num_heads=self.model.config.n_heads,
            head_dim=self.model.config.emb_dim // self.model.config.n_heads,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill

        row_states = [RowState() for _ in range(batch_size)]

        num_generated = 0
        first_iteration = True
        rng = torch.Generator(device=device).manual_seed(seed)

        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            
            if all(row_state.completed for row_state in row_states):
                break

            if first_iteration:
                # Use the next_ids that were already sampled from the prefill forward pass
                sampled_tokens = [next_ids[0, 0].item()] * batch_size
                first_iteration = False
            else:
                logits = self.model.forward(ids, kv_cache=kv_cache_decode)
                logits = logits[:, -1, :]
                next_ids = sample_next_token(logits, rng, temperature, top_k)
                sampled_tokens = next_ids[:, 0].tolist()
            
            token_column = []
            for i, token in enumerate(sampled_tokens):
                if token == self.tokenizer.get_bos_token_id():
                    row_states[i].completed = True
                token_column.append(token)
                    
            yield token_column
            num_generated += 1
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
        
    
    def generate_batch(self, tokens, max_tokens=None, num_samples=1,  temperature=1.0, top_k=None):
        results = [tokens.copy() for _ in range(num_samples)]
        completed = [False for _ in range(num_samples)]
        for token_column in self.generate(
            tokens,
            max_tokens=max_tokens,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
        ):
            for i, token in enumerate(token_column):
                if not completed[i]:
                    if token == self.tokenizer.get_bos_token_id():
                        completed[i] = True
                    else:
                        results[i].append(token)
            
            if all(completed):
                break
        return results

        
            