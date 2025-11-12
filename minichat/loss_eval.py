
import math
import torch

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    '''
    bits per byte for tokenization-vocab-size independent metric for comparison.

    1. Compute loss
    2. Calculate the total nats:
        total_nats = sum(loss for valid tokens) Note: Special tokens such as <bos> are excluded (they carry y as -1)
    3. Calculate total bytes
        total_bytes = sum(token_bytes for valid tokens)
    4. bpb = total_nats / log(2) * total_bytes

    '''
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.device())

    batch_iter = iter(batches)
    token_bytes = token_bytes.to(device=model.device())

    for _ in range(steps):
        x, y = next(batch_iter)
        loss = model(x, y, loss_reduction='none')
        loss = loss.view(-1) # flatten
        y = y.view(-1)
        if (y.int()<0).any():
            valid = (y>=0)
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            num_bytes = torch.where(valid, token_bytes[y_safe], torch.zeros_like(y, dtype=token_bytes.dtype))
            total_nats += (loss * (num_bytes > 0)).sum()
            total_bytes += num_bytes.sum()
        else:
            num_bytes = token_bytes[y]
            total_nats += (loss *(num_bytes>0)).sum()
            total_bytes += num_bytes.sum()
        
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
