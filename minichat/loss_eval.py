
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
    total_nats = torch.zeros_((), dtype=torch.float64, device=model.device())
    total_bytes = torch.zeros_((), dtype=torch.int32, device=model.device())

    batch_iter = iter(batches)

    for _ in range(steps):
        batch, y = next(batch_iter)
        loss = model(batch, y, loss_reduction=None)
        loss = loss.view(-1) # flatten
        y = y.view(-1)
        if (y<0).any():
            valid = (y>=0)
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            num_bytes = torch.where(valid, token_bytes[y_safe], torch.zeros_like(y))
            total_nats += (loss * (num_bytes > 0)).sum()
            total_bytes += num_bytes.sum()
        else:
            num_bytes = token_bytes[y]
            loss = (loss *(num_bytes>0)).sum()
        
        bpb = total_nats/(math.log(2) * total_bytes)
        return bpb
