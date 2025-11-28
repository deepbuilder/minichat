
import math
import torch
import torch.distributed as dist

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    '''
    bits per byte for tokenization-vocab-size independent metric for comparison.

    1. Compute loss
    2. Calculate the total nats:
        total_nats = sum(loss for valid tokens) Note: Special tokens such as <bos> are excluded (they carry y as -1)
    3. Calculate total bytes
        total_bytes = sum(token_bytes for valid tokens)
    4. bpb = total_nats / log(2) / total_bytes

    '''
    model.eval()  # Ensure model is in eval mode
    
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())  # Use float64 for precision
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())

    batch_iter = iter(batches)

    for _ in range(steps):
        x, y = next(batch_iter)
        loss = model(x, y, loss_reduction='none')
        loss = loss.view(-1) # flatten
        y = y.view(-1)
        if (y.int()<0).any():
            valid = (y>=0)
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            num_bytes = torch.where(valid, token_bytes[y_safe], torch.zeros_like(y, dtype=token_bytes.dtype))
            valid_loss = loss * (num_bytes > 0)
            total_nats += valid_loss.sum()
            total_bytes += num_bytes.sum()
        else:
            num_bytes = token_bytes[y]
            total_nats += (loss * (num_bytes > 0)).sum()
            total_bytes += num_bytes.sum()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
            
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
        
    if total_bytes == 0:
        return float('inf')
        
    # Correct BPB formula: nats/ln(2)/bytes = bits/bytes
    bpb = total_nats /( math.log(2) * total_bytes )   
    return bpb
