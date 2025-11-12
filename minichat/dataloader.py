from collections import deque
import os
import torch

from minichat.common import get_ddp_info
from minichat.tokenizer import get_tokenizer
from minichat.dataset import list_files, iter_batched



def data_loader(B, T, split, tokenizer_batch_size=16, tokenizer_threds=4, device='cuda'):
    assert split in ['train', 'val'], f"Invalid split: {split}"
    needed_tokens = B*T + 1
    tokenizer = get_tokenizer()
    bos_token_id = tokenizer.get_bos_token_id()
    
    token_buffer = deque()

    def document_batches():
        while True:
            for batch in iter_batched(split):
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i + tokenizer_batch_size]
    
    batches = document_batches()
    batch_idx = 0
    while True:
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)       
            token_lists = tokenizer.encode(doc_batch, prepand=bos_token_id)
            for token in token_lists:
                token_buffer.extend(token)
        batch_idx += 1
    
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        scrach = torch.tensor(tokens, dtype=torch.int32, pin_memory=(device == 'cuda'))
        inputs_cpu = scrach[:-1].to(dtype=torch.int32)
        targets_cpu = scrach[1:]

        inputs = inputs_cpu.view(B, T).to(device=device, dtype=torch.int32, non_blocking=True)
        targets = targets_cpu.view(B, T).to(device=device, dtype=torch.int32, non_blocking=True)

        yield inputs, targets




