from collections import deque
import os
import torch
import pyarrow.parquet as pq

from minichat.common import get_ddp_info
from minichat.tokenizer import get_tokenizer
from minichat.dataset import list_files, iter_batched
from minichat.common import get_ddp_info



def data_loader_w_state(B, T, split, tokenizer_batch_size=16, tokenizer_threds=4, device='cuda', resume_state_dict=None):
    assert split in ['train', 'val'], f"Invalid split: {split}"
    ddp, rank, local_rank, world_size = get_ddp_info()
    def document_batches():
        parquet_files = list_files()
        parquet_files = parquet_files[:-1] if split == 'train' else parquet_files[-1:]
        resume_pq_index =  resume_state_dict['pq_idx'] if resume_state_dict is not None else 0
        resume_rg_index =  resume_state_dict['rg_idx'] if resume_state_dict is not None else None
        pq_idx = resume_pq_index
        while True:
            while pq_idx < len(parquet_files):
                filepath = parquet_files[pq_idx]
                pf = pq.ParquetFile(filepath)
                if resume_rg_index is not None:
                    base_idx = resume_rg_index // world_size
                    base_idx +=1
                    rg_idx = base_idx * world_size + rank
                    resume_rg_index = None
                else:
                    rg_idx = rank
                while rg_idx < pf.num_row_groups:
                    row_group = pf.read_row_group(rg_idx)
                    documents = row_group.column('text').to_pylist()
                    for i in range(0, len(documents), tokenizer_batch_size):
                        batch_docs = documents[i:i+tokenizer_batch_size]
                        yield batch_docs, (pq_idx, rg_idx)
                    rg_idx += world_size
                pq_idx +=1
    
    batches = document_batches()


    needed_tokens = B*T + 1
    tokenizer = get_tokenizer()
    bos_token_id = tokenizer.get_bos_token_id()
    
    token_buffer = deque()

    while True:
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)       
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token_id, num_threads=tokenizer_threds)
            for token in token_lists:
                token_buffer.extend(token)
    
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        scrach = torch.tensor(tokens, dtype=torch.long, pin_memory=(device == 'cuda'))
        inputs_cpu = scrach[:-1]
        targets_cpu = scrach[1:]

        inputs = inputs_cpu.view(B, T).to(device=device, dtype=torch.long, non_blocking=True)
        targets = targets_cpu.view(B, T).to(device=device, dtype=torch.long, non_blocking=True)

        state_dict = {
            'pq_idx': pq_idx,
            'rg_idx': rg_idx,
        }

        yield inputs, targets, state_dict


def data_loader(*args, **kwargs):
    for inputs, targets, _ in data_loader_w_state(*args, **kwargs):
        yield inputs, targets


