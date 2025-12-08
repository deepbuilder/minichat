import torch
import json
import os
import logging
from minichat.gpt import GPT, GPTConfig
from minichat.tokenizer import get_tokenizer
from minichat.common import get_base_dir, setup_logger, find_rank

setup_logger()
logger = logging.getLogger(__name__)
def log(message):
    # log only from rank 0 in DDP
    if find_rank() == 0:
        logger.info(message)
    
def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        log(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        log(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        log(f"Saved optimizer state to: {optimizer_path}")
    
def load_checkpoint(checkpoint_path, step, device=None, load_optimizer=False, rank=0):
    model_path = os.path.join(checkpoint_path, f'model_{step:06d}.pt')
    model_data = torch.load(model_path, map_location=device)

    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_path, f'optim_{step:06d}_rank{rank:d}.pt')
        optimizer_data = torch.load(optimizer_path, map_location=device)

    meta_path = os.path.join(checkpoint_path, f'meta_{step:06d}.json')
    with open(meta_path, 'r') as file:
        meta_data = json.load(file)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    model_data, optimizer_state_, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    # fix for torch.compile issue
    model_data = {k.removeprefix("_orig_mod."): v for k,v in model_data.items()}
    model_config_params = meta_data['model_config']
    model_config = GPTConfig(**model_config_params)
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval() if phase == 'eval' else model.train()
    tokenizer = get_tokenizer()
    return model, tokenizer, meta_data