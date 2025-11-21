import os
import torch

def save_checkpoint(checkpoint_dir, model, optimizer, step, meta_data):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f'model_step{step:06d}.pt')
    torch.save(model, model_path)
    if optimizer is not None:
        optim_path = os.path.join(checkpoint_dir, f'optimizer_step{step:06d}.pt')
        torch.save(optimizer, optim_path)
    meta_path = os.path.join(checkpoint_dir, f'meta_step{step:06d}.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(meta_data, f, indent=2)
    print(f"Checkpoint saved at step {step} to {checkpoint_dir}")

def load_checkpoint(checkpoint_dir, step, device='cpu'):
    model_path = os.path.join(checkpoint_dir, f'model_step{step:06d}.pt')
    model = torch.load(model_path, map_location=device)
    optim_path = os.path.join(checkpoint_dir, f'optimizer_step{step:06d}.pt')
    optimizer = None
    if os.path.exists(optim_path):
        optimizer = torch.load(optim_path, map_location=device)
    meta_path = os.path.join(checkpoint_dir, f'meta_step{step:06d}.json')
    meta_data = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            import json
            meta_data = json.load(f)
    print(f"Checkpoint loaded from step {step} in {checkpoint_dir}")
    return model, optimizer, meta_data