import torch
import json
from minichat.gpt import GPT, GPTConfig
from minichat.tokenizer import get_tokenizer

def save_checkpoint(check_dir, step, model, optimizer, meta_data):
    os.makedirs(check_dir, exist_ok=True)
    checkpoint_path = os.path.join(check_dir, f'checkpoint_{step:06d}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'meta_data': meta_data
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    if optimizer.state is not None:
        optimizer_state_path = os.path.join(check_dir, f'optimizer_state_{step:06d}.pt')
        torch.save(optimizer.state_dict(), optimizer_state_path)
        print(f"Optimizer state saved at {optimizer_state_path}")
    meta_data_path = os.path.join(check_dir, f'meta_data_{step:06d}.json')
    with open(meta_data_path, 'w') as f:
        json.dump(meta_data, f, indent=4)
    print(f"Meta data saved at {meta_data_path}")

def load_checkpoint(checkpoint_path, step, device=None, load_optimizer=False):
    model_path = os.path.join(checkpoint_path, f'model_{step:06d}.pt')
    model_data = torch.load(model_path, map_location=device)

    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_path, f'optim_{step:06d}.pt' )
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
    with torch.device(meta):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    tokenizer = get_tokenizer()
    return model, tokenizer