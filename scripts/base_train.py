import torch

from minichat.gpt import GPTConfig, GPT
from minichat.common import get_base_dir
from minichat.dataloader import data_loader
from minichat.loss_eval import evaluate_bpb
from minichat.tokenizer import get_token_bytes, get_tokenizer
from base_eval import evaluate_model

sequence_len =  2048
n_layers = 4
vocab_size = 65_536
emb_dim = 64
n_heads = 8
device = "cuda"

device_batch_size = 4
total_batch_size = 256
core_metric_every = 5

num_iterations = 50
eval_tokens = 20 * 256

model_config_kwargs = {
    "sequence_len": sequence_len,
    "n_layers": n_layers,
    "vocab_size": vocab_size,
    "emb_dim": emb_dim,
    "n_heads": n_heads,
}

# Tokenizer related info for evaluation
tokenizer = get_tokenizer()
token_bytes = get_token_bytes()
vocab_size = tokenizer.get_vocab_size()

with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)

model.to_empty(device=device)
model.init_weights()

orig_model = model

model = torch.compile(model, dynamic=False)

num_params = sum(p.numel() for p in model.parameters())
num_flops_per_token = model.estimate_flops_per_token()
print(f"Model parameters: {num_params/1e6:.2f}M")
print(f"Model FLOPs per token: {num_flops_per_token/1e9:.2f}B")


tokens_per_fwdbwd = device_batch_size * sequence_len
grad_accum_steps = total_batch_size // device_batch_size
total_tokens = total_batch_size * num_iterations

print(f"Training for {total_tokens/1e9:.2f}B tokens")
print(f"Tokens : Params ratio : {total_tokens/num_params:.2f}")
print(f"Total training FLOPs : {total_tokens * num_flops_per_token / 1e18:.2f} EFLOPs")

# ---------
# Initialize the Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Initialize input batch
base_dir = get_base_dir()
train_loader = data_loader(device_batch_size, sequence_len, 'train')
val_loader = data_loader(device_batch_size, sequence_len, 'val')
token_bytes = get_token_bytes()


eval_every = 4
x, y = next(train_loader)

# Training loop
for step in range(num_iterations):
    last_step = (step == num_iterations - 1)
    if step % eval_every == 0 or last_step:
        model.eval()
        eval_steps = eval_tokens
        val_bpb = evaluate_bpb(model, val_loader, steps=1, token_bytes=token_bytes)
        print(f"Step {step}/{num_iterations}, Validation BPB: {val_bpb:.4f}")
        model.train()

    results = {}
    if core_metric_every > 0 and (step % core_metric_every == 0 or last_step):
        model.eval()
        results = evaluate_model(orig_model, tokenizer, device, max_per_task=100)
        print(f"Step {step} | CORE metric: {results['core_metric']:.4f}")
        model.train()


    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y = next(train_loader)

    optimizer.step()
    optimizer.zero_grad()

    print(f"Step {step}/{num_iterations}, Train Loss: {train_loss:.4f}")
# ---------