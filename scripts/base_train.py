import torch

from minichat.gpt import GPTConfig, GPT
from minichat.common import get_base_dir
from minichat.dataloader import data_loader
from minichat.loss_eval import evaluate_bpb
from minichat.tokenizer import get_token_bytes

sequence_len =  128
n_layers = 4
vocab_size = 65_536
emb_dim = 64
n_heads = 8
device = "cuda"

num_iterations = 10
total_batch_size = 4

model_config_kwargs = {
    "sequence_len": sequence_len,
    "n_layers": n_layers,
    "vocab_size": vocab_size,
    "emb_dim": emb_dim,
    "n_heads": n_heads,
}

with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)

model.to_empty(device=device)
model.init_weights()

model = torch.compile(model, dynamic=False)

num_params = sum(p.numel() for p in model.parameters())
num_flops_per_token = model.estimate_flops_per_token()
print(f"Model parameters: {num_params/1e6:.2f}M")
print(f"Model FLOPs per token: {num_flops_per_token/1e9:.2f}B")


total_tokens = total_batch_size * num_iterations
print(f"Training for {total_tokens/1e9:.2f}B tokens")
print(f"Tokens : Params ratio : {total_tokens/num_params:.2f}")
print(f"Total training FLOPs : {total_tokens * num_flops_per_token / 1e18:.2f} EFLOPs")

# ---------
# Initialize the Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


# Initialize input batch
base_dir = get_base_dir()
train_loader = data_loader(total_batch_size, sequence_len, 'train')
val_loader = data_loader(total_batch_size, sequence_len, 'val')
token_bytes = get_token_bytes()


# Training loop
for step in range(num_iterations):
    if step % 2 == 0:
        model.eval()
        val_bpb = evaluate_bpb(model, val_loader, steps=1, token_bytes=token_bytes)
        print(f"Step {step}/{num_iterations}, Validation BPB: {val_bpb:.4f}")

        model.train()
    optimizer.zero_grad()
    x, y = next(train_loader)
    loss = model(x, targets=y)
    loss.backward()
    optimizer.step()
    if (step + 1) % 1 == 0:
        print(f"Step {step+1}/{num_iterations}, Loss: {loss.item():.4f}")
# ---------