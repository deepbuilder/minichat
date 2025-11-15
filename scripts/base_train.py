import torch

from minichat.gpt import GPTConfig, GPT

sequence_len =  512
n_layers = 12
vocab_size = 64
emb_dim = 128
n_heads = 8
device = "cuda"

num_iterations = 10
total_batch_size = 64

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
num_flops = model.estimate_flops_per_token()
print(f"Model parameters: {num_params/1e6:.2f}M")
print(f"Model FLOPs per token: {num_flops/1e9:.2f}B")


total_tokens = total_batch_size * num_iterations
print(f"Training for {total_tokens/1e9:.2f}B tokens")
print(f"Tokens : Params ratio : {total_tokens/num_params:.2f}")
print(f"Total training FLOPs : {total_tokens * num_flops / 1e18:.2f} EFLOPs")

# ---------
# Initialize the Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


# Initialize a dummy input batch
input_batch = torch.randint(0, vocab_size, (total_batch_size, sequence_len), device=device)
target_batch = torch.randint(0, vocab_size, (total_batch_size, sequence_len), device=device)

# Training loop
for step in range(num_iterations):
    optimizer.zero_grad()
    loss = model(input_batch, targets=target_batch)
    loss.backward()
    optimizer.step()
    if (step + 1) % 1 == 0:
        print(f"Step {step+1}/{num_iterations}, Loss: {loss.item():.4f}")
# ---------