import torch
import time

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
total_batch_size = 256 # in tokens
core_metric_every = 5

num_iterations = 100
eval_tokens = 20 * 256

# Optimizer params
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0

# LR Scheduler params
warmup_ratio =0.0
warmdown_ratio = 0.2
final_lr_frac = 0.0

total_peak_memory = torch.cuda.max_memory_allocated


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

# Initialize input batch
base_dir = get_base_dir()
train_loader = data_loader(device_batch_size, sequence_len, 'train')
val_loader = data_loader(device_batch_size, sequence_len, 'val')
token_bytes = get_token_bytes()

# Optimizer
optimizer = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)

eval_every = 4
x, y = next(train_loader)

def get_lr_multiplier(step):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if step < warmup_iters:
        return (step + 1) / warmup_iters
    elif step <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - step) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac


min_val_bpb = float("inf")
smooth_train_loss = 0 # EMA of training loss
ema_beta = 0.9 # EMA decay factor
total_training_time = 0.0

t0 = time.time()
# Training loop
for step in range(num_iterations):
    last_step = (step == num_iterations - 1)
    if step % eval_every == 0 or last_step:
        model.eval()
        eval_steps = eval_tokens
        val_bpb = evaluate_bpb(model, val_loader, steps=1, token_bytes=token_bytes)
        print(f"Step {step:05d}, Validation BPB: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        model.train()

    results = {}
    if core_metric_every > 0 and (step % core_metric_every == 0 or last_step):
        model.eval()
        # results = evaluate_model(orig_model, tokenizer, device, max_per_task=100)
        # print(f"Step {step} | CORE metric: {results['core_metric']:.4f}")
        model.train()


    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y = next(train_loader)

    lrm = get_lr_multiplier(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['initial_lr'] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0

    smooth_ema_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_ema_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100.0 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)

    flops_per_sec =  num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec = 989e12
    mfu = 100 * flops_per_sec / promised_flops_per_sec
    if step > 10:
        total_training_time += dt
    print(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
print(f"Total training time: {total_training_time/60:.2f}m")
print(f"Peak Memory Usage: {total_peak_memory() / 1024 / 1024:.2f}MiB")  
print(f"Minimum validation bpb: {min_val_bpb:.4f}")

# ---------