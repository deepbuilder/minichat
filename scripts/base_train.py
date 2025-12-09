import time
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch

from minichat.gpt import GPTConfig, GPT
from minichat.common import get_base_dir
from minichat.dataloader import data_loader_w_state, data_loader
from minichat.loss_eval import evaluate_bpb
from minichat.tokenizer import get_token_bytes, get_tokenizer
from scripts.base_eval import evaluate_model
from minichat.checkpoint import save_checkpoint, load_checkpoint
from minichat.common import get_ddp_info, print_log, DummyWandb, compute_cleanup, compute_init
from minichat.engine import Engine

run = "test_run"
device_type = "cuda"

# Model hyperparameters
depth = 8
max_seq_len = 2048

# Training params
num_iterations = -1
target_flops = -1
target_param_data_ratio = 20

# Optimization
device_batch_size = 16
total_batch_size = 2**19 # in tokens
unembedding_lr = 0.000004  # was 0.00002, reduced 5x more
embedding_lr = 0.000008   # was 0.00004, reduced 5x more
matrix_lr = 0.00002       # was 0.0001, reduced 5x more
grad_clip = 1.0 
weight_decay = 0.2       # was 0.1, doubled for stronger regularization
warmup_ratio = 0.25      # was 0.15, longer warmup for stability
warmdown_ratio = 0.2
final_lr_frac = 0.1      # was 0.0, allow some final learning
resume_from_step = -1

# Evaluation
eval_every = 50  # was 100, catch overfitting much sooner
eval_tokens = 8*2**19  # was 4*2**19, double for more stable metrics
core_metric_every = -1  # Disable core metrics evaluation temporarily
core_metric_max_per_task = 500
sample_every = 250
save_every = 250
# Debug settings
debug_mode = True
track_grad_norms = True
validation_debug_every = 50  # Even more frequent validation checks
# Early stopping
early_stopping_patience = 2  # was 3, stop even sooner
early_stopping_threshold = 0.005  # was 0.01, more sensitive to degradation

# Output
model_tag = ""
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, str, bool))]
user_config = {k: globals()[k] for k in config_keys}

# Compute Init
ddp, rank, local_rank, world_size, device = compute_init()
master_rank = rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else torch.cpu.amp.autocast
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
total_peak_memory = torch.cuda.max_memory_allocated

# Wandb logging
use_wandb = run is not None and run != "dummy_run" and master_rank
if use_wandb:
    import wandb
    wandb_run = wandb.init(project="minichat", name=run, config=user_config)
else:
    wandb_run = DummyWandb()
# Tokenizer related info for evaluation
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print_log(f"Vocab size: {vocab_size}")

# model kwargs
num_layers = depth
model_dim = depth * 64
num_heads = max(1, (model_dim + 127) // 128)
num_kv_heads = num_heads 
print_log(f"Model dim: {model_dim}")
print_log(f"Num heads: {num_heads}")
print_log(f"Num kv heads: {num_kv_heads}")
sequence_len = max_seq_len

# Training length hyperparameters
tokens_per_fwdbwd = device_batch_size * sequence_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0, f"Total batch size {total_batch_size} must be divisible by world tokens per fwdbwd {world_tokens_per_fwdbwd}"
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
total_tokens = total_batch_size * num_iterations
print_log(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print_log(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print_log(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Model initialization
model_config_kwargs = dict(sequence_len=sequence_len,
                           vocab_size=vocab_size,
                           emb_dim=model_dim,
                           n_layers=num_layers,
                           n_heads=num_heads,
                           n_kv_heads=num_kv_heads,)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)

model.to_empty(device=device)
model.init_weights()

# Resume from checkpoint if specified
base_dir = get_base_dir()
output_dirname = model_tag if model_tag else f"d{num_layers}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = resume_from_step >= 0
if resuming:
    print_log(f"Resuming from checkpoint at step {resume_from_step}")
    model_state_dict, optimizer_state_dict, extra_state = load_checkpoint(checkpoint_dir, resume_from_step, device=device, load_optimizer=True, rank=rank)
    model.load_state_dict(model_state_dict, strict=True, assign=True)
    del model_state_dict

orig_model = model
model = torch.compile(model, dynamic=False)

num_params = sum(p.numel() for p in model.parameters())
num_flops_per_token = model.estimate_flops_per_token()
print_log(f"Model parameters: {num_params/1e6:.2f}M")
print_log(f"Model FLOPs per token: {num_flops_per_token/1e9:.2f}B")

# Determine number of iterations if specified via target flops or param-data ratio
assert num_iterations > 0 or target_flops > 0 or target_param_data_ratio > 0, "Must specify num_iterations, target_flops, or target_param_data_ratio"
if num_iterations > 0:
    print_log(f"User-specified number of iterations: {num_iterations}")
elif target_flops > 0:
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print_log(f"Computed number of iterations from target flops: {num_iterations}")
elif target_param_data_ratio > 0:
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print_log(f"Computed number of iterations from target param-data ratio: {num_iterations}")
else:
    raise ValueError("Unable to determine number of iterations")

total_tokens = total_batch_size * num_iterations
print_log(f"Total training tokens: {total_tokens/1e7:.2f}M")
print_log(f"Tokens: Param data ratio: {total_tokens / num_params:.2f}")
print_log(f"Total training FLOPs: {num_flops_per_token * total_tokens / 1e12:.2f} TFLOPs")
# ---------

# Optimizer setup
optimizer = model.setup_optimizer(
    embedding_lr=embedding_lr,
    unembedding_lr=unembedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)
if resuming and optimizer_state_dict is not None:
    optimizer.load_state_dict(optimizer_state_dict)
    del optimizer_state_dict

# Initialize input batch
tokens_dir = os.path.join(base_dir, "tokenized_data")
data_loader_state_dict = None if not resuming else extra_state.get("data_loader_state_dict", None)
train_loader = data_loader_w_state(
    B=device_batch_size,
    T=sequence_len,
    split='train',
    tokenizer_batch_size=16,
    tokenizer_threds=4,
    device=device,
    resume_state_dict=data_loader_state_dict,
)
val_loader = data_loader(
    B=device_batch_size,
    T=sequence_len,
    split='val',
    tokenizer_batch_size=16,
    tokenizer_threds=4,
    device=device,
)

x, y, data_loader_state_dict = next(train_loader)

def get_lr_multiplier(step):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if step == 0:  # Debug output
        print_log(f"Warmup iterations: {warmup_iters}, Total iterations: {num_iterations}")
    if step < warmup_iters:
        return (step + 1) / warmup_iters
    elif step <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - step) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Loop state

if not resuming:
    step = 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0.0
    # Debug tracking
    val_bpb_history = []
    train_loss_history = []
    overfitting_counter = 0
    last_val_bpb = float("inf")
    early_stop_triggered = False
else:
    step = extra_state.get("step", 0)
    loop_state = extra_state.get("loop_state", {})
    min_val_bpb = loop_state.get("min_val_bpb", float("inf"))
    smooth_train_loss = loop_state.get("smooth_train_loss", 0.0)
    total_training_time = loop_state.get("total_training_time", 0.0)
    # Debug tracking
    val_bpb_history = loop_state.get("val_bpb_history", [])
    train_loss_history = loop_state.get("train_loss_history", [])
    overfitting_counter = loop_state.get("overfitting_counter", 0)
    last_val_bpb = loop_state.get("last_val_bpb", float("inf"))
    early_stop_triggered = False


def sample_inference(model, prompt_tokens, max_new_tokens):
    model.eval()
    generated_tokens = list(prompt_tokens)
    for token in model.generate(prompt_tokens, max_new_tokens):
        generated_tokens.append(token)
    return generated_tokens


def compute_gradient_stats(model):
    """Compute gradient statistics for debugging"""
    total_norm = 0.0
    param_count = 0
    layer_norms = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            param_count += 1
            
            # Group by layer type
            layer_type = name.split('.')[0] if '.' in name else name
            if layer_type not in layer_norms:
                layer_norms[layer_type] = []
            layer_norms[layer_type].append(param_norm)
    
    total_norm = total_norm ** 0.5
    avg_layer_norms = {k: sum(v)/len(v) for k, v in layer_norms.items()}
    
    return total_norm, param_count, avg_layer_norms

def inference(model, tokenizer):
    model.eval()
    prompts = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
    ]
    engine = Engine(model, tokenizer)
    for prompt in prompts:
        prompt_tokens = tokenizer.encode(prompt, prepend="<|bos|>")
        with autocast_ctx:
            sample = engine.generate_batch(
                tokens=prompt_tokens,
                num_samples=1,
                max_tokens=16,
                temperature=0
            )
        print_log(tokenizer.decode(sample[0]))
    model.train()
# --------- 

# Training loop
while True:
    last_step = (step == num_iterations)
    flops_performed = step * total_batch_size * num_flops_per_token

    val_bpb = None

    # More frequent validation for debugging
    should_eval = last_step or (step > 0 and step % eval_every == 0)
    should_debug_eval = debug_mode and step > 0 and step % validation_debug_every == 0
    
    if should_eval or should_debug_eval:
        model.eval()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len* world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, steps=eval_steps, token_bytes=token_bytes)
        
        # Debug logging
        val_bpb_history.append(val_bpb)
        train_loss_history.append(smooth_train_loss)
        
        # Check for overfitting pattern
        if val_bpb > last_val_bpb + early_stopping_threshold:
            overfitting_counter += 1
        else:
            overfitting_counter = max(0, overfitting_counter - 1)
        
        # Early stopping check
        if overfitting_counter >= early_stopping_patience:
            print_log(f"🛑 EARLY STOPPING triggered after {early_stopping_patience} consecutive validation increases")
            print_log(f"   Current val BPB: {val_bpb:.4f}, Best: {min_val_bpb:.4f}")
            early_stop_triggered = True
        
        # Detailed logging
        val_trend = "📈" if val_bpb > last_val_bpb else "📉"
        print_log(f"Step {step:05d}, Validation BPB: {val_bpb:.4f} {val_trend} (prev: {last_val_bpb:.4f})")
        
        if debug_mode:
            print_log(f"  Train Loss: {smooth_train_loss:.4f}, Val BPB: {val_bpb:.4f}")
            print_log(f"  Overfitting counter: {overfitting_counter}/{early_stopping_patience}")
            print_log(f"  Gap (train vs val): {val_bpb - smooth_train_loss:.4f}")
            
            # Enhanced warnings
            if overfitting_counter >= early_stopping_patience - 1:
                print_log(f"  ⚠️  WARNING: One step away from early stopping!")
            elif overfitting_counter >= 2:
                print_log(f"  🔶 CAUTION: Validation degrading trend detected")
            
            if len(val_bpb_history) >= 3:
                recent_trend = sum(val_bpb_history[-3:])
                older_trend = sum(val_bpb_history[-6:-3]) if len(val_bpb_history) >= 6 else recent_trend
                if recent_trend > older_trend:
                    print_log(f"  ⚠️  Validation BPB trending upward over last 3 evaluations!")
        
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
            print_log(f"  🎯 New best validation BPB: {min_val_bpb:.4f}")
        
        last_val_bpb = val_bpb
        
        # Enhanced wandb logging
        log_data = {
            "step": step,
            "total_training_flops": flops_performed,
            "total_training_time": total_training_time,
            "val_bpb": val_bpb,
            "train_val_gap": val_bpb - smooth_train_loss,
            "overfitting_counter": overfitting_counter,
        }
        if len(val_bpb_history) >= 2:
            log_data["val_bpb_trend"] = val_bpb_history[-1] - val_bpb_history[-2]
        
        wandb_run.log(log_data)
        model.train()
        
        # Check for early stopping
        if early_stop_triggered:
            print_log(f"\n🛑 Training stopped early due to validation degradation")
            print_log(f"Final validation BPB: {val_bpb:.4f}")
            print_log(f"Best validation BPB: {min_val_bpb:.4f}")
            break

    results = {}
    if step >0 and (core_metric_every > 0 and (step % core_metric_every == 0 or last_step)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(
                model,
                tokenizer,
                max_per_task=core_metric_max_per_task,
                device=device,
            )
            print_log(f"Step {step:05d}, Core Metrics: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_performed,
            "core_metric": results['core_metric'],
            "centered_results": results.get('centered_results', 0.0),
        })
        model.train()
    
    if master_rank and (last_step or (sample_every > 0 and  step >0 and step % sample_every == 0)):
        # Inference samples
        inference(orig_model, tokenizer)

    
    if last_step or (step >0 and step != resume_from_step and save_every > 0 and step % save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            meta_data={
                "step": step,
                "val_bpb": val_bpb,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
                "data_loader_state_dict": data_loader_state_dict,
                "loop_state": {
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                    "val_bpb_history": val_bpb_history[-20:],  # Keep last 20 values
                    "train_loss_history": train_loss_history[-20:],
                    "overfitting_counter": overfitting_counter,
                    "last_val_bpb": last_val_bpb,
                    "early_stop_triggered": early_stop_triggered,
                },
            }
        )
    if last_step or early_stop_triggered:
        break

    synchronize()
    t0 = time.time()

    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        if micro_step < grad_accum_steps - 1:  # Don't load extra batch on last step
            x, y, data_loader_state_dict  = next(train_loader)

    # Gradient analysis before clipping
    if track_grad_norms and step % 10 == 0:
        total_grad_norm, param_count, layer_grad_norms = compute_gradient_stats(orig_model)
        if step % 100 == 0:  # Less frequent detailed logging
            print_log(f"  Grad stats - Total: {total_grad_norm:.4f}, Params with grad: {param_count}")
            for layer, norm in layer_grad_norms.items():
                print_log(f"    {layer}: {norm:.4f}")
    
    # Add gradient clipping to prevent exploding gradients
    grad_clip_enabled = grad_clip > 0
    if grad_clip_enabled:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
        grad_norm = grad_norm_tensor.item()
        
        # Warn about gradient clipping
        if grad_norm > grad_clip and step % 10 == 0:
            print_log(f"  ⚠️  Gradient clipped: {grad_norm:.4f} -> {grad_clip}")
    
    lrm = get_lr_multiplier(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['initial_lr'] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)
    
    # Load next batch for next iteration
    x, y, data_loader_state_dict = next(train_loader)

    t1 = time.time()
    dt = t1 - t0

    # Logging
    ema_beta = 0.9
    smooth_ema_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_ema_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100.0 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)

    flops_per_sec =  num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec = 989e12
    mfu = 100 * flops_per_sec / promised_flops_per_sec
    if step > 10:
        total_training_time += dt
    print_grad_norm = f" grad_norm: {grad_norm:.4f} " if grad_clip_enabled else ""
    
    # More detailed logging every 50 steps
    if step % 50 == 0:
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        lr_info = f" LRs: emb={current_lrs[0]:.2e}, mat={current_lrs[1]:.2e}, unemb={current_lrs[2]:.2e}" if len(current_lrs) >= 3 else ""
        print_log(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f}{lr_info}{print_grad_norm}| dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    else:
        print_log(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f}{print_grad_norm}| dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")

    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_performed,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        
        # Add learning rate tracking
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        if len(current_lrs) >= 3:
            log_data["train/lr_embedding"] = current_lrs[0]
            log_data["train/lr_matrix"] = current_lrs[1]
            log_data["train/lr_unembedding"] = current_lrs[2]
        
        if grad_clip_enabled:
            log_data["train/grad_norm"] = grad_norm
            if grad_norm > grad_clip:
                log_data["train/grad_clipped"] = 1
        
        # Add gradient monitoring
        if track_grad_norms:
            total_grad_norm, _, _ = compute_gradient_stats(orig_model)
            log_data["train/grad_norm_total"] = total_grad_norm
        
        wandb_run.log(log_data)
    step += 1

print_log(f"Total training time: {total_training_time/60:.2f}m")
print_log(f"Peak Memory Usage: {total_peak_memory() / 1024 / 1024:.2f}MiB")  
print_log(f"Minimum validation bpb: {min_val_bpb:.4f}")

wandb_run.finish()
compute_cleanup()

# ---------