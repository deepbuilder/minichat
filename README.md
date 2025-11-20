# minichat

The minimal full-stack ChatGPT clone

## Installation & Setup

### Prerequisites
- Python 3.10-3.13
- CUDA 12.8 compatible GPU (optional, for GPU acceleration)

### Quick Setup with uv (Recommended)

```bash

# Cloning the repository
git clone https://github.com/deepbuilder/minichat.git
cd minichat

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On macOS/Linux

# Install the package and dependencies
uv pip install -e .

# Install development dependencies (optional)
uv pip install -e ".[dev]"

# Install the environment as a Jupyter kernel
python -m ipykernel install --user --name=minichat --display-name="minichat"
```

### Alternative setup with pip

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install the package and dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"

# Install the environment as a Jupyter kernel
python -m ipykernel install --user --name=minichat --display-name="minichat"
```

### Using with VS Code Notebooks

1. Open your notebook in VS Code
2. Click on the kernel selector (top-right of notebook)
3. Choose "minichat" from the kernel list
4. The notebook will now use the environment with all dependencies

### Using with Jupyter Lab

```bash
# Start Jupyter Lab
jupyter lab
```

## Testing the run
```bash
# Activate the virtual environment if not already active
source .venv/bin/activate  # On macOS/Linux

# Download the dataset
python minichat/dataset.py -w 10

# Train the tokenizer
python scripts/tok_train.py

```

### Verification

Verify the setup is working correctly:

```bash
# Check PyTorch installation and CUDA availability
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check other key dependencies
python -c "import fastapi, tiktoken, wandb; print('All dependencies loaded successfully')"
```

### Running Tests

```bash
# Run the test suite
pytest

# Run tests excluding slow ones
pytest -m "not slow"
```

## Development

Once setup is complete, you can run the notebook `minichat/gpt.ipynb` with all dependencies available in the virtual environment.

### Notes
- This project includes Rust components (rustbpe) that will be compiled during installation
- PyTorch is configured to use CUDA 12.8 by default for GPU acceleration
- The project supports Python 3.10-3.13


# Learning-through-building plan for scaling a transformer

This roadmap stacks skills in the right order, so each milestone teaches you something you’ll use at the next level. You’ll ship working artifacts and measure progress at every step.

---

## Foundations with a vanilla transformer on Shakespeare

### Goal 1: Minimal GPT, end-to-end
- **Goal:** Train a tiny, readable GPT on Tiny Shakespeare and generate samples.
- **Deliverables:**
  - **Data:** Loader for Tiny Shakespeare with train/val split; fixed seeds.
  - **Tokenizer:** Character-level with BOS/EOS; later swap to byte-level BPE.
  - **Model:** Token embedding, learned positional embeddings, causal self-attention, MLP, residuals, LayerNorm.
  - **Training loop:** Cross-entropy loss, AdamW, cosine LR with warmup, gradient clipping.
  - **Evaluation:** Bits-per-char, perplexity, and sample generation script.
- **Success criteria:**  
  - **Convergence:** Smooth training and validation curves; finite perplexity.  
  - **Reproducibility:** Single config file, deterministic output with fixed seed.  
  - **Code quality:** Clear module boundaries (embeddings, attention, MLP, norms).

### Goal 2: Refactor, tests, and basic performance hygiene
- **Goal:** Make the code modular and testable; add mixed precision safely.
- **Deliverables:**
  - **Tests:** Unit tests for attention masks, shape invariants, numerical sanity; a golden perplexity test on a tiny split.
  - **Mixed precision:** fp16/bf16 autocast with grad scaling; verify stability.
  - **Profiling:** Tokens/sec, step-time breakdown, GPU memory; gradient norm logging.
  - **Artifacts:** run_all.sh to execute tests and a perf check notebook.
- **Success criteria:**  
  - **Determinism:** Greedy decoding reproduces identical outputs across runs.  
  - **Throughput:** Baseline tokens/sec documented for batch sizes 1/4/8.

---

## Scale to OpenWebText and replicate GPT‑2 behavior

### Goal 3: Data pipeline and GPT‑2 tokenizer
- **Goal:** Prepare a robust OWT pipeline and GPT‑2 BPE tokenizer.
- **Deliverables:**
  - **Preprocessing:** Deduplication, length filtering, UTF‑8 normalization, document boundaries.
  - **Tokenizer:** GPT‑2 BPE (vocab/merges) with special tokens; cache tokenized shards.
  - **Sharding:** Deterministic shard sampling, epoch definition, exact resume on failure.
  - **Manifests:** Dataset hashes, stats, and versioned preprocessing configs.
- **Success criteria:**  
  - **Throughput:** Dataloader tokens/sec meets expected targets; no loader stalls.  
  - **Resume:** Kill-and-resume recovers precisely to the same step.

### Goal 4: Train GPT‑2 small (~124M)
- **Goal:** Reach competitive validation perplexity on OWT with a 12-layer model.
- **Deliverables:**
  - **Config:** d_model≈768, 12 layers, 12 heads, MLP ratio 4x, small dropout.
  - **Optimizer:** AdamW (β1=0.9, β2=0.95), weight decay tuned; cosine LR with warmup.
  - **Evaluation:** Held-out OWT perplexity, qualitative prompt bank, checkpointing regime.
- **Success criteria:**  
  - **Quality:** Validation perplexity in the expected range for small GPT‑2.  
  - **Stability:** No NaNs; recoverable restarts; learning curves free from spikes.

### Goal 5: Medium scale (~355M) and training efficiency
- **Goal:** Scale depth/width and keep training stable and efficient.
- **Deliverables:**
  - **Config:** d_model≈1024, 24 layers; adjust batch/grad-accum to fit memory.
  - **Activation checkpointing:** Enable on attention and MLP blocks; measure compute vs. memory.
  - **Fused kernels:** Integrate FlashAttention for prefill and decode where available.
- **Success criteria:**  
  - **Utilization:** Tokens/sec scales with batch/sequence length; memory within budget.  
  - **Parity:** Perplexity does not regress relative to small-scale baselines per token budget.

---

## Inference pipeline aligned with nanoGPT

### Goal 6: Single-GPU CLI and Python API
- **Goal:** NanoGPT‑style inference that’s fast, hackable, and deterministic.
- **Deliverables:**
  - **Loader:** Checkpoint + config; eval mode; precision auto-detect; dropout disabled.
  - **Tokenizer:** GPT‑2 BPE encode/decode with batch support and attention masks.
  - **Generation:** Greedy/top‑k/top‑p/temperature; EOS handling; seed control; streaming generator.
  - **KV cache:** Prefill and incremental decode with O(1) step cost; per-request cache management.
- **Success criteria:**  
  - **Latency:** Document tokens/sec and per-step latency for batch sizes 1/4/8/16.  
  - **Determinism:** Identical greedy outputs for fixed seeds and configs.

### Goal 7: Efficiency features and MQA
- **Goal:** Reduce inference memory and improve throughput.
- **Deliverables:**
  - **MQA:** Implement shared K/V per head; adapt KV cache shapes; preserve per-head queries.
  - **Paged KV:** Optional for long prompts; chunked prefill; guard against OOM.
  - **Precision:** bf16 decode with fp32 sampling for numerical stability.
- **Success criteria:**  
  - **Memory:** KV cache reduction with MQA quantified; no unexpected perplexity hit.  
  - **Speed:** Measured throughput gains for MQA on/off across batch sizes.

---

## Architecture and training ablations

### Goal 8: Position and normalization
- **Goal:** Test alternatives that often improve stability and long-context handling.
- **Deliverables:**
  - **RoPE:** Replace learned positional embeddings with rotary embeddings; long-context eval prompts.
  - **Norm after embed:** Apply RMSNorm/LayerNorm directly after token embedding.
  - **RMSNorm without learnable gain:** Fixed-scale variant.
- **Success criteria:**  
  - **Quality:** Compute-adjusted perplexity comparisons on identical token budgets.  
  - **Stability:** Early training convergence speed; absence of loss spikes.

### Goal 9: Attention and MLP tweaks
- **Goal:** Probe changes that affect optimization dynamics.
- **Deliverables:**
  - **QK norm:** Normalize queries/keys before dot-product; try fixed vs. learnable scaling.
  - **ReLU² MLP:** Replace GELU with ReLU²; analyze gradient distribution and training speed.
  - **Untied heads:** Untie token embedding and lm_head; measure parameter vs. quality trade-offs.
- **Success criteria:**  
  - **Signal:** Attention entropy trends; gradient norm stability.  
  - **Value:** Perplexity gains relative to parameter increase for untied heads.

### Goal 10: Parameter economy
- **Goal:** Simplify without losing quality.
- **Deliverables:**
  - **No bias:** Remove biases in attention/MLP linears; verify parity.
  - **Consolidated report:** Tables of perplexity, tokens/sec, memory, and stability deltas for each ablation.
- **Success criteria:**  
  - **Keepers:** Identify a stack that improves sample efficiency and inference speed.  
  - **Documentation:** Each result linked to commit, config, dataset manifest, environment hash.

---

## Ultra-scale playbook–inspired optimization

### Goal 11: Memory and parallelism
- **Goal:** Train larger models efficiently on multiple GPUs.
- **Deliverables:**
  - **Sharding:** ZeRO/FSDP for optimizer states, gradients, and parameters.
  - **Parallelism:** Data + tensor parallel baseline; add pipeline when depth increases.
  - **Checkpointing:** Verify recomputation strategy and memory savings.
- **Success criteria:**  
  - **Scale:** Fit larger models without OOM; utilization remains high.

### Goal 12: Throughput and communication overlap
- **Goal:** Increase tokens/sec with system-level tuning.
- **Deliverables:**
  - **Overlap:** All-reduce/all-gather overlapped with compute; tune bucket sizes and NCCL params.
  - **Topology:** Placement aligned with NVLink/PCIe domains to minimize cross-node traffic.
  - **Fused ops:** Fused optimizers and bias+activation paths where available.
- **Success criteria:**  
  - **Benchmarks:** Step-time breakdown shows reduced comm stalls; higher tokens/sec.

### Goal 13: Data pipeline at scale
- **Goal:** Keep GPUs fed without hiccups.
- **Deliverables:**
  - **Loader:** Async prefetch, pinned memory, memory-mapped datasets, CPU thread pools.
  - **Determinism:** Seeded shuffles; shard tracking; exact resume semantics.
- **Success criteria:**  
  - **Smooth ingest:** No data starvation; throughput stable across long runs.

### Goal 14: Monitoring and guardrails
- **Goal:** Observe everything that matters, catch regressions.
- **Deliverables:**
  - **Metrics:** Tokens/sec, GPU utilization, step-time components, comm overhead, memory, gradient norms, attention entropy.
  - **Dashboards:** Centralized plots; config/environment capture; regression alarms.
- **Success criteria:**  
  - **Visibility:** Clear correlation between changes and throughput/quality.

---

## Consolidation and larger run

### Goal 15: Choose the winning stack and pilot a bigger model
- **Goal:** Lock in architecture, training, and inference choices that win.
- **Deliverables:**
  - **Architecture:** RoPE, norm-after-embed, QK norm (if helpful), ReLU² or GELU, bias removal (if neutral), MQA for inference.
  - **Training:** Mixed precision, activation checkpointing, ZeRO/FSDP, fused kernels, tuned communication overlap.
  - **Pilot run:** 355M or 774M on a shorter token budget to validate stability and speed.
- **Success criteria:**  
  - **Compute-adjusted wins:** Better perplexity per token and improved inference latency/memory.

### Goal 16: Final report and release pack
- **Goal:** Make results reproducible and portable.
- **Deliverables:**
  - **Release:** Config pack, train/eval scripts, infer.py, ablation tables, benchmark notebook, reproducibility guide.
  - **Documentation:** Clear instructions for data ingestion, training, inference, and benchmarking.
- **Success criteria:**  
  - **Plug-and-play:** A new machine can rerun baselines and get similar numbers.

---

## Experiment protocol you’ll follow throughout

- **Fixed budgets:** Compare variants with identical tokens and schedules; report mean ± std over ≥3 seeds.
- **Learning curves:** Loss vs. tokens and compute-adjusted comparisons (quality per GPU-hour).
- **Inference benchmarks:** Latency, tokens/sec, KV cache memory; batch sizes 1/4/8/16; MQA on/off.
- **Traceability:** Every result links to commit, config, dataset manifest, and environment details.

---

## Optional next steps after the core plan

- **Long-context:** Extend context window; evaluate degradation and RoPE scaling behavior.  
- **Precision exploration:** FP8 with calibration if hardware supports; careful monitoring for stability.  
- **Serving:** Minimal web endpoint with streaming; dynamic batching scheduler. 
