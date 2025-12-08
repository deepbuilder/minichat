# minichat

The minimal full-stack ChatGPT clone

## Installation & Setup

### Prerequisites
- Python 3.10-3.13
- Rust (for tokenizer)
- CUDA 12.8 compatible GPU (optional, for GPU acceleration)

### Setup Instructions

#### 1. Clone the Repository
```bash
git clone https://github.com/deepbuilder/minichat.git
cd minichat
```

#### 2. Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 3. Set Up Python Environment
```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On macOS/Linux

# Install Rust dependencies
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Install the package and dependencies (this will automatically build the Rust extension)
uv pip install -e .

# Install the environment as a Jupyter kernel
python -m ipykernel install --user --name=minichat --display-name="minichat"
```

## Quick Start

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Download the dataset
python -m minichat.dataset -w 100

# Train the tokenizer
python -m scripts.tok_train --max_chars=4000000000

# Train the model
torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- --depth=4 --device_batch_size=8

```

## Roadmap

- [x] Cleanup Eval script
- [x] Add Gradient accumulation
- [x] Add optimizer
- [x] Add LR scheduler
- [x] Add Wandb logging
- [x] Add Mixed precision training
- [x] Add Inference engine
- [x] Use Muon optimizer
- [x] DDP support
- [x] Inference Engine
- [ ] Training on full dataset
- [ ] SFT training
- [ ] RLHF training
- [ ] Web UI

## Acknowledgments

This project builds upon the excellent educational foundation provided by Andrej Karpathy's nanochat. We've adapted and extended many of the core concepts and architectural decisions from that work to create this minimal ChatGPT implementation.

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```