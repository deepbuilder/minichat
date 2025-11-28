# minichat

The minimal full-stack ChatGPT clone

## Installation & Setup

### Prerequisites
- Python 3.10-3.13
- CUDA 12.8 compatible GPU (optional, for GPU acceleration)
- Rust (for building the tokenizer extension)

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

# Install the package and dependencies
uv pip install -e .

# Install development dependencies (optional)
uv pip install -e ".[dev]"

# Install the environment as a Jupyter kernel
python -m ipykernel install --user --name=minichat --display-name="minichat"
```

#### 4. Install Rust and Build Tokenizer
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add maturin to the virtual environment
uv add maturin

# Build and install the Rust tokenizer extension
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
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
- [ ] Inference Engine
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