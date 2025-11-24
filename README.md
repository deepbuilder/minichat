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


## Testing the run
```bash
# Activate the virtual environment if not already active
source .venv/bin/activate  # On macOS/Linux

# Download the dataset
python minichat/dataset.py -w 10

# Train the tokenizer
python scripts/tok_train.py

```

## Things to add
- Cleanup Eval script - Done
- Add Wandb logging
- Add Gradient accumulation - Done
- Add optimizer - Done 
- Add LR scheduler - Done
- Add Mixed precision training
- Add Inference engine
- Use Muon optimizer
- DDP support
- Training on full dataset