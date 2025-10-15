# minichat

The minimal full-stack ChatGPT clone

## Quick Setup

### Poetry Environment + Jupyter Setup

```bash
# Install dependencies using Poetry
poetry install

# Install the Poetry environment as a Jupyter kernel
poetry run python -m ipykernel install --user --name=minichat --display-name="minichat"
```

### Alternative one-liner setup:
```bash
# Quick setup in one command
poetry install && poetry run python -m ipykernel install --user --name=minichat --display-name="minichat"
```

### Using with VS Code Notebooks

1. Open your notebook in VS Code
2. Click on the kernel selector (top-right of notebook)
3. Choose "minichat" from the kernel list
4. The notebook will now use the Poetry environment with all dependencies

### Using with Jupyter Lab

```bash
# Start Jupyter Lab with Poetry environment
poetry run jupyter lab
```

### Verification

Verify the setup is working correctly:

```bash
poetry run python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### Optional: Activate Poetry Shell

```bash
# Activate the Poetry shell for direct command line usage
poetry shell
```

## Development

Once setup is complete, you can run the notebook `minichat/gpt.ipynb` with all dependencies available in the Poetry environment.
