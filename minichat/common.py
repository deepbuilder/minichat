import os
import fcntl
import urllib.request
import tempfile
import zipfile
import torch
import random
import shutil
import logging

class ColoredFormatter(logging.Formatter):
    """Logging Formatter to add colors based on log level."""
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    RESET = '\033[0m'
    BOLD = '\033[1m'
    COLORS = {
        'DEBUG': grey,
        'INFO': grey,
        'WARNING': yellow,
        'ERROR': red,
        'CRITICAL': bold_red
    }
    def  format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            levelname_color = self.COLORS[levelname] + levelname + self.reset
            record.levelname = levelname_color
        message = super().format(record)
        if levelname == 'INFO':
            # highlight numbers and percentages in info messages
            import re
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)   
        return message

def is_ddp():
    return int(os.environ.get("RANK", "-1")) != -1

def get_ddp_info():
    if is_ddp():
        assert "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ, "RANK, WORLD_SIZE, and LOCAL_RANK must be set in DDP mode"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        return True, rank, local_rank, world_size
    else:
        return False, 0, 0, 1

def get_base_dir():
    home_dir = os.path.expanduser("/home/shadeform/minichat")
    base_dir = os.path.join(home_dir, 'minichat')
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with open(lock_path, 'w', encoding='utf-8') as lock_file:

        # Only a single rank can acquire this lock
        # All other ranks block until it is released
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        # Recheck after acquiring lock (another process may have downloaded it)
        if os.path.exists(file_path):
            return file_path

        # Download the content as bytes
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read() # bytes

        # Write to local file
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    # Clean up the lock file after the lock is released
    try:
        os.remove(lock_path)
    except OSError:
        pass  # Ignore if already removed by another process


def place_eval_bundle(file_path):
    # here file_path is the path to the eval_bundle.zip file
    # we need to unzip it and place it in the base directory
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print(f"Placed eval_bundle directory at {eval_bundle_dir}")


def is_ddp():
    return int(os.environ.get("RANK", "-1")) != -1

def get_dist_info():
    if is_ddp():
        assert "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ, "RANK, WORLD_SIZE, and LOCAL_RANK must be set in DDP mode"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        return True, rank, local_rank, world_size
    else:
        return False, 0, 0, 1

def compute_init():
    assert torch.cuda.is_available(), "CUDA must be available for compute initialization"
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('high')
    ddp, rank, local_rank, world_size = get_dist_info()
    print(f"DDP: {ddp}, Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
    device = torch.device("cuda", local_rank)
    if ddp:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        torch.distributed.barrier()
        print(f"Initialized DDP: rank {rank}, local_rank {local_rank}, world_size {world_size}")
    return ddp, rank, local_rank, world_size, device

def compute_cleanup():
    ddp, rank, local_rank, world_size = get_dist_info()
    if ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

class DummyWandb:
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass


def setup_logger():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])

def find_rank():
    return int(os.environ.get("RANK", "0"))

def print_log(s="",**kwargs):
    if find_rank() == 0:
        print(s, **kwargs)