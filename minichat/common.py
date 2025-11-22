import os
import fcntl
import urllib.request
import tempfile
import zipfile
import torch
import random
import shutil

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