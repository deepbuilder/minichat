import os

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