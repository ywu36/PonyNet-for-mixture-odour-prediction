"""Pick the GPU to run on. This machine has two: RTX 2080 (8 GB, drives the display) and
GTX 1080 Ti (11 GB). Set OSMO_DEVICE (e.g. "cuda:0", "cpu") to override."""
import os


def pick_device(min_free_gb: float = 0.0) -> str:
    env = os.environ.get("OSMO_DEVICE")
    if env:
        return env
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    best = max(range(len(free)), key=free.__getitem__)       # most free memory
    if free[best] < min_free_gb * 2**30:
        raise RuntimeError(f"no GPU has {min_free_gb} GB free (best: {free[best] / 2**30:.1f} GB)")
    return f"cuda:{best}"
