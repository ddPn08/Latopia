import torch
import torch.backends.mps


def get_optimal_torch_device(index: int = 0):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index % torch.cuda.device_count()}")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def half_supported(device: torch.device):
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 5.3
