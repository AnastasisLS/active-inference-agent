"""
Device utilities for PyTorch operations.
Handles automatic device selection for CPU, GPU (CUDA), and Apple Silicon (MPS).
"""

import torch
from typing import Optional, Union, Any


def get_device(device_name: Optional[str] = None) -> torch.device:
    """
    Get the best available device for PyTorch operations.
    
    Priority order:
    1. CUDA (NVIDIA GPU) if available
    2. MPS (Apple Silicon GPU) if available
    3. CPU as fallback
    
    Args:
        device_name: Optional device name override ('cpu', 'cuda', 'mps')
        
    Returns:
        torch.device: The selected device
    """
    if device_name is not None:
        if device_name == 'cpu':
            return torch.device('cpu')
        elif device_name == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif device_name == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            print(f"Warning: Requested device '{device_name}' not available, falling back to best available device")
    
    # Auto-select best available device
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        dict: Information about available devices
    """
    info = {
        'cpu': True,
        'cuda': torch.cuda.is_available(),
        'mps': torch.backends.mps.is_available(),
        'current_device': None
    }
    
    if info['cuda']:
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_device_memory'] = torch.cuda.get_device_properties(0).total_memory
    
    if info['mps']:
        info['mps_built'] = torch.backends.mps.is_built()
    
    return info


def print_device_info():
    """
    Print information about available devices.
    """
    info = get_device_info()
    
    print("=" * 50)
    print("DEVICE INFORMATION")
    print("=" * 50)
    print(f"CPU: Available")
    print(f"CUDA: {'Available' if info['cuda'] else 'Not Available'}")
    if info['cuda']:
        print(f"  - Device Count: {info['cuda_device_count']}")
        print(f"  - Device Name: {info['cuda_device_name']}")
        print(f"  - Total Memory: {info['cuda_device_memory'] / 1024**3:.1f} GB")
    
    print(f"MPS (Apple Silicon): {'Available' if info['mps'] else 'Not Available'}")
    if info['mps']:
        print(f"  - Built: {info['mps_built']}")
    
    best_device = get_device()
    print(f"\nBest Available Device: {best_device}")
    print("=" * 50)


def to_device(data: Any, device: torch.device) -> Any:
    """
    Move data to the specified device.
    Handles tensors, models, and nested structures.
    
    Args:
        data: Data to move to device (tensor, model, or nested structure)
        device: Target device
        
    Returns:
        Data moved to the specified device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif hasattr(data, 'to'):
        # For models and other objects with .to() method
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(item, device) for item in data)
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    else:
        return data


def create_tensor_on_device(*args, device: torch.device, **kwargs) -> torch.Tensor:
    """
    Create a tensor directly on the specified device.
    
    Args:
        *args: Arguments to pass to torch.tensor
        device: Target device
        **kwargs: Keyword arguments to pass to torch.tensor
        
    Returns:
        torch.Tensor: Tensor created on the specified device
    """
    return torch.tensor(*args, device=device, **kwargs)


def get_device_memory_info(device: torch.device) -> dict:
    """
    Get memory information for the specified device.
    
    Args:
        device: Device to get memory info for
        
    Returns:
        dict: Memory information
    """
    if device.type == 'cuda':
        return {
            'total': torch.cuda.get_device_properties(device).total_memory,
            'allocated': torch.cuda.memory_allocated(device),
            'cached': torch.cuda.memory_reserved(device),
            'free': torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
        }
    elif device.type == 'mps':
        # MPS doesn't provide detailed memory info like CUDA
        return {
            'total': None,
            'allocated': None,
            'cached': None,
            'free': None,
            'note': 'MPS memory info not available'
        }
    else:
        return {
            'total': None,
            'allocated': None,
            'cached': None,
            'free': None,
            'note': 'CPU memory info not available'
        }


def print_memory_info(device: torch.device):
    """
    Print memory information for the specified device.
    
    Args:
        device: Device to print memory info for
    """
    info = get_device_memory_info(device)
    
    print(f"Memory Info for {device}:")
    if device.type == 'cuda':
        print(f"  Total: {info['total'] / 1024**3:.1f} GB")
        print(f"  Allocated: {info['allocated'] / 1024**3:.1f} GB")
        print(f"  Cached: {info['cached'] / 1024**3:.1f} GB")
        print(f"  Free: {info['free'] / 1024**3:.1f} GB")
    else:
        print(f"  {info['note']}")


# Global device variable for easy access
DEVICE = get_device()


def get_global_device() -> torch.device:
    """
    Get the global device that was automatically selected.
    
    Returns:
        torch.device: The global device
    """
    return DEVICE


def set_global_device(device_name: str):
    """
    Set the global device.
    
    Args:
        device_name: Device name ('cpu', 'cuda', 'mps')
    """
    global DEVICE
    DEVICE = get_device(device_name)
    print(f"Global device set to: {DEVICE}") 