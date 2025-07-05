"""
Device utilities for CPU/GPU detection and fallback
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def get_device(preferred_device: Optional[str] = None) -> str:
    """
    Get the appropriate device for computation (CPU or CUDA).
    
    Args:
        preferred_device: The preferred device ('cpu', 'cuda', or None for auto-detection)
    
    Returns:
        Device string ('cpu' or 'cuda')
    """
    # Check environment variable first
    env_device = os.getenv('DIMOS_DEVICE', '').lower()
    if env_device in ['cpu', 'cuda']:
        logger.info(f"Using device from environment: {env_device}")
        return env_device
    
    # Use preferred device if specified
    if preferred_device is not None:
        if preferred_device.lower() == 'cpu':
            logger.info("Using CPU as specified")
            return 'cpu'
        elif preferred_device.lower() == 'cuda':
            # Check if CUDA is available
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("Using CUDA as specified")
                    return 'cuda'
                else:
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    return 'cpu'
            except ImportError:
                logger.warning("PyTorch not available, falling back to CPU")
                return 'cpu'
    
    # Auto-detect device
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA available, using GPU")
            return 'cuda'
        else:
            logger.info("CUDA not available, using CPU")
            return 'cpu'
    except ImportError:
        logger.info("PyTorch not available, using CPU")
        return 'cpu'

def get_gpu_layers() -> int:
    """
    Get the number of GPU layers to use for models.
    
    Returns:
        Number of GPU layers (0 for CPU-only)
    """
    # Check environment variable first
    env_layers = os.getenv('DIMOS_GPU_LAYERS')
    if env_layers is not None:
        try:
            layers = int(env_layers)
            logger.info(f"Using GPU layers from environment: {layers}")
            return layers
        except ValueError:
            logger.warning(f"Invalid DIMOS_GPU_LAYERS value: {env_layers}")
    
    # Auto-detect based on device
    device = get_device()
    if device == 'cuda':
        logger.info("Using default GPU layers: 50")
        return 50
    else:
        logger.info("Using CPU-only (0 GPU layers)")
        return 0

def is_cuda_available() -> bool:
    """
    Check if CUDA is available.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def get_torch_dtype(device: str):
    """
    Get the appropriate torch dtype for the device.
    
    Args:
        device: Device string ('cpu' or 'cuda')
    
    Returns:
        torch.dtype
    """
    try:
        import torch
        if device == 'cuda':
            return torch.float16
        else:
            return torch.float32
    except ImportError:
        # Return string representation if torch not available
        return 'float32' if device == 'cpu' else 'float16'