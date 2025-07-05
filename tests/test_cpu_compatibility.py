#!/usr/bin/env python3
"""
Test script to verify CPU compatibility and device detection functionality.
"""

import os
import sys
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dimos.utils.device_utils import get_device, get_gpu_layers, is_cuda_available, get_torch_dtype

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_device_detection():
    """Test device detection functionality"""
    logger.info("=== Testing Device Detection ===")
    
    # Test auto-detection
    device = get_device()
    logger.info(f"Auto-detected device: {device}")
    
    # Test forced CPU
    cpu_device = get_device("cpu")
    logger.info(f"Forced CPU device: {cpu_device}")
    assert cpu_device == "cpu", "CPU device selection failed"
    
    # Test forced CUDA (should fallback to CPU if not available)
    cuda_device = get_device("cuda")
    logger.info(f"Requested CUDA device: {cuda_device}")
    
    # Test CUDA availability
    cuda_available = is_cuda_available()
    logger.info(f"CUDA available: {cuda_available}")
    
    return True

def test_gpu_layers():
    """Test GPU layer configuration"""
    logger.info("=== Testing GPU Layer Configuration ===")
    
    # Test default behavior
    layers = get_gpu_layers()
    logger.info(f"Default GPU layers: {layers}")
    
    # Test environment variable override
    os.environ['DIMOS_GPU_LAYERS'] = '10'
    env_layers = get_gpu_layers()
    logger.info(f"Environment-controlled GPU layers: {env_layers}")
    assert env_layers == 10, "Environment variable override failed"
    
    # Test CPU-only mode
    os.environ['DIMOS_DEVICE'] = 'cpu'
    cpu_layers = get_gpu_layers()
    logger.info(f"CPU-only GPU layers: {cpu_layers}")
    
    # Cleanup
    del os.environ['DIMOS_GPU_LAYERS']
    del os.environ['DIMOS_DEVICE']
    
    return True

def test_torch_dtype():
    """Test torch dtype selection"""
    logger.info("=== Testing Torch Dtype Selection ===")
    
    # Test CPU dtype
    cpu_dtype = get_torch_dtype('cpu')
    logger.info(f"CPU dtype: {cpu_dtype}")
    
    # Test CUDA dtype
    cuda_dtype = get_torch_dtype('cuda')
    logger.info(f"CUDA dtype: {cuda_dtype}")
    
    return True

def test_environment_variables():
    """Test environment variable functionality"""
    logger.info("=== Testing Environment Variables ===")
    
    # Test DIMOS_DEVICE
    os.environ['DIMOS_DEVICE'] = 'cpu'
    device = get_device()
    logger.info(f"Environment-forced device: {device}")
    assert device == 'cpu', "DIMOS_DEVICE environment variable not respected"
    
    # Test DIMOS_GPU_LAYERS
    os.environ['DIMOS_GPU_LAYERS'] = '25'
    layers = get_gpu_layers()
    logger.info(f"Environment-controlled layers: {layers}")
    assert layers == 25, "DIMOS_GPU_LAYERS environment variable not respected"
    
    # Cleanup
    del os.environ['DIMOS_DEVICE']
    del os.environ['DIMOS_GPU_LAYERS']
    
    return True

def test_import_agents():
    """Test that agents can be imported without GPU dependencies"""
    logger.info("=== Testing Agent Imports ===")
    
    try:
        # Set CPU-only mode
        os.environ['DIMOS_DEVICE'] = 'cpu'
        os.environ['DIMOS_GPU_LAYERS'] = '0'
        
        # Try importing agents
        from dimos.agents.agent_ctransformers_gguf import CTransformersGGUFAgent
        logger.info("✓ CTransformersGGUFAgent imported successfully")
        
        from dimos.agents.agent_huggingface_local import HuggingFaceLocalAgent
        logger.info("✓ HuggingFaceLocalAgent imported successfully")
        
        # Test device utilities import
        from dimos.utils.device_utils import get_device as test_get_device
        logger.info("✓ Device utilities imported successfully")
        
        # Cleanup
        del os.environ['DIMOS_DEVICE']
        del os.environ['DIMOS_GPU_LAYERS']
        
        return True
        
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting CPU compatibility tests...")
    
    tests = [
        test_device_detection,
        test_gpu_layers,
        test_torch_dtype,
        test_environment_variables,
        test_import_agents
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                logger.info(f"✓ {test.__name__} PASSED")
                passed += 1
            else:
                logger.error(f"✗ {test.__name__} FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"✗ {test.__name__} FAILED with exception: {e}")
            failed += 1
    
    logger.info(f"\n=== Test Results ===")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed! CPU compatibility is working correctly.")
        return 0
    else:
        logger.error(f"❌ {failed} tests failed. Please check the configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())