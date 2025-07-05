# CPU Migration Guide

This guide documents the changes made to enable CPU-only execution of the DIMOS project, removing the hard dependency on NVIDIA GPUs while maintaining backward compatibility.

## Overview

The DIMOS project previously required NVIDIA GPUs for all operations. This migration enables:

- **CPU-only execution** for all components
- **Automatic device detection** with graceful fallback
- **Environment-variable based configuration** for deployment flexibility
- **Backward compatibility** with existing GPU setups

## Changes Made

### 1. Dependencies

#### Modified `requirements.txt`
- **Removed**: `ctransformers[cuda]==0.2.27` (GPU-only version)
- **Added**: `ctransformers==0.2.27` (CPU-compatible version)
- **Commented out**: `xformers==0.0.20` (GPU-specific acceleration)

#### Created `requirements-gpu.txt`
- Contains GPU-specific dependencies that can be installed optionally
- Includes CUDA-enabled versions of libraries
- Used for GPU-accelerated deployments

### 2. Device Management

#### Created `dimos/utils/device_utils.py`
Central device management utility with the following functions:

- `get_device(preferred_device)`: Auto-detects or uses specified device
- `get_gpu_layers()`: Returns appropriate GPU layer count
- `is_cuda_available()`: Checks CUDA availability
- `get_torch_dtype(device)`: Returns appropriate data type

**Environment Variables:**
- `DIMOS_DEVICE`: Force specific device ("cpu" or "cuda")
- `DIMOS_GPU_LAYERS`: Override GPU layer count (0 for CPU-only)

### 3. Docker Configuration

#### GPU Configurations (existing)
- `docker/models/*/docker-compose.yml`: NVIDIA runtime required
- `docker/models/*/Dockerfile`: CUDA base images

#### CPU Configurations (new)
- `docker/models/*/docker-compose-cpu.yml`: CPU-only containers
- `docker/models/*/Dockerfile-cpu`: Standard Python base images
- Environment variables for CPU execution

### 4. Runner Script

#### Updated `run.sh`
- **Added CPU options**: `hf-local-cpu` (option 5) and `gguf-cpu` (option 6)
- **Clarified labels**: GPU vs CPU execution modes
- **New commands**:
  ```bash
  ./run.sh 5  # or ./run.sh hf-local-cpu
  ./run.sh 6  # or ./run.sh gguf-cpu
  ```

### 5. Agent Updates

#### `dimos/agents/agent_ctransformers_gguf.py`
- **Device detection**: Uses `get_device()` utility
- **GPU layers**: Respects `DIMOS_GPU_LAYERS` environment variable
- **Graceful fallback**: Automatically falls back to CPU if GPU unavailable

#### `dimos/agents/agent_huggingface_local.py`
- **Device detection**: Uses `get_device()` utility
- **Data types**: Automatic float32 (CPU) vs float16 (GPU)
- **Error handling**: Graceful fallback for device issues

### 6. Model Updates

#### `dimos/models/depth/metric3d.py`
- **Conditional CUDA**: Only calls `.cuda()` when GPU available
- **Device detection**: Uses centralized device utilities
- **CPU fallback**: Maintains full functionality on CPU

#### `dimos/perception/semantic_seg.py`
- **Auto device**: Defaults to "auto" instead of "cuda"
- **Device propagation**: Passes detected device to sub-components

### 7. Configuration Files

#### Environment Variables
Set these in your `.env` file or environment:

```bash
# Force CPU execution
DIMOS_DEVICE=cpu
DIMOS_GPU_LAYERS=0

# Force GPU execution (if available)
DIMOS_DEVICE=cuda
DIMOS_GPU_LAYERS=50

# Auto-detection (default)
# DIMOS_DEVICE=auto
# DIMOS_GPU_LAYERS=auto
```

## Usage

### CPU-Only Execution

#### Option 1: Environment Variables
```bash
export DIMOS_DEVICE=cpu
export DIMOS_GPU_LAYERS=0
./run.sh 2  # Uses CPU despite being "GPU" labeled
```

#### Option 2: CPU-Specific Containers
```bash
./run.sh 5  # HuggingFace Local (CPU)
./run.sh 6  # CTransformers GGUF (CPU)
```

#### Option 3: Manual Docker Commands
```bash
# CPU version
docker compose -f docker/models/huggingface_local/docker-compose-cpu.yml up

# GPU version (requires NVIDIA runtime)
docker compose -f docker/models/huggingface_local/docker-compose.yml up
```

### GPU Execution (Optional)

#### Install GPU Dependencies
```bash
pip install -r requirements-gpu.txt
```

#### Use GPU Containers
```bash
./run.sh 2  # HuggingFace Local (GPU)
./run.sh 4  # CTransformers GGUF (GPU)
```

## Performance Considerations

### CPU Performance
- **Slower inference**: 5-10x slower than GPU
- **Higher memory usage**: CPU models use more RAM
- **Better compatibility**: Runs on any system

### GPU Performance
- **Faster inference**: Leverages parallel processing
- **Lower memory usage**: More efficient memory management
- **Hardware dependency**: Requires NVIDIA GPU

### Recommendations

1. **Development**: Use CPU mode for development and testing
2. **Production**: Use GPU mode for production workloads
3. **Deployment**: Use CPU mode for broad compatibility
4. **Resource-constrained**: Use CPU mode with smaller models

## Migration Checklist

If migrating existing GPU-dependent code:

- [ ] Update imports to use `dimos.utils.device_utils`
- [ ] Replace hard-coded `device="cuda"` with `device="auto"`
- [ ] Remove hard-coded `.cuda()` calls
- [ ] Add device detection logic
- [ ] Test both CPU and GPU execution paths
- [ ] Update documentation and deployment scripts

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `dimos.utils.device_utils` is available
2. **CUDA errors**: Check CUDA installation and compatibility
3. **Memory errors**: Reduce model size or batch size for CPU
4. **Performance issues**: Use GPU mode for heavy workloads

### Environment Variables Not Working
```bash
# Check current values
echo $DIMOS_DEVICE
echo $DIMOS_GPU_LAYERS

# Set for current session
export DIMOS_DEVICE=cpu
export DIMOS_GPU_LAYERS=0
```

### Docker Issues
```bash
# Remove GPU runtime requirement
docker compose -f docker/models/*/docker-compose-cpu.yml up

# Check container logs
docker compose logs -f
```

## Backward Compatibility

All existing GPU-based configurations continue to work unchanged:
- Existing Docker configurations remain functional
- GPU detection still works automatically
- Performance characteristics are preserved
- No breaking changes to public APIs

## Future Enhancements

Potential improvements for CPU optimization:
- Model quantization for faster CPU inference
- Multi-threading optimizations
- Memory-mapped model loading
- CPU-specific acceleration libraries
- Dynamic batch size adjustment