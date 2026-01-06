# Implementation Guide

**Status**: To be completed during Phase 4  
**Last Updated**: January 6, 2026

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Project Reproduction](#project-reproduction)
3. [Running the Baseline](#running-the-baseline)
4. [Implementation Steps](#implementation-steps)
5. [Performance Profiling](#performance-profiling)
6. [Troubleshooting](#troubleshooting)

---

## Environment Setup

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- cuDNN (optional, for optimized kernels)
- Triton (for kernel development)

### Step 1: Create Virtual Environment

```bash
cd /Users/ymlin/Downloads/003-Study/138-Projects/nano-vllm/nano-vllm-coconut
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install triton>=2.0.0
pip install numpy pandas matplotlib

# Development tools
pip install pytest pytest-cov black flake8 mypy

# Optional: performance profiling
pip install nsight-systems  # Requires manual installation
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import triton; print(f'Triton: {triton.__version__}')"
```

---

## Project Reproduction

### Step 1: Copy nano-vLLM Source

```bash
# From nano-vllm directory
cp -r ../nano-vllm/nanovllm ./nanovllm
cp ../nano-vllm/example.py ./
cp ../nano-vllm/bench.py ./
cp ../nano-vllm/README.md ./README_original.md
```

### Step 2: Verify Original Code Works

```bash
# Test original nano-vLLM
python example.py
```

**Expected Output**: [Describe expected output]

### Step 3: Create Directory Structure

```bash
# Create necessary directories
mkdir -p nanovllm/kernels
mkdir -p tests
mkdir -p benchmarks/bench_results
mkdir -p examples
mkdir -p docs
```

---

## Running the Baseline

### Step 1: Generate Baseline Metrics

```bash
# Run baseline benchmark
python benchmarks/bench_baseline.py --model llama-7b --batch_size 1 --seq_len 128
```

**Configuration**:
- Models: llama-7b, llama-13b, mistral-7b
- Batch sizes: 1, 4, 8, 16, 32
- Sequence lengths: 128, 256, 512, 1024, 2048

### Step 2: Store Results

```bash
# Results saved to
benchmarks/bench_results/baseline_metrics.csv
```

**Columns**:
- model_name
- batch_size
- seq_len
- latency_ms
- throughput_tokens_per_sec
- peak_memory_mb

---

## Implementation Steps

### Phase 1: Analysis (Weeks 1-6)

1. **Read Source Code**
   ```bash
   # Key files to read
   cat nanovllm/llm.py
   cat nanovllm/engine/*.py
   cat nanovllm/layers/*.py
   ```

2. **Document Architecture**
   - Complete `docs/01_nano_vllm_architecture.md`
   - Create code annotations
   - Build understanding document

3. **Baseline Profiling**
   ```bash
   # Use nsys to profile
   nsys profile -o baseline.nsys-rep python example.py
   ```

### Phase 2: Implementation (Weeks 7-14)

1. **Create Fusion Kernel**

```python
# File: nanovllm/kernels/rms_norm_linear_fusion.py

import triton
import triton.language as tl
import torch

@triton.jit
def rms_norm_linear_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    eps=1e-6,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_M: tl.constexpr = 32,
):
    """
    Fused RMSNorm + Linear kernel.
    
    x: (M, K) - input
    weight: (K, N) - linear weight
    bias: (N,) - linear bias
    output: (M, N) - output
    """
    # Implementation to be completed
    pass
```

2. **Implement Wrapper**

```python
def fused_rms_norm_linear(x, weight, bias, eps=1e-6):
    """Fused RMSNorm + Linear operation."""
    # Implementation
    pass
```

3. **Test Correctness**

```bash
# Run tests
python -m pytest tests/test_fusion_kernel.py -v
```

### Phase 3: Profiling (Weeks 15-18)

1. **Profile Baseline**
   ```bash
   nsys profile -o profiles/baseline.nsys-rep python benchmarks/bench_baseline.py
   ```

2. **Profile Optimized**
   ```bash
   nsys profile -o profiles/optimized.nsys-rep python benchmarks/bench_fusion_operator.py
   ```

3. **Analyze Results**
   ```bash
   # Compare metrics
   python scripts/compare_profiles.py profiles/baseline.nsys-rep profiles/optimized.nsys-rep
   ```

---

## Performance Profiling

### Using Nsight Systems

```bash
# Profile with Nsight Systems
nsys profile \
  --output=profile \
  --trace cuda,osrt \
  --gpu-metrics-device 0 \
  python benchmarks/bench_fusion_operator.py

# View results
nsys-ui profile.nsys-rep
```

### Using PyTorch Profiler

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    on_trace_ready=profiler.tensorboard_trace_handler('./logs'),
) as prof:
    output = fused_rms_norm_linear(x, weight, bias)
    prof.step()
```

### Metrics to Track

| Metric | Tool | Command |
|--------|------|---------|
| Latency | Nsight/Profiler | Profile inference |
| Throughput | Custom script | tokens/sec calculation |
| Memory | PyTorch | torch.cuda.memory_reserved() |
| Bandwidth | Nsight | GPU memory bandwidth |

---

## Running Benchmarks

### Benchmark 1: Latency vs Sequence Length

```bash
python benchmarks/bench_fusion_operator.py \
  --benchmark latency_vs_seqlen \
  --output results_latency.csv
```

### Benchmark 2: Throughput vs Batch Size

```bash
python benchmarks/bench_fusion_operator.py \
  --benchmark throughput_vs_batch \
  --output results_throughput.csv
```

### Benchmark 3: Memory Efficiency

```bash
python benchmarks/bench_fusion_operator.py \
  --benchmark memory_efficiency \
  --output results_memory.csv
```

---

## Troubleshooting

### Issue: CUDA out of memory

```bash
# Solution: Reduce batch size or sequence length
python example.py --batch_size 1 --seq_len 128
```

### Issue: Triton kernel compilation error

```bash
# Solution: Check Triton version and CUDA compatibility
python -c "import triton; print(triton.version)"
```

### Issue: Numerical mismatch

```bash
# Solution: Run numerical correctness test
python -m pytest tests/test_fusion_kernel.py::test_numerical_correctness -v
```

See [CHALLENGES_AND_SOLUTIONS.md](CHALLENGES_AND_SOLUTIONS.md) for more troubleshooting.

---

## Verification Checklist

Before moving to next phase:

- [ ] Environment setup complete
- [ ] Dependencies installed
- [ ] Original code reproduces
- [ ] Baseline metrics captured
- [ ] Tests pass
- [ ] Documentation updated

---

## Next Steps

1. Follow [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) for phase-by-phase details
2. Consult [docs/00_README.md](00_README.md) for documentation overview
3. Check [CHALLENGES_AND_SOLUTIONS.md](CHALLENGES_AND_SOLUTIONS.md) if issues arise

---

**Last Updated**: January 6, 2026  
**Estimated Setup Time**: 2-3 hours
