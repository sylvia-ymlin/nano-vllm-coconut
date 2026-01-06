# 7. Kernel Optimization & Implementation Details

**Phase**: 2 (Weeks 8-11)  
**Status**: To be completed  
**Last Updated**: January 6, 2026

## Overview

Detailed implementation notes for the RMSNorm+Linear fusion kernel in Triton, including optimization techniques, data type handling, and batch processing strategies.

## Sections to Complete

### 7.1 Triton Kernel Implementation
- [ ] Basic kernel structure
- [ ] Memory layout and coalescing
- [ ] Computation pipeline
- [ ] Synchronization and barriers

### 7.2 Memory Optimization
- [ ] Global memory access patterns
- [ ] Shared memory utilization
- [ ] Register allocation
- [ ] Bank conflict avoidance

### 7.3 Data Type Support
- [ ] fp32 implementation
- [ ] bf16 implementation
- [ ] fp16 implementation
- [ ] Type casting and conversion

### 7.4 Batch Dimension Handling
- [ ] Single batch processing
- [ ] Multi-batch processing
- [ ] Variable batch sizes
- [ ] Load balancing

### 7.5 Performance Tuning
- [ ] Block size selection
- [ ] Grid size calculation
- [ ] Loop unrolling
- [ ] Instruction pipelining

### 7.6 Integration & Wrapper
- [ ] PyTorch wrapper function
- [ ] Input validation
- [ ] Output shape handling
- [ ] Error handling

## Triton Kernel Template

```python
import triton
import triton.language as tl

@triton.jit
def rms_norm_linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, K, N,
    eps: tl.float32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused RMSNorm + Linear kernel.
    
    Args:
        x_ptr: Input tensor (M, K)
        w_ptr: Weight matrix (K, N)
        b_ptr: Bias vector (N,)
        out_ptr: Output tensor (M, N)
        M, K, N: Dimensions
        eps: Epsilon for normalization
        BLOCK_M, BLOCK_N, BLOCK_K: Block sizes
    """
    # Grid and thread indexing
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Tile offsets
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offset = tl.arange(0, BLOCK_K)
    
    # Implementation to be completed
    pass
```

## Implementation Phases

### Phase 1: Basic Implementation (Weeks 8-9)
- [ ] Simple version without optimization
- [ ] Verify numerical correctness
- [ ] Test against reference implementation
- [ ] Handle basic edge cases

### Phase 2: Optimization (Weeks 9-10)
- [ ] Memory coalescing
- [ ] Cache optimization
- [ ] Synchronization reduction
- [ ] Performance tuning

### Phase 3: Robustness (Week 11)
- [ ] Data type support
- [ ] Variable shapes
- [ ] Edge cases
- [ ] Error handling

## Memory Access Pattern Analysis

### Optimal Coalescing
```
Global Memory Access:
- Row-major reads from x
- Row-major reads from w (transposed indexing)
- Sequential writes to out

Shared Memory:
- Tile for x: BLOCK_M × BLOCK_K
- Tile for w: BLOCK_K × BLOCK_N
```

### Bank Conflict Avoidance
- [ ] Shared memory layout optimization
- [ ] Padding strategies
- [ ] Access pattern verification

## Data Type Implementations

### float32
```python
# Full precision computation
x_data = tl.load(x_ptr, ...)  # float32
```

### bfloat16
```python
# Reduced precision
x_data = tl.load(x_ptr, ..., dtype=tl.bfloat16)
```

### float16
```python
# Ultra-reduced precision
x_data = tl.load(x_ptr, ..., dtype=tl.float16)
```

## Block Size Selection Guide

| Dimension | Batch Size | Seq Len | Hidden Dim | Recommended BLOCK Size |
|-----------|-----------|---------|-----------|------------------------|
| M (batch) | 1 | N/A | N/A | 1-4 |
| M (batch) | 4-8 | N/A | N/A | 4-8 |
| M (batch) | 16+ | N/A | N/A | 8-16 |
| N (output) | N/A | N/A | 768 | 32-64 |
| N (output) | N/A | N/A | 1024 | 64-128 |
| N (output) | N/A | N/A | 4096 | 128-256 |
| K (input) | N/A | N/A | 768 | 32-64 |

## Performance Profiling Points

```python
# Profile key operations
with triton.kernel_profile():
    output = fused_rms_norm_linear(x, w, b)
    
# Measure:
# - Kernel execution time
# - Memory bandwidth utilization
# - GPU utilization
# - Register pressure
```

## Debugging Techniques

### Assertion Checks
```python
# Verify shape assumptions
tl.static_assert(M % BLOCK_M == 0, "M must be divisible by BLOCK_M")
tl.static_assert(N % BLOCK_N == 0, "N must be divisible by BLOCK_N")
```

### Print Debugging
```python
# Debug values in kernel
tl.debug_barrier()
tl.printf("Value: %f", value)
```

### Verification Kernels
```python
# Reference implementation for testing
def reference_rms_norm_linear(x, w, b, eps=1e-6):
    normalized = x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return normalized @ w + b
```

## Common Optimization Patterns

### Pattern 1: Loop Unrolling
```python
# Unroll inner loops for better performance
for k in tl.range(0, K, BLOCK_K):
    # Process block
    pass
```

### Pattern 2: Pipeline Parallelism
```python
# Overlap computation with memory access
# Load next tile while computing current
```

### Pattern 3: Reduced Synchronization
```python
# Minimize __syncthreads() calls
# Use appropriate memory barriers
```

## Integration with PyTorch

```python
def fused_rms_norm_linear(x, weight, bias, eps=1e-6):
    """
    Fused RMSNorm + Linear wrapper for PyTorch.
    
    Args:
        x: Input tensor (batch, seq_len, hidden_dim)
        weight: Weight matrix (hidden_dim, hidden_dim)
        bias: Bias vector (hidden_dim,)
        eps: Epsilon for normalization
        
    Returns:
        Output tensor (batch, seq_len, hidden_dim)
    """
    # Input validation
    assert x.dim() == 3, "Input must be 3D"
    assert weight.shape == (x.shape[-1], x.shape[-1])
    
    # Reshape for kernel
    M, K = x.shape[0] * x.shape[1], x.shape[2]
    N = weight.shape[1]
    x_reshaped = x.reshape(M, K)
    
    # Allocate output
    output = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    grid = (M // BLOCK_M, N // BLOCK_N)
    rms_norm_linear_kernel[grid](
        x_reshaped, weight.t(), bias, output,
        M, K, N, eps,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    
    # Reshape back
    return output.reshape(x.shape[0], x.shape[1], N)
```

## Questions for Implementation

- [ ] What is the optimal BLOCK_M for different batch sizes?
- [ ] How to handle non-divisible dimensions?
- [ ] What is the best shared memory layout?
- [ ] How much register pressure is acceptable?
- [ ] How to minimize bank conflicts?

## References

### Internal
- [06_fusion_design.md](06_fusion_design.md)
- [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md)

### External
- Triton documentation: https://triton-lang.org/
- CUDA best practices guide
- GPU kernel optimization techniques

## Next Steps

1. Implement basic kernel version
2. Test against reference implementation
3. Optimize memory access patterns
4. Add data type support
5. Proceed to validation (doc 08)

---

**Estimated Duration**: 4 weeks  
**Difficulty**: Very High  
**Dependencies**: Phase 2.1 (design complete)
