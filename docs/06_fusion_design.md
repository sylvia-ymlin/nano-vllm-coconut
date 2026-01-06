# 6. RMSNorm+Linear Fusion Kernel Design

**Phase**: 2 (Week 7)  
**Status**: To be completed  
**Last Updated**: January 6, 2026

## Overview

Design specification for the RMSNorm+Linear fusion kernel. Includes algorithm design, mathematical formulation, memory optimization strategy, and parameter selection.

## Sections to Complete

### 6.1 Kernel Specifications
- [ ] Input/output shapes
- [ ] Data types supported
- [ ] Numerical requirements
- [ ] Edge cases

### 6.2 Algorithm Design
- [ ] RMSNorm computation
- [ ] Linear transformation
- [ ] Fused implementation strategy
- [ ] Data dependencies

### 6.3 Memory Optimization
- [ ] Global memory I/O analysis
- [ ] Memory access patterns
- [ ] Coalescing strategy
- [ ] Reduced I/O calculation

### 6.4 Computational Analysis
- [ ] FLOPs calculation
- [ ] Memory bandwidth requirements
- [ ] Compute intensity
- [ ] Roofline analysis

### 6.5 Parameter Selection
- [ ] Block size selection
- [ ] Tile sizes
- [ ] Register allocation
- [ ] Shared memory usage

## Mathematical Formulation

### RMSNorm
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot w$$

### Linear
$$y = xW + b$$

### Fused Operation
$$\text{output} = \text{RMSNorm}(x) \cdot W + b$$

## Kernel Signature

```python
@triton.jit
def rms_norm_linear_kernel(
    x_ptr,      # Input: (M, K)
    w_ptr,      # Weight: (K, N)
    b_ptr,      # Bias: (N,)
    out_ptr,    # Output: (M, N)
    M, K, N,    # Dimensions
    eps,        # Epsilon for normalization
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pass
```

## Memory I/O Analysis

### Without Fusion (Separate Operations)
```
Input read:           M × K × 4 bytes (fp32)
RMSNorm output write: M × K × 4 bytes
RMSNorm output read:  M × K × 4 bytes
Linear output write:  M × N × 4 bytes
Total I/O:            M × K × 12 bytes + M × N × 4 bytes
```

### With Fusion
```
Input read:           M × K × 4 bytes
Linear output write:  M × N × 4 bytes
Total I/O:            M × K × 4 + M × N × 4 bytes

Reduction: ~66% for typical dimensions (K ≈ N ≈ 4096)
```

## Design Decisions

### 1. Data Type Support
- [ ] fp32 (default precision)
- [ ] bf16 (reduced precision)
- [ ] fp16 (lowest precision)

### 2. Memory Layout
- [ ] Row-major (default)
- [ ] Column-major support
- [ ] Strided tensor handling

### 3. Batch Processing
- [ ] Process along M dimension
- [ ] Handle variable M
- [ ] Minimize synchronization

### 4. Numerical Stability
- [ ] Epsilon parameter selection
- [ ] Normalization approach
- [ ] Precision handling

## Performance Targets

### Expected Improvements
- Memory I/O reduction: ~60-70%
- Latency improvement: 10-15%
- Memory bandwidth improvement: ~20-30%

### Constraints
- No negative impact on accuracy
- Support all model architectures
- Handle all shape combinations
- Graceful fallback if not applicable

## Design Validation

### Pre-implementation Checklist
- [ ] Mathematical correctness verified
- [ ] Memory I/O reduction quantified
- [ ] Performance targets realistic
- [ ] All edge cases identified
- [ ] Integration points identified

## Preliminary Testing Strategy

### Unit Tests
- [ ] Numerical correctness vs reference
- [ ] Different data types
- [ ] Various input shapes
- [ ] Edge cases

### Integration Tests
- [ ] Layer replacement works
- [ ] Gradients flow correctly (if training)
- [ ] No performance regression
- [ ] Memory usage reduced

## Questions for Design

- [ ] What block size minimizes bank conflicts?
- [ ] What tile size maximizes cache utilization?
- [ ] How to handle non-aligned dimensions?
- [ ] How to minimize synchronization overhead?
- [ ] How to support different data types?

## References

### Internal
- [03_attention_analysis.md](03_attention_analysis.md)
- [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md)

### External
- Triton kernel design best practices
- CUDA memory optimization guide
- Roofline model analysis

## Next Steps

1. Verify mathematical formulation
2. Calculate expected memory I/O reduction
3. Select initial parameter values
4. Proceed to implementation (doc 07)

---

**Estimated Duration**: 1 week  
**Difficulty**: High  
**Dependencies**: Phase 1 (all documents)
