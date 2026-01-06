# nano-vLLM Optimization: RMSNorm and Linear Fusion Research

A research project implementing and optimizing the nano-vLLM lightweight inference framework with Triton-based kernel fusion.

## Project Overview

This project reproduces the nano-vLLM lightweight LLM inference framework and implements RMSNorm and Linear operator fusion using Triton to achieve 10-15% latency reduction through reduced global memory I/O.

### Key Objectives

1. Understand nano-vLLM and vLLM architectures
   - PagedAttention scheduling mechanisms
   - Memory management strategies
   - Multi-task performance characteristics

2. Implement RMSNorm and Linear fusion kernel
   - Design in Triton for GPU optimization
   - Integrate into model layers
   - Validate numerical correctness

3. Validate and profile performance improvements
   - Use Nsight Systems for detailed analysis
   - Benchmark across different sequence lengths and batch sizes
   - Document performance gains and insights

## Project Structure

```
├── implementation_plan.md          # Detailed phase-by-phase plan
├── README.md                       # Project overview
├── docs/                           # Research and analysis documents
│   ├── 00_README.md               # Documentation index
│   ├── 01_architecture.md
│   ├── 02_memory_management.md
│   ├── 03_attention.md
│   ├── 04_baseline.md
│   ├── 05_comparison.md
│   ├── 06_fusion_design.md
│   ├── 07_kernel_optimization.md
│   ├── 08_validation.md
│   ├── 09_performance.md
│   ├── 10_benchmarks.md
│   ├── challenges.md
│   └── implementation_guide.md
├── nanovllm/                       # Modified nano-vLLM
│   ├── kernels/
│   │   └── rms_norm_linear_fusion.py
│   ├── engine/
│   ├── layers/
│   ├── models/
│   └── utils/
├── benchmarks/                     # Performance benchmarks
│   ├── bench_fusion_operator.py
│   └── bench_results/
├── tests/                          # Unit tests
│   ├── test_fusion_kernel.py
│   └── test_integration.py
└── examples/                       # Usage examples
    ├── fusion_inference.py
    └── baseline_inference.py
```

## Environment Setup

```bash
cd nano-vllm-coconut
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install torch triton numpy pandas matplotlib
```

### Implementation Phases

See [implementation_plan.md](implementation_plan.md) for detailed phases:

- Phase 1 (Weeks 1-6): Source code analysis
- Phase 2 (Weeks 7-14): Implement fusion kernel
- Phase 3 (Weeks 15-18): Profile and optimize
- Phase 4 (Weeks 19-22): Document findings
- Phase 5 (Weeks 23-26): Polish and publish

## Technical Background

### nano-vLLM
- Original repository: nano-vllm
- Key files: `nanovllm/llm.py`, `nanovllm/engine/`, `nanovllm/layers/`

### vLLM (for comparison)
- Original repository: vllm
- PagedAttention: `vllm/attention/`
- Memory manager: `vllm/engine/memory_controller.py`

### Triton Kernels
- Triton Documentation: https://triton-lang.org/
- Key concepts: block-level parallelism, warp-level operations, memory coalescing

## Research Objectives

| Phase | Deliverable | 
|-------|-------------|
| Phase 1 | Analysis documents and baselines |
| Phase 2 | RMSNorm and Linear fusion kernel |
| Phase 3 | Performance validation |
| Phase 4 | Comprehensive documentation |
| Phase 5 | Production-ready implementation |

## Key Technical Concepts

### 1. Memory Management in LLMs
- KV Cache: Key-value cache storage and management
- Paging: Block-based memory allocation
- Memory Bandwidth: Minimizing data movement

### 2. PagedAttention
- Token-to-block mapping
- Memory layout and access patterns
- Scheduling constraints

### 3. Kernel Optimization
- **RMSNorm**: Layer normalization variant
- **Fusion**: Combining separate operations to reduce memory I/O
- **Triton Programming**: GPU programming in Python-like syntax

### 4. Performance Profiling
- **Nsight Systems**: Timeline visualization and bottleneck analysis
- **Metrics**: Latency, throughput, memory bandwidth, utilization

## 💡 Implementation Highlights

### RMSNorm+Linear Fusion Strategy

**Goal**: Reduce memory I/O by fusing two commonly paired operations

```
Before (separate operations):
  x --→ RMSNorm --→ y (write to global memory)
  y --→ Linear --→ z (read from global memory)

After (fused kernel):
  x → RMSNorm and Linear → z (minimal global memory I/O)
```

Expected performance gains:
- Reduce intermediate tensor write/read: approximately 30% memory I/O reduction
- Target latency improvement: 10-15% (hardware dependent)

## References

- nano-vLLM Repository: https://github.com/nanovllm/nano-vllm
- vLLM Repository: https://github.com/vllm-project/vllm
- Triton: https://triton-lang.org/
- PagedAttention Paper: Efficient Memory Management for Large Language Model Serving with PagedAttention
