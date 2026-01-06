# nano-vLLM Optimization: RMSNorm+Linear Fusion Research

> A research project implementing and optimizing the nano-vLLM lightweight inference framework with Triton-based kernel fusion.

## 📋 Project Overview

This project reproduces the **nano-vLLM** lightweight LLM inference framework and implements a **RMSNorm+Linear operator fusion** using Triton to achieve 10-15% latency reduction through reduced global memory I/O.

**Timeline**: April 2025 – September 2025 (6 months)

### Key Objectives

1. **Understand** nano-vLLM and vLLM architectures
   - PagedAttention scheduling mechanisms
   - Memory management strategies
   - Multi-task performance characteristics

2. **Implement** RMSNorm+Linear fusion kernel
   - Design in Triton for GPU optimization
   - Integrate into model layers
   - Validate numerical correctness

3. **Validate & Profile** performance improvements
   - Use Nsight Systems for detailed analysis
   - Benchmark across different sequence lengths and batch sizes
   - Document performance gains and insights

## 📁 Project Structure

```
├── IMPLEMENTATION_PLAN.md          # Detailed phase-by-phase plan
├── README.md                       # This file
├── docs/                           # Learning & analysis documents
│   ├── 00_README.md               # Documentation index
│   ├── 01_nano_vllm_architecture.md
│   ├── 02_memory_management.md
│   ├── 03_attention_analysis.md
│   ├── 04_baseline_metrics.md
│   ├── 05_nano_vs_vllm_comparison.md
│   ├── 06_fusion_design.md
│   ├── 07_kernel_optimization.md
│   ├── 08_validation_report.md
│   ├── 09_performance_analysis.md
│   ├── 10_benchmark_comparison.md
│   ├── CHALLENGES_AND_SOLUTIONS.md
│   └── IMPLEMENTATION_GUIDE.md
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

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd nano-vllm-coconut
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch triton numpy pandas matplotlib
```

### 2. Reproduce nano-vLLM

```bash
# Copy nano-vLLM code
cp -r ../nano-vllm/nanovllm ./nanovllm
cp -r ../nano-vllm/bench.py ./

# Run baseline
python example.py
```

### 3. Follow Implementation Plan

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed phases:

- **Phase 1 (Weeks 1-6)**: Source code analysis
- **Phase 2 (Weeks 7-14)**: Implement fusion kernel
- **Phase 3 (Weeks 15-18)**: Profile and optimize
- **Phase 4 (Weeks 19-22)**: Document findings
- **Phase 5 (Weeks 23-26)**: Polish and publish

## 📚 Learning Resources

### nano-vLLM
- Original repository: [/nano-vllm](../nano-vllm)
- Key files: `nanovllm/llm.py`, `nanovllm/engine/`, `nanovllm/layers/`

### vLLM (for comparison)
- Original repository: [/vllm](../vllm)
- PagedAttention: `vllm/attention/`
- Memory manager: `vllm/engine/memory_controller.py`

### Triton Kernels
- [Triton Documentation](https://triton-lang.org/)
- [Triton Examples](https://github.com/openai/triton/tree/main/python/examples)
- Key concepts: block-level parallelism, warp-level operations, memory coalescing

## 📊 Success Criteria

| Phase | Deliverable | Status |
|-------|-------------|--------|
| Phase 1 | 5 analysis documents + baselines | ⏳ Pending |
| Phase 2 | RMSNorm+Linear fusion kernel | ⏳ Pending |
| Phase 3 | Performance validation (10-15% gain) | ⏳ Pending |
| Phase 4 | Comprehensive documentation | ⏳ Pending |
| Phase 5 | Production-ready code & repo | ⏳ Pending |

## 🔍 Key Concepts to Master

### 1. Memory Management in LLMs
- **KV Cache**: Key-value cache storage and management
- **Paging**: Block-based memory allocation (nano-vLLM vs vLLM)
- **Memory Bandwidth**: How to minimize data movement

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
  x --→ RMSNorm+Linear --→ z (minimal global memory I/O)
```

**Expected gains**:
- Reduce intermediate tensor write/read: ~30% memory I/O reduction
- Target latency improvement: 10-15% (hardware dependent)

## 📝 Progress Tracking

Use this checklist to track overall progress:

- [ ] **Phase 1**: All 5 analysis documents completed
- [ ] **Phase 2**: Fusion kernel implemented and validated
- [ ] **Phase 3**: Performance targets achieved and profiled
- [ ] **Phase 4**: Documentation complete and polished
- [ ] **Phase 5**: Repository ready for publication

## 🤝 Contributing

This is a research project. If you discover improvements or alternative approaches:

1. Document your findings in `docs/`
2. Add test cases if implementing new features
3. Update `docs/CHALLENGES_AND_SOLUTIONS.md` with learnings
4. Commit with clear messages

## 📄 License

See LICENSE file (inherited from nano-vLLM and vLLM)

## 🔗 References

- nano-vLLM Repository: https://github.com/...
- vLLM Repository: https://github.com/lm-sys/vllm
- Triton: https://triton-lang.org/
- PagedAttention Paper: [PagedAttention paper link]

---

**Status**: Starting Phase 1 (January 2026)  
**Last Updated**: January 6, 2026
