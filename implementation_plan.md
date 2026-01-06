# nano-vLLM Optimization Project - Implementation Plan

**Duration**: 2025.04 – 2025.09 (6 months)  
**Location**: `/Users/ymlin/Downloads/003-Study/138-Projects/nano-vllm/nano-vllm-coconut`

## Project Overview

Reproduce and extend the nano-vLLM lightweight inference framework with a focus on:
1. **Understanding** PagedAttention scheduling, memory management, and multi-task performance
2. **Implementing** RMSNorm+Linear fusion operator using Triton (target: 10-15% latency reduction)
3. **Validating** improvements through performance profiling with Nsight Systems

---

## Phase 1: Source Code Analysis & Understanding (Weeks 1-6)

### 1.1 nano-vLLM Core Study
**Goal**: Understand the lightweight inference framework architecture

- [ ] **Week 1-2: Project Structure**
  - [ ] Read [nano-vllm/README.md](../nano-vllm/README.md) and understand design philosophy
  - [ ] Map directory structure: `engine/`, `layers/`, `models/`, `utils/`
  - [ ] Identify entry points: `llm.py`, `config.py`, `sampling_params.py`
  - [ ] Document: `docs/01_nano_vllm_architecture.md`

- [ ] **Week 2-3: Memory Management**
  - [ ] Study KV cache management in `engine/`
  - [ ] Understand paging mechanism vs vLLM's PagedAttention
  - [ ] Analyze memory allocation patterns in `layers/`
  - [ ] Identify optimization opportunities
  - [ ] Document: `docs/02_memory_management.md`

- [ ] **Week 3-4: Attention Mechanism**
  - [ ] Trace attention computation flow
  - [ ] Understand current attention implementation (vanilla vs optimized)
  - [ ] Study RMSNorm layers in model implementations
  - [ ] Identify Linear layers paired with RMSNorm
  - [ ] Document: `docs/03_attention_analysis.md`

- [ ] **Week 4-5: Multi-task Performance**
  - [ ] Run nano-vLLM with different batch sizes
  - [ ] Measure baseline latency, throughput, memory usage
  - [ ] Test with different sequence lengths (128, 512, 2048, etc.)
  - [ ] Document: `docs/04_baseline_metrics.md`

- [ ] **Week 5-6: Comparative Study with vLLM**
  - [ ] Read vLLM's PagedAttention implementation
  - [ ] Compare scheduler designs
  - [ ] Analyze performance gaps
  - [ ] Document: `docs/05_nano_vs_vllm_comparison.md`

### 1.2 Deliverables
- [ ] 5 analysis documents
- [ ] Baseline performance metrics (CSV/JSON format)
- [ ] Code annotation/comments in key files
- [ ] Presentation slides: "nano-vLLM Architecture Deep Dive"

---

## Phase 2: RMSNorm+Linear Fusion Operator (Weeks 7-14)

### 2.1 Kernel Design & Planning (Week 7)
- [ ] Identify all RMSNorm+Linear patterns in models
  - [ ] Count occurrences
  - [ ] Analyze tensor shapes
  - [ ] Estimate memory I/O reduction (theory)
- [ ] Design Triton kernel signature
  - [ ] Input shapes and data types
  - [ ] Block sizes and optimization strategy
  - [ ] Memory coalescing patterns
- [ ] Document: `docs/06_fusion_design.md`

### 2.2 Triton Implementation (Weeks 8-11)

**Week 8-9: Basic Kernel**
- [ ] Implement basic RMSNorm+Linear in Triton
  - [ ] [ ] RMSNorm: $\text{out} = x / \sqrt{\text{mean}(x^2) + \epsilon} \cdot w$
  - [ ] [ ] Linear: $y = xW + b$
  - [ ] [ ] Fused: Single pass, minimal global memory writes
- [ ] Test correctness against PyTorch reference
  - [ ] Numerical accuracy (fp32, bf16)
  - [ ] Different input shapes
- [ ] Benchmark vs separate kernels

**Week 9-10: Optimization**
- [ ] Memory coalescing optimization
- [ ] Block size tuning (grid/block for different shapes)
- [ ] Data type support (bf16, fp16, fp32)
- [ ] Batch dimension handling
- [ ] Document: `docs/07_kernel_optimization.md`

**Week 11: Integration**
- [ ] Integrate into nano-vLLM model layers
- [ ] Auto-detect fusible RMSNorm+Linear pairs
- [ ] Fallback to original implementation if needed
- [ ] Create configuration option to enable/disable

### 2.3 Validation (Week 12)
- [ ] End-to-end correctness tests
  - [ ] Compare outputs with reference implementation (tolerance: 1e-5)
  - [ ] Multiple model architectures
  - [ ] Various batch sizes and sequence lengths
- [ ] Unit tests
- [ ] Document: `docs/08_validation_report.md`

### 2.4 Challenges & Solutions
Create `docs/challenges.md` to document:
- [ ] Memory layout complexities
- [ ] Triton block size selection
- [ ] Data type casting edge cases
- [ ] Integration with existing code

### 2.5 Deliverables
- [ ] `nanovllm/kernels/rms_norm_linear_fusion.py` (Triton kernel)
- [ ] Unit tests in `tests/test_fusion_kernel.py`
- [ ] Integration examples
- [ ] Performance comparison report

---

## Phase 3: Performance Profiling & Optimization (Weeks 13-18)

### 3.1 Nsight Systems Profiling (Weeks 13-14)
- [ ] Install and setup Nsight Systems
- [ ] Profile baseline nano-vLLM
  - [ ] Identify bottlenecks
  - [ ] Measure kernel execution time
  - [ ] Analyze memory bandwidth utilization
  - [ ] Create baseline timeline report
- [ ] Profile optimized version
  - [ ] Compare kernel time
  - [ ] Measure memory bandwidth improvement
  - [ ] Identify secondary bottlenecks

### 3.2 Performance Evaluation (Weeks 15-16)
- [ ] Create benchmark suite: `bench_fusion_operator.py`
  - [ ] **Scenario 1**: Fixed model, varying sequence lengths (128, 256, 512, 1024, 2048, 4096)
  - [ ] **Scenario 2**: Fixed sequence length, varying batch sizes (1, 4, 8, 16, 32, 64)
  - [ ] **Scenario 3**: End-to-end inference latency
  - [ ] **Scenario 4**: Throughput (tokens/sec)
  - [ ] **Scenario 5**: Memory efficiency (peak memory usage)

- [ ] Generate performance reports
  - [ ] Latency reduction: target 10-15%
  - [ ] Throughput improvement: measure actual
  - [ ] Memory savings: measure actual
  - [ ] Power efficiency: if available

- [ ] Document: `docs/09_performance_analysis.md`

### 3.3 Comparative Analysis (Week 17)
- [ ] Compare nano-vLLM (optimized) vs vLLM on same hardware
  - [ ] Same models
  - [ ] Same test scenarios
  - [ ] Identify strengths/weaknesses
- [ ] Document: `docs/10_benchmark_comparison.md`

### 3.4 Deliverables
- [ ] Nsight Systems timeline profiles (HTML/screenshots)
- [ ] `benchmarks/bench_fusion_operator.py`
- [ ] Performance metrics (CSV format for easy plotting)
- [ ] Analysis plots (latency, throughput vs sequence length/batch size)

---

## Phase 4: Documentation & Knowledge Transfer (Weeks 19-22)

### 4.1 Learning Documentation
- [ ] Consolidate all `docs/` files
- [ ] Create main guide: `docs/00_README.md`
- [ ] Structure:
  - Architecture overview
  - Memory model explanation
  - Attention mechanisms
  - Fusion operator design rationale
  - Performance results and insights

### 4.2 Implementation Guide
- [ ] `docs/IMPLEMENTATION_GUIDE.md`
  - Step-by-step reproduction instructions
  - Environment setup
  - Building and testing
  - Performance profiling steps

### 4.3 Challenges & Solutions Document
- [ ] `docs/CHALLENGES_AND_SOLUTIONS.md`
  - Problems encountered
  - Root causes
  - Solutions implemented
  - Alternative approaches considered

### 4.4 Presentation Materials
- [ ] Slide deck (10-15 slides)
  - Project motivation and goals
  - Architecture analysis findings
  - Fusion operator design
  - Performance results
  - Key learnings and insights
- [ ] Demo script (live or video)

### 4.5 Deliverables
- [ ] Comprehensive documentation (8-10 pages)
- [ ] Presentation slides
- [ ] Code comments and docstrings
- [ ] README updates

---

## Phase 5: Polish & Publication (Weeks 23-26)

### 5.1 Code Quality
- [ ] Add docstrings to all functions
- [ ] Add type hints
- [ ] Code formatting and linting
- [ ] Create example scripts in `examples/`

### 5.2 Repository Organization
- [ ] Clean directory structure
- [ ] Add LICENSE (copy from original projects)
- [ ] Update main README
- [ ] Create CONTRIBUTING guide if applicable

### 5.3 Final Testing
- [ ] Regression tests
- [ ] Documentation accuracy check
- [ ] Run all benchmarks one final time

### 5.4 Deliverables
- [ ] Production-ready code
- [ ] Final documentation
- [ ] Benchmark results
- [ ] GitHub-ready repository

---

## Directory Structure

```
nano-vllm-coconut/
├── README.md                          # Project overview
├── IMPLEMENTATION_PLAN.md             # This file
├── docs/                              # Learning & analysis documents
│   ├── 00_README.md                   # Documentation index
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
│   ├── IMPLEMENTATION_GUIDE.md
│   └── presentations/
│       └── nano_vllm_optimization.pptx
├── nanovllm/                          # Modified nano-vLLM with fusion
│   ├── __init__.py
│   ├── config.py
│   ├── llm.py
│   ├── sampling_params.py
│   ├── kernels/                       # NEW: Custom kernels
│   │   ├── __init__.py
│   │   └── rms_norm_linear_fusion.py  # Triton fusion kernel
│   ├── engine/
│   ├── layers/
│   ├── models/
│   └── utils/
├── tests/                             # Unit tests
│   ├── test_fusion_kernel.py
│   └── test_integration.py
├── benchmarks/                        # Performance benchmarks
│   ├── bench_fusion_operator.py       # NEW: Fusion operator benchmark
│   ├── bench_baseline.py
│   ├── bench_results/                 # Output directory
│   │   ├── metrics.csv
│   │   └── plots/
│   └── nsight_profiles/               # Nsight Systems outputs
├── examples/                          # Usage examples
│   ├── fusion_inference.py            # NEW: Example using fusion
│   └── baseline_inference.py
└── requirements.txt                   # Project dependencies
```

---

## Success Metrics

### Phase 1: Understanding
- ✅ 5 detailed analysis documents (>2 pages each)
- ✅ Baseline metrics captured and validated
- ✅ Code annotated with key insights

### Phase 2: Implementation
- ✅ RMSNorm+Linear fusion kernel implemented
- ✅ Numerical correctness validated (error < 1e-5)
- ✅ All edge cases handled (bf16, fp16, various shapes)

### Phase 3: Performance
- ✅ Latency reduction: 10-15% achieved
- ✅ Memory bandwidth improvement measured
- ✅ Nsight profile analysis completed
- ✅ Benchmarks for various scenarios documented

### Phase 4-5: Documentation
- ✅ Comprehensive documentation (>15 pages)
- ✅ Reproducible setup from scratch
- ✅ Code production-ready
- ✅ GitHub repository clean and organized

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Triton kernel development slow | Start with reference implementations, use existing examples |
| Performance gains < 10% | Have backup optimization targets (e.g., other fusion pairs) |
| Numerical instability | Implement careful dtype casting, extensive validation tests |
| GPU memory constraints | Profile actively, implement memory-efficient variants |
| Integration complexity | Maintain separate kernel, gradual integration with fallback |

---

## Timeline Summary

```
Week 1-6:   Source code analysis & baseline measurements
Week 7-14:  RMSNorm+Linear fusion implementation
Week 15-18: Performance profiling & optimization
Week 19-22: Documentation & knowledge transfer
Week 23-26: Polish & publication
```

**Checkpoint Reviews** (every 2 weeks):
- Week 4: Architecture analysis complete
- Week 8: Basic kernel working
- Week 12: Fusion fully integrated
- Week 16: Performance targets achieved
- Week 20: Documentation drafted
- Week 24: Final review

---

## Getting Started

1. **Clone repositories**
   ```bash
   cd /Users/ymlin/Downloads/003-Study/138-Projects/nano-vllm
   # nano-vllm and vllm should already be available
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch triton numpy pandas matplotlib
   # nano-vLLM dependencies as per its requirements
   ```

4. **Start with Phase 1**
   - Read nano-vLLM README
   - Create `docs/` directory
   - Begin code analysis

---

## Notes

- **Version Control**: Use git to track progress. Commit at phase boundaries.
- **Documentation Style**: Use clear headings, code blocks, and diagrams where helpful
- **Reproducibility**: All commands should be documented; results should be reproducible on different hardware (with caveats)
- **Learning First**: Understanding the codebase is the foundation—don't rush Phase 1
