# 1. nano-vLLM Architecture Analysis

**Phase**: 1 (Weeks 1-2)  
**Status**: 🔧 In Progress  
**Last Updated**: January 6, 2026

## Overview

This document provides a comprehensive analysis of the nano-vLLM lightweight LLM inference framework architecture. We examine the project structure, design philosophy, core components, and execution flow to establish a foundation for understanding the system.

**Key Finding**: nano-vLLM is a **minimal reimplementation of vLLM** (~1,200 lines of Python) with comparable performance but much simpler codebase.

## Sections to Complete

### 1.1 Project Design Philosophy
- [x] Lightweight vs full-featured frameworks - **DONE**
- [x] Design trade-offs - **DONE**
- [x] Target use cases - **DONE**
- [x] Key design decisions - **DONE**

### 1.2 Directory Structure Analysis
- [x] `/nanovllm/` core module organization - **DONE**
- [x] `/engine/` - execution engine - **DONE**
- [x] `/layers/` - neural network layers - **IN PROGRESS**
- [ ] `/models/` - model implementations
- [ ] `/utils/` - utilities
- [ ] `/benchmarks/` - performance evaluation

### 1.3 Core Components
- [x] `llm.py` - main inference interface - **DONE**
- [x] `config.py` - configuration management - **IN PROGRESS**
- [x] `sampling_params.py` - sampling parameters - **DONE**
- [ ] Integration between components

### 1.4 Execution Flow
- [ ] Model loading pipeline
- [ ] Token generation process
- [ ] Batch processing
- [ ] KV cache management
- [ ] Attention computation

### 1.5 Comparison with vLLM
- [ ] Architectural differences
- [ ] Feature parity
- [ ] Performance characteristics
- [ ] Size and complexity metrics

## Key Files to Study

```
nano-vllm/
├── nanovllm/
│   ├── __init__.py              # Package initialization (exports LLM, SamplingParams)
│   ├── llm.py                   # Main inference class ⭐ Entry point
│   ├── config.py                # Configuration management ⭐
│   ├── sampling_params.py        # Sampling parameters
│   ├── engine/                   # Core inference engine ⭐
│   │   ├── llm_engine.py        # Main engine (request management, scheduling)
│   │   ├── model_runner.py      # Model execution (forward pass)
│   │   ├── scheduler.py         # Request scheduling & batching
│   │   ├── sequence.py          # Token sequence management
│   │   └── block_manager.py     # KV cache block allocation
│   ├── layers/                   # Layer implementations ⭐
│   │   ├── attention.py         # Attention with KV cache management
│   │   ├── layernorm.py         # RMSNorm implementation
│   │   ├── linear.py            # Linear layers (Q, K, V, output projections)
│   │   ├── rotary_embedding.py  # RoPE positional encoding
│   │   ├── activation.py        # Activation functions (SwiGLU)
│   │   ├── embed_head.py        # Embedding and output head
│   │   └── sampler.py           # Token sampling
│   ├── models/                   # Model definitions (Qwen, LLaMA, etc.)
│   └── utils/                    # Utilities
├── example.py                   # Usage example ⭐
├── bench.py                     # Benchmarking script
└── README.md                    # Project overview
```

**High Priority** (⭐): Start with these files for core understanding

## Findings

### Architecture Overview

**nano-vLLM is a reimplementation-from-scratch approach** to vLLM, not a fork. Key characteristics:

1. **Scale**: ~1,200 lines of Python code (vs vLLM's ~100K+ lines)
2. **Philosophy**: Readable, educational codebase with comparable performance
3. **Performance**: Achieves 1434.13 tokens/s vs vLLM's 1361.84 tokens/s on Qwen3-0.6B
4. **Model Support**: Originally designed for smaller models (Qwen3-0.6B tested)

**Package Structure**:
```
nanovllm/
├── llm.py ..................... Main user-facing class (inherits from LLMEngine)
├── config.py .................. Configuration management
├── sampling_params.py ......... Sampling configuration
├── engine/ .................... Core inference engine
│   ├── llm_engine.py ......... Main engine implementation
│   ├── model_runner.py ....... Model execution (forward pass)
│   ├── scheduler.py .......... Request scheduling & batching
│   ├── sequence.py ........... Token sequence management
│   └── block_manager.py ...... KV cache block allocation
├── layers/ .................... Neural network layers (optimized kernels)
├── models/ .................... Model-specific implementations
└── utils/ ..................... Utility functions
```

### Design Principles

**1. Minimalism**: Only essential components for inference
- No training support
- No distributed training features
- No advanced distributed serving features (yet)

**2. Readability**: Educational codebase for learning
- Clear function names and structure
- Minimal abstraction layers
- Direct implementation of core concepts

**3. Performance**: Without sacrificing simplicity
- Uses CUDA kernels where critical
- Memory-efficient KV cache management
- Efficient scheduling and batching

**4. Compatibility**: vLLM-like API
- Similar `LLM` class interface
- Similar `SamplingParams`
- Drop-in replacement for small models

### Performance Characteristics

**Tested Configuration**:
- Hardware: RTX 4070 Laptop (8GB VRAM)
- Model: Qwen3-0.6B (very small model)
- Test: 256 sequences, 100-1024 token input/output

**Results**:
| Engine | Output Tokens | Time | Throughput |
|--------|--------------|------|-----------|
| vLLM | 133,966 | 98.37s | 1361.84 tok/s |
| nano-vLLM | 133,966 | 93.41s | 1434.13 tok/s |

**Why faster**: Simplified implementation overhead, less memory management complexity

### Limitations and Opportunities

**Current Limitations**:
1. Tested only on small models (0.6B parameters)
2. No async/concurrent request handling (likely)
3. Simpler scheduler than vLLM
4. No tensor parallelism (in current version)
5. Fewer optimization options

**Optimization Opportunities**:
1. **RMSNorm+Linear Fusion**: Paired operations in attention/MLP layers (our target!)
2. **Prefix Caching**: Reuse computed KV cache for repeated prefixes
3. **Attention Optimization**: More efficient attention implementations
4. **Memory Layout**: Better KV cache organization
5. **Quantization Support**: Reduce model size

## Key Insights

*Key learnings from initial analysis*

1. **Simplicity is Powerful**: nano-vLLM achieves better throughput than vLLM with 1/100th the code complexity
2. **Clean Architecture**: Separation of concerns (engine, scheduler, model_runner, block_manager) makes code understandable
3. **KV Cache is Critical**: Block manager and sequence management are central to performance
4. **Small Models Ready**: Designed and tested for small models (0.6B-13B range)
5. **Fusion Opportunity Clear**: Many RMSNorm+Linear pairs likely exist in attention/MLP blocks - ideal for our optimization 

## Code Annotations

### Execution Flow Diagram

```
User Call: llm.generate(prompts, sampling_params)
    ↓
LLMEngine.__init__
    ├─ Load tokenizer
    ├─ Create ModelRunner (main GPU process + parallel processes for tensor parallelism)
    └─ Initialize Scheduler (manages request queue and batching)
    ↓
LLMEngine.add_request (for each prompt)
    ├─ Tokenize prompt → token_ids
    └─ Create Sequence object → Add to scheduler
    ↓
LLMEngine.step (repeatedly until all requests done)
    ├─ Scheduler.schedule()
    │   └─ Return: (seqs, is_prefill flag)
    │       - is_prefill: True for first step (prefill phase)
    │       - is_prefill: False for subsequent steps (decode phase)
    │
    ├─ ModelRunner.run(seqs, is_prefill)
    │   ├─ For each sequence:
    │   │   ├─ Embed tokens (embed_head.py)
    │   │   └─ For each layer in model:
    │   │       ├─ RMSNorm (layernorm.py) ← target for fusion!
    │   │       ├─ Attention (attention.py)
    │   │       │   ├─ Project to Q, K, V (linear.py) ← target for fusion!
    │   │       │   ├─ Update KV cache (block_manager.py)
    │   │       │   └─ FlashAttention (from flash-attn)
    │   │       ├─ RMSNorm (residual) ← target for fusion!
    │   │       └─ MLP (attention.py)
    │   │           ├─ RMSNorm (layernorm.py) ← target for fusion!
    │   │           ├─ Linear gate projection (linear.py) ← target for fusion!
    │   │           ├─ Activation (SwiGLU)
    │   │           └─ Linear output projection (linear.py)
    │   │
    │   ├─ Output embedding (embed_head.py)
    │   └─ Sample next token (sampler.py)
    │
    └─ Update sequences with new tokens
    ↓
LLMEngine.generate (yields results as sequences complete)
```

### Key Layer Components

**RMSNorm Layer** (`layernorm.py`):
```python
@torch.compile
def rms_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Compute: x / sqrt(mean(x^2) + eps) * weight
    
    This is RMS Layer Normalization.
    Often followed immediately by a Linear layer.
    """
    orig_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))
    x = x.to(orig_dtype).mul_(self.weight)
    return x
```

**Linear Layer** (`linear.py`):
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Compute: x @ weight.T + bias
    
    Used for:
    - Q, K, V projections in attention
    - MLP hidden→output
    """
```

### Model Architecture Pattern

All models follow this pattern (from models/ directory):

```
Model:
    for each token position:
        Embed(token)
        for each TransformerBlock:
            ┌─ Layer Norm
            ├─ Attention
            └─ (residual connection)
            ┌─ Layer Norm
            ├─ MLP
            │   ├─ Linear (expand to 4x)
            │   ├─ Activation (SwiGLU)
            │   └─ Linear (project back)
            └─ (residual connection)
        Output projection
        Sample next token
```

**Fusion Opportunities**:
- Layer Norm → Attention Q/K/V projections: **2-3 pairs per layer**
- Layer Norm → MLP gate projection: **1 pair per layer**
- For 32-layer model: **96-128 RMSNorm+Linear pairs total**

## Questions for Investigation

- [x] How is the KV cache managed across requests? → **Block manager handles it**
- [x] What scheduling algorithm is used? → **Custom scheduler (need to examine scheduler.py)**
- [x] How does memory allocation work? → **Block allocation for KV cache (block_manager.py)**
- [x] What are the bottlenecks in current implementation? → **RMSNorm+Linear pairs likely candidates**
- [x] How are attention operations implemented? → **Uses FlashAttention library with custom KV cache storage**
- [x] What parallelization strategies are used? → **Tensor parallelism via multiprocessing, no data parallelism visible yet**

**To Investigate Further**:
- [ ] Exact scheduler implementation and batching strategy
- [ ] KV cache memory overhead
- [ ] Impact of is_prefill flag on computation
- [ ] Exact transformer block implementation in models/

## References

### Internal
- [README.md](../README.md) - Project overview
- [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) - Full project plan

### External
- nano-vLLM repository: [link]
- Related papers: [links]

## Next Steps

1. Read nano-vLLM README and documentation
2. Examine file structure and import hierarchy
3. Trace execution flow from example.py
4. Compare with vLLM equivalent components
5. Document findings in this file

---

**Estimated Duration**: 2 weeks  
**Difficulty**: Medium  
**Dependencies**: None (foundation work)
