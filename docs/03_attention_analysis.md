# 3. Attention Mechanism Analysis

**Phase**: 1 (Weeks 3-4)  
**Status**: Completed (Phase 1.3)  
**Last Updated**: January 6, 2026

## Overview

Analysis of attention computation in nano-vLLM, identifying RMSNorm+Linear fusion opportunities and current bottlenecks.

## Sections Completed

### 3.1 Attention Implementation
- [x] Attention forward pass
- [x] Query, key, value computations
- [x] Attention scores calculation (FlashAttention)
- [x] Output computation

### 3.2 RMSNorm Analysis
- [x] RMSNorm layer implementations
- [x] Normalization formula
- [x] Data type handling
- [x] Performance characteristics

### 3.3 Linear Layers
- [x] Linear layer implementations
- [x] Weight matrix organization
- [x] Bias handling
- [x] Performance characteristics

### 3.4 Fusion Opportunities
- [x] RMSNorm+Linear pairing analysis
- [x] Frequency of occurrence
- [x] Memory I/O reduction potential
- [x] Fusion feasibility

### 3.5 Current Bottlenecks
- [x] Identified bottlenecks
- [x] Root cause analysis
- [x] Impact quantification
- [x] Optimization targets

## Key Files to Study

```
nano-vllm/
├── nanovllm/layers/attention.py    # Attention ⭐
├── nanovllm/layers/norm.py         # Normalization ⭐
├── nanovllm/layers/linear.py       # Linear layers ⭐
└── nanovllm/models/               # Model definitions ⭐
```

## Findings

### Attention Mechanism Overview
- Attention wrapper lives in [nanovllm/layers/attention.py](nanovllm/layers/attention.py). It relies on two FlashAttention entry points: `flash_attn_varlen_func` for prefill (variable-length, optional block-table for prefix cache) and `flash_attn_with_kvcache` for decode (single token per sequence, KV cache reuse).
- KV writes are done via a Triton kernel `store_kvcache_kernel` that copies contiguous per-token K/V vectors into the preallocated cache using `slot_mapping`; avoids Python loops and keeps strides contiguous. Called only when cache tensors are already bound.
- Prefill path: if prefix cache is present, K/V are swapped to cached tensors (`k_cache`, `v_cache`) and FlashAttention is invoked with `block_table` so attention reads from cached blocks instead of freshly computed tokens. Variable-length metadata comes from `cu_seqlens_q/k` and `max_seqlen_q/k` prepared in the runner.
- Decode path: queries are expanded to shape `[B, 1, H, D]`, then `flash_attn_with_kvcache` consumes `block_table` + `context_lens` to attend over cached KV without rebuilding key/value tensors.
- KV cache tensors are allocated once in [nanovllm/engine/model_runner.py](nanovllm/engine/model_runner.py) and injected into each Attention module during initialization; shapes are `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]` (2 for K/V).

### RMSNorm+Linear Pairing
- RMSNorm implementation in [nanovllm/layers/layernorm.py](nanovllm/layers/layernorm.py) is torch.compile’d; two entry points: `rms_forward` (no residual) and `add_rms_forward` (in-place residual add + norm). Both cast to `float`, normalize by mean square, then cast back to original dtype and scale by learnable weight.
- Pairing in Qwen3 stack ([nanovllm/models/qwen3.py](nanovllm/models/qwen3.py)):
  - `input_layernorm` → `qkv_proj` (ColumnParallel) before attention.
  - `post_attention_layernorm` → `gate_up_proj` (MergedColumnParallel) before MLP.
  - Final `norm` → `lm_head`.
  - Optional `q_norm` / `k_norm` (per-head RMSNorm) executed after QKV split when `qkv_bias=False`; acts on reshaped heads, making fusion with the preceding projection less straightforward.

### Fusion Opportunities
- Most frequent pattern: `RMSNorm` immediately followed by a linear projection (three times per decoder layer, four if counting final norm+head). These are prime candidates for RMSNorm+Linear fusion to eliminate an extra memory read/write of the normalized activation.
- The `add_rms_forward` path already fuses residual add + norm; adding the matmul would further reduce bandwidth (one read of residual+input, one write of matmul output).
- Q/K per-head norms are less amenable: norms happen after reshape to `[B*T, n_heads, head_dim]`; fusing with QKV matmul would require changing projection output layout or introducing head-major GEMM, which is larger scope.
- Tensor Parallelism: Column/Row parallel linears shard weights. Fusion must preserve shard-wise behavior; implementing fused kernels per shard is feasible because inputs are identical to current matmuls.

### Current Bottlenecks
- Bandwidth-bound segments: RMSNorm outputs are immediately re-read by following linears; two extra global memory passes per fused pair.
- Attention path itself is FlashAttention-bound (compute/bandwidth balanced); the non-fused norms/linears likely dominate residual memory traffic in small-batch decode.
- KV write path uses Triton and is efficient; remaining overhead comes from slot_mapping preparation on CPU → pinned → H2D per step.
- With `enforce_eager=False`, decode uses CUDA graphs; fusion will provide benefit both in eager and captured graphs (smaller graph, fewer nodes).

## Key Insights

- 1. Attention uses FlashAttention for both prefill (varlen) and decode (kv-cache), minimizing O(T^2) memory; cache-aware via block tables and `context_lens`.
- 2. KV cache writes are Triton-based and slot-mapped, avoiding scatter inefficiencies and Python overhead.
- 3. RMSNorm is already fused with residual add; main remaining I/O cost is RMSNorm → Linear handoff.
- 4. Three stable RMSNorm+Linear pairs per decoder layer (four including final head); these are the highest-payoff fusion targets.
- 5. Q/K per-head norms are niche and harder to fuse without changing head layout; treat as optional.

## Diagrams

Document attention computation flow:

```
Input (B, T, D)
  ↓
RMSNorm
  ↓
Linear (Project to Q, K, V)
  ↓
Attention computation
  ↓
RMSNorm
  ↓
Linear (Project to output)
  ↓
Output (B, T, D)
```

## Questions for Investigation

- [ ] Where are RMSNorm+Linear pairs located?
- [ ] How many such pairs per layer?
- [ ] What is the computation/memory I/O ratio?
- [ ] Can all pairs be fused?
- [ ] What are the constraints?

## References

### Internal
- [02_memory_management.md](02_memory_management.md)
- [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md)

### External
- RMSNorm paper and implementations
- Attention mechanism references

## Next Steps

1. Examine attention code structure
2. Identify RMSNorm+Linear pairs
3. Analyze fusion potential
4. Document findings
5. Create fusion design (doc 06)

---

**Estimated Duration**: 2 weeks  
**Difficulty**: Medium-High  
**Dependencies**: Phase 1.1, 1.2
