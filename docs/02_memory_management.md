# 2. Memory Management in nano-vLLM

## Overview

This document analyzes memory management strategies in nano-vLLM, focusing on KV cache handling, memory allocation patterns, and optimization opportunities compared to vLLM.

## Sections to Complete

### 2.1 KV Cache Management
- [x] KV cache storage structure — **DONE** (block-based tables)
- [x] Cache allocation and initialization — **DONE** (BlockManager)
- [x] Cache update patterns during inference — **DONE** (prefill/decode paths)
- [ ] Memory layout and access patterns — **TODO** (need profiling)

### 2.2 Memory Allocation Strategies
- [x] Dynamic vs static allocation — **DONE** (static pool, dynamic assignment)
- [x] Paging mechanisms — **DONE** (fixed-size blocks, hash reuse)
- [ ] Memory pooling and reuse — **TODO** (measure reuse rate)
- [ ] Fragmentation analysis — **TODO** (analyze partial blocks)

### 2.3 Comparison with vLLM
- [ ] PagedAttention memory model
- [ ] Block-based allocation vs contiguous
- [ ] Memory efficiency trade-offs
- [ ] Scalability characteristics

### 2.4 Optimization Opportunities
- [x] Identified bottlenecks — **DONE** (hash lookup + partial-block handling)
- [x] Potential improvements — **DONE** (prefetch, better hashing reuse)
- [ ] Impact analysis — **TODO** (quantify cache hit rate)
- [ ] Trade-offs — **TODO** (memory vs compute)

## Key Files to Study

```
nano-vllm/
├── nanovllm/engine/        # Execution engine ⭐
├── nanovllm/layers/        # Layer implementations ⭐
└── nanovllm/utils/         # Utilities ⭐

vllm/
├── vllm/engine/            # Engine comparisons
├── vllm/attention/         # Attention memory model
└── vllm/core_exec.py       # Memory management
```

## Findings

### KV Cache Organization
- **Block size**: 256 tokens (`Sequence.block_size`)
- **Blocks**: Fixed pool created at `BlockManager(num_blocks, block_size)` from config
- **Block table**: Each sequence holds a list of block IDs (`seq.block_table`)
- **Hashing for reuse**: xxhash64 of token_ids (optionally chained with previous block hash) used to deduplicate full blocks
- **Prefix reuse**: If hash matches and token_ids match, block is reused and `ref_count` increments; otherwise allocate new block
- **Partial blocks**: Last block is unhashed until full; hashed and inserted into `hash_to_block_id` only when full

### Memory Allocation Patterns
- **Static pool, dynamic assignment**: Blocks are preallocated IDs; allocation only updates tables, not tensors (tensors live elsewhere)
- **Free/used tracking**: `free_block_ids` deque and `used_block_ids` set manage availability
- **Prefill phase**: `allocate(seq)` reserves blocks for entire prompt; cache hits add `num_cached_tokens`
- **Decode phase**: `may_append(seq)` handles single-token appends; allocates a new block when a new block starts, hashes and finalizes when block fills
- **Deallocation**: On sequence finish or preempt, `deallocate(seq)` decrements `ref_count`; frees when zero
- **Cache miss path**: Falls back to next free block; updates hash table only when block is full

### Performance Implications
- **Reuse benefits**: Prefix reuse can skip reading/writing KV for repeated prefixes; tracked via `num_cached_tokens`
- **Hash lookup cost**: xxhash64 per block; should be negligible vs compute but may matter with many short blocks
- **Fragmentation risk**: Partial blocks cannot be reused until full; prompts that end mid-block reduce reuse efficiency
- **Capacity constraint**: Scheduling is gated by `can_allocate` / `can_append`; OOM avoided by pre-checks, but head-of-line blocking can occur
- **Cache locality**: Block IDs do not imply physical proximity; actual KV tensors likely contiguous by block ID (needs confirmation in model runner)

## Key Insights

1. **Block-based KV cache** with hash-chained prefixes enables prefix reuse without copying data.
2. **Static pool + dynamic mapping**: Allocation is just table updates; no per-request tensor allocation → low overhead.
3. **Partial-block inefficiency**: Reuse only triggers on full blocks; short prompts reduce hit rate — opportunity to pad or group.
4. **Scheduler is cache-aware**: Refuses to schedule if blocks unavailable; can preempt to free blocks during decode.
5. **Fusion impact**: Reducing per-token latency helps decode, but cache hit rate also critical—should measure `num_cached_tokens` vs total.

## Code Examples

Document important memory management patterns:

```python
# Example: KV cache allocation
# cache = torch.zeros((batch_size, seq_len, num_heads, head_dim))
```

## Questions for Investigation

- [x] How is memory pre-allocated? → **Static block pool at init**
- [x] What is the maximum batch size? → **Scheduler limited by `max_num_seqs` and `max_num_batched_tokens` plus block availability**
- [x] How does memory scale with sequence length? → **Linear in blocks: ceil(seq_len / 256) blocks**
- [x] How is memory reclaimed between batches? → **On finish or preempt: `deallocate` decrements `ref_count` and frees**
- [ ] What are memory fragmentation issues? → **Partial blocks reduce reuse; need measurement**
- [ ] What is cache hit rate in practice? → **Requires runtime metrics on real prompts**
- [ ] How are KV tensors laid out physically? → **Need to inspect model_runner implementation**

## References

### Internal
- [01_nano_vllm_architecture.md](01_nano_vllm_architecture.md)
- [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md)

### External
- vLLM PagedAttention implementation
- CUDA memory management best practices

## Next Steps

1. Examine KV cache implementation
2. Trace memory allocation calls
3. Compare with vLLM approach
4. Identify optimization opportunities
5. Document findings

---

**Estimated Duration**: 2 weeks  
**Difficulty**: Medium  
**Dependencies**: Phase 1.1 (Architecture understanding)
