# Phase 1.4: Baseline Performance Metrics

## Objective
Establish baseline performance metrics using vLLM on Qwen2-1.5B-Instruct for comparison with future nano-vLLM optimizations.

## Test Configuration

### Hardware
- **GPU:** NVIDIA RTX 3090 (24GB VRAM)
- **CPU:** 2xSandy Bridge-EP @ 2.60 GHz
- **GPU Compute Capability:** SM86 (Ada)
- **CUDA:** 12.1.105
- **Driver:** NVIDIA Driver Version 550

### Software Stack
- **PyTorch:** 2.4.0+cu121
- **vLLM:** 0.6.1.post2
- **Model:** Qwen/Qwen2-1.5B-Instruct
- **Model Size:** ~1.5B parameters
- **Precision:** bfloat16

### Benchmark Parameters
- **Number of Sequences:** 256
- **Max Input Length:** 1024 tokens
- **Max Output Length:** 100-1024 tokens per sequence (randomly distributed)
- **Temperature:** 0.6
- **Ignore EOS:** True
- **GPU Memory Utilization:** 0.9

## Results

### vLLM Baseline (Qwen2-1.5B)

| Metric | Value |
|--------|-------|
| **Total Tokens Generated** | 144,831 |
| **Execution Time** | 25.43s |
| **Throughput (tok/s)** | **5,695.29** |
| **Avg Tokens/Sequence** | 565.7 |
| **Model Load Time** | ~13s (CUDA graph capture) |
| **Warmup Time** | ~1s |

### Performance Breakdown

**Model Initialization:**
- Engine init: ~5s
- Weight loading: 2.89 GB in ~2s
- GPU block allocation: 37,416 blocks
- CUDA graph capture: 20s (includes warmup optimization)

**Inference Performance:**
- Input speed: 80.65 toks/s (prefill)
- Output speed: 5,703.10 toks/s (decoding phase)
- Throughput is decoding-bound (expected for short sequences)

## System State During Benchmark

### GPU Utilization
```
GPU Blocks: 37,416
CPU Blocks: 9,362
Total KV Cache Size: ~46GB equivalent block slots
```

### Configuration Details
- **Tensor Parallel Size:** 1 (single GPU)
- **Pipeline Parallel Size:** 1
- **Block Manager Version:** V1 (legacy)
- **Prefix Caching:** Disabled
- **Async Output Processing:** Enabled

## Observations & Insights

### 1. Throughput Characteristics
- **Decoding-Dominant Workload:** 5,703 tok/s output vs 80.65 tok/s input indicates the benchmark is primarily measuring decoding speed
- **256 Parallel Sequences:** Excellent GPU utilization with high batch size
- **Random Token Lengths:** Realistic distribution mimics real-world serving scenarios

### 2. Memory Efficiency
- **Loaded Model Size:** 2.89 GB (weights only)
- **GPU Block Allocation:** 37K blocks suggest ~11GB total KV cache space allocated
- **GPU Memory Utilization:** 0.9 (90%) = reasonable safety margin

### 3. Latency Components
- **Model Load:** ~13s CUDA graph capture overhead (one-time, amortized)
- **Batch Processing:** ~25s for 256 sequences
- **Per-Sequence Latency:** ~99ms average (25.43s / 256)

### 4. Comparison Points for Optimization
- **Baseline Decoding Speed:** 5,695 tok/s
- **Target for Phase 2 (RMSNorm+Linear Fusion):** 10-20% improvement → 6,800+ tok/s
- **Potential via nano-vLLM optimizations:** Kernel fusion, better memory layout

## Network Acceleration Note
This benchmark was executed on AutoDL GPU server with **network turbo enabled** (`source /etc/network_turbo`), which provides:
- HuggingFace mirror access (hf-mirror.com)
- GitHub acceleration via proxy
- Critical for accessing models in mainland China

## Test Date & Environment
- **Date:** 2026-01-06
- **Server:** AutoDL GPU Instance (news-server)
- **Region:** Mainland China (GFW considerations)
- **Network:** Academic acceleration enabled

## Next Steps (Phase 1.5 & Beyond)

### Phase 1.5: Baseline Comparison
- [ ] Install nano-vllm with working flash-attn
- [ ] Run same benchmark with nano-vllm
- [ ] Compare throughput, latency, memory

### Phase 2: Optimization Implementation
- [ ] RMSNorm+Linear kernel fusion
- [ ] Memory layout optimization
- [ ] KV cache access pattern improvement
- [ ] Re-benchmark to measure improvements

### Phase 3: Detailed Analysis
- [ ] Profile with nsys/nsight
- [ ] Identify remaining bottlenecks
- [ ] Measure kernel-level improvements

## References
- vLLM: https://github.com/vllm-project/vllm
- Qwen2 Model: https://huggingface.co/Qwen/Qwen2-1.5B-Instruct
- AutoDL Network Turbo: https://www.autodl.com/docs/network_turbo/
