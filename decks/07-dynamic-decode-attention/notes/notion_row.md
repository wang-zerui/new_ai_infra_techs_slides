# Notion row (fetched 2026-10-12T19:13:52Z, read-only)

URL: https://app.notion.com/p/395fb76ae58944d0ae141c2a047478b6
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties

- **Technique**: Dynamic Decode Attention (dynamic task scheduling)
- **Category**: Attention Kernels
- **Core Mechanism**: Splits all requests into uniform KV tiles, builds a task graph before each decode step and
  greedily bin-packs it across CTAs; the attention kernel consumes the task map directly, with a combine kernel
  merging split-k results.
- **Impact / Measured Results**: Up to 2.88x over static split-k scheduling (long context and mixed-length
  batches), sharply cutting CTA tail latency.
- **Parent Library / Framework**: HPC-Ops (https://app.notion.com/p/85f1abe2180f4b86bf1e9ea6d23705ce)
- **Reference**: https://github.com/Tencent/hpc-ops
- **Source**: HPC-Ops

## Body

### Papers & slides (PDF)
- FlashAttention-3 (PDF) — https://arxiv.org/pdf/2407.08608
- PagedAttention / vLLM (PDF) — https://arxiv.org/pdf/2309.06180

### Public resources
- HPC-Ops repo — https://github.com/Tencent/hpc-ops — kernel source and benchmarks
- LMSYS: HPC-Ops × SGLang — https://www.lmsys.org/blog/2026-08-07-hpc-ops-sglang/ — attention scheduling results
