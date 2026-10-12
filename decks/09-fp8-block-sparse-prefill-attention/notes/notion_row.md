# Notion row (fetched 2026-10-12T19:13:52Z, read-only)

URL: https://app.notion.com/p/6ee53460854a4c988e083e2ba08d1f88
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties

- **Technique**: FP8 block-sparse prefill attention
- **Category**: Attention Kernels
- **Core Mechanism**: Takes a precomputed block mask, skips masked KV tiles wholesale, and applies per-tile FP8
  scaling to preserve numerical quality under sparsity (paged KV supported).
- **Impact / Measured Results**: Up to 3.16x over MIT-BSA / FlashPrefill-BSA (BF16) and a dense FP8 baseline; the
  companion kernel that makes DSA-style sparsity deployable.
- **Parent Library / Framework**: HPC-Ops (https://app.notion.com/p/85f1abe2180f4b86bf1e9ea6d23705ce)
- **Reference**: https://github.com/Tencent/hpc-ops
- **Source**: HPC-Ops

## Body

### Papers & slides (PDF)
- Native Sparse Attention (PDF) — https://arxiv.org/pdf/2502.11089
- FlashAttention-3 (PDF) — https://arxiv.org/pdf/2407.08608

### Public resources
- HPC-Ops repo — https://github.com/Tencent/hpc-ops — block-sparse prefill kernel
- LMSYS: HPC-Ops × SGLang — https://www.lmsys.org/blog/2026-08-07-hpc-ops-sglang/ — measured speedups
- NVIDIA cuDNN sparse attention ops — https://docs.nvidia.com/deeplearning/cudnn/latest/fe-oss-apis/dsa.html — vendor counterpart
