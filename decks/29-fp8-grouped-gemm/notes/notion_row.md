# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/4b75d7ab1bfc40239d6b9566c5d6fc14
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: FP8 grouped GEMM (per-tensor / block-wise scaling)
- Category: GEMM & MoE Kernels
- Core Mechanism: FP8 grouped GEMM for EP and grouped-expert matmuls, accepting cu_num_tokens input with unequal per-group token counts and both quantization granularities.
- Impact / Measured Results: vs DeepGEMM: 1.1x on prefill and 1.88x on decode (H20).
- Parent Library / Framework: HPC-Ops (https://app.notion.com/p/85f1abe2180f4b86bf1e9ea6d23705ce)
- Reference: https://github.com/Tencent/hpc-ops
- Source: HPC-Ops

## Body
### Papers & slides (PDF)
- MegaBlocks: dropless MoE (PDF) — https://arxiv.org/pdf/2211.15841
- FP8 formats for deep learning (PDF) — https://arxiv.org/pdf/2209.05433
### Public resources
- HPC-Ops repo — https://github.com/Tencent/hpc-ops (grouped GEMM source)
- LMSYS: HPC-Ops × SGLang — https://www.lmsys.org/blog/2026-08-07-hpc-ops-sglang/ (prefill/decode comparison vs DeepGEMM)
- DeepGEMM repo — https://github.com/deepseek-ai/DeepGEMM (the baseline being compared)
