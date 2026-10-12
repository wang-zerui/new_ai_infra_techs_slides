# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/3e1e63852a934fad8b5d53773913bce5
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: M-axis grouped GEMM (contiguous layout)
- Category: GEMM & MoE Kernels
- Core Mechanism: Unlike CUTLASS generic grouped GEMM, only the M axis is grouped while N/K stay fixed (MoE experts share a shape); per-expert tokens are concatenated into one tensor with each segment aligned to the GEMM M block.
- Impact / Measured Results: Fits training forward and prefill, where per-expert token counts vary, without padding waste.
- Parent Library / Framework: DeepGEMM (https://app.notion.com/p/c0133413563f4ee390f9e4fb55027780)
- Reference: https://github.com/deepseek-ai/DeepGEMM
- Source: DeepGEMM

## Body
### Papers & slides (PDF)
- MegaBlocks: dropless MoE (PDF) — https://arxiv.org/pdf/2211.15841
- DeepSeek-V3 technical report (PDF) — https://arxiv.org/pdf/2412.19437
### Public resources
- DeepGEMM repo — https://github.com/deepseek-ai/DeepGEMM — grouped GEMM APIs
- Exploring DeepGEMM FP8 kernels — https://kingsleykim.dev/blog/deepgemm/ — contiguous grouped layout explained
