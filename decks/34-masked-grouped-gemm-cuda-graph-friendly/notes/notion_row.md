# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/d8af0edc72e64994bbb502dc9a3bfcea
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: Masked grouped GEMM (CUDA Graph friendly)
- Category: GEMM & MoE Kernels
- Core Mechanism: Under decode plus CUDA Graph the CPU cannot know per-expert token counts, so a mask tensor tells the kernel to compute only the valid parts - and it consumes DeepEP low-latency kernel output directly.
- Impact / Measured Results: Removes the per-MoE-layer host sync, a prerequisite for enabling CUDA Graphs in EP inference and rollout.
- Parent Library / Framework: DeepGEMM (https://app.notion.com/p/c0133413563f4ee390f9e4fb55027780)
- Reference: https://github.com/deepseek-ai/DeepGEMM
- Source: DeepGEMM, DeepEP

## Body
### Papers & slides (PDF)
- MegaBlocks: dropless MoE (PDF) — https://arxiv.org/pdf/2211.15841
- FlashInfer (CUDA-graph friendly kernels) (PDF) — https://arxiv.org/pdf/2501.01005
### Public resources
- DeepGEMM repo — https://github.com/deepseek-ai/DeepGEMM — masked grouped GEMM
- LMSYS: large-scale EP with CUDA-graph decode — https://www.lmsys.org/blog/2025-05-05-large-scale-ep — why host syncs must go
