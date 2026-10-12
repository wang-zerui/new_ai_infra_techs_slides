# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/4d0436aff16240398bbe1c41ab56d5ad
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: Fused FP8 MoE low-latency pipeline
- Category: GEMM & MoE Kernels
- Core Mechanism: Fuses routing, Gate-Up GEMM, activation quantization, Down GEMM and top-k weighted reduction into a single pipeline; Gate-Up reads original tokens by routing index to drop the gather stage; shared-memory counting relieves global atomics; the low-latency mode trades warp specialization for CTA residency and chains stages with PDL to cut launch bubbles.
- Impact / Measured Results: Up to 1.6x (TP) and 1.5x (EP) over vLLM CUTLASS, Triton and SGLang on DeepSeek-V3 / Hunyuan-V3 / Qwen3-235B.
- Parent Library / Framework: HPC-Ops (https://app.notion.com/p/85f1abe2180f4b86bf1e9ea6d23705ce)
- Reference: https://github.com/Tencent/hpc-ops
- Source: HPC-Ops

## Body
### Papers & slides (PDF)
- MegaBlocks: dropless MoE (PDF) — https://arxiv.org/pdf/2211.15841
- FlashInfer (PDF) — https://arxiv.org/pdf/2501.01005
### Public resources
- HPC-Ops repo — https://github.com/Tencent/hpc-ops (fused MoE kernels)
- LMSYS: HPC-Ops × SGLang — https://www.lmsys.org/blog/2026-08-07-hpc-ops-sglang/ (router + MoE benchmark results)
