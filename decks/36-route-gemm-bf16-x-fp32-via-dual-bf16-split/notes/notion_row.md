# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/1ca14bb9a0ce4d439cc9a0686b60a02d
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)
Page last edited: 2026-10-12T19:13:52Z

## Properties
- Technique: Route GEMM (BF16 x FP32 via dual-BF16 split)
- Category: GEMM & MoE Kernels
- Core Mechanism: Splits FP32 weights into a high BF16 term plus a low residual BF16 term (fixed scale 1/256) and fuses two BF16 Tensor-Core GEMMs in one kernel with shared input gathering, register-resident accumulators and a single result write.
- Impact / Measured Results: Up to 3.22x over cuBLAS FP32/TF32 while keeping FP32 accuracy; suits precision-sensitive small matrices such as MoE router GEMMs and the state-compress GEMMs of linear/sparse attention.
- Parent Library / Framework: HPC-Ops (https://app.notion.com/p/85f1abe2180f4b86bf1e9ea6d23705ce)
- Reference: https://github.com/Tencent/hpc-ops
- Source: HPC-Ops

## Body
### Papers & slides (PDF)
- FP8 formats for deep learning (PDF) — https://arxiv.org/pdf/2209.05433
- Scalable MoE training with Megatron-Core (PDF) — https://arxiv.org/pdf/2603.07685
### Public resources
- HPC-Ops repo — https://github.com/Tencent/hpc-ops (Route GEMM implementation)
- LMSYS: HPC-Ops × SGLang — https://www.lmsys.org/blog/2026-08-07-hpc-ops-sglang/ (router GEMM accuracy/speed results)
