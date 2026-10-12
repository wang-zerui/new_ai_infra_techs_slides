# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/2a55d26c7c7a4f858f5816d7ca24b473
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: Fused AllReduce + residual + RMSNorm
- Category: Comm/Compute Overlap
- Core Mechanism: Fuses the post-TP AllReduce, residual add and RMSNorm into an NVLink-native kernel: a CUDA multicast high-throughput mode for many tokens and a Lamport P2P dual-kernel mode with PDL overlap for few tokens, both on two-shot communication schedules.
- Impact / Measured Results: Up to 1.76x over NCCL / FlashInfer for single-node BF16 TP shapes at hidden 4096/5120/7168.
- Parent Library / Framework: HPC-Ops (https://app.notion.com/p/85f1abe2180f4b86bf1e9ea6d23705ce)
- Reference: https://github.com/Tencent/hpc-ops
- Source: HPC-Ops

## Body
### Papers & slides (PDF)
- Flux: fine-grained comm/compute overlap (PDF) — https://arxiv.org/pdf/2406.06858
- FlashInfer (PDF) — https://arxiv.org/pdf/2501.01005
### Public resources
- HPC-Ops repo — https://github.com/Tencent/hpc-ops (fused collective kernels)
- LMSYS: HPC-Ops × SGLang — https://www.lmsys.org/blog/2026-08-07-hpc-ops-sglang/ (vs NCCL / FlashInfer)
- vLLM fusion passes — https://docs.vllm.ai/en/v0.19.1/design/fusions/ (allreduce+residual+RMSNorm fusion in vLLM)
