# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/0a5b3fd612d24e81a538b34ef561d06c
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: Wave-based fine-grained communication/compute overlap
- Category: Comm/Compute Overlap
- Core Mechanism: Experts are split into waves and compute starts the moment a wave's dispatch lands, so in steady state the current wave's compute, the next wave's transfer and the return of finished results all run concurrently, fused into one pipelined kernel.
- Impact / Measured Results: Fully hides MoE communication whenever per-layer comm time < compute time; derives the hardware balance point C/B <= 2d (about 6144 FLOPs/Byte), so 1 GBps of interconnect covers 6.1 TFLOP/s; up to 1.96x on RL rollout.
- Reference: https://arxiv.org/abs/2606.19348
- Source: DeepSeek V4

## Body
### Papers & slides (PDF)
- DeepSeek-V4 technical report (PDF) — https://arxiv.org/pdf/2606.19348
- Flux: fine-grained comm/compute overlap (PDF) — https://arxiv.org/pdf/2406.06858
### Public resources
- DeepSeek-V4 paper review — https://andlukyane.com/blog/paper-review-deepseekv4 (comm/compute balance point)
- HF technical-report summary — https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/discussions/129
- NVIDIA: Hybrid-EP communication for MoE — https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/ (vendor view of the same problem)
