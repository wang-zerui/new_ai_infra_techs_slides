# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/56c25e6b3cbe4425b251e2b59530b802
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: K-axis grouped GEMM (MoE weight backward)
- Category: GEMM & MoE Kernels
- Core Mechanism: k_grouped_fp8_gemm_tn_contiguous keeps M/N fixed and groups along K, purpose-built for MoE weight gradients (wgrad), with a matching K-grouped UE8M0 packing kernel.
- Impact / Measured Results: Puts the MoE backward pass on the FP8 fine-grained-scaling path as well - critical for training, unlike inference-only operator libraries.
- Parent Library / Framework: DeepGEMM (https://app.notion.com/p/c0133413563f4ee390f9e4fb55027780)
- Reference: https://github.com/deepseek-ai/DeepGEMM
- Source: DeepGEMM

## Body
### Papers & slides (PDF)
- DeepSeek-V3 technical report (PDF) — https://arxiv.org/pdf/2412.19437
- Scalable MoE training with Megatron-Core (PDF) — https://arxiv.org/pdf/2603.07685
### Public resources
- DeepGEMM repo — https://github.com/deepseek-ai/DeepGEMM — k-grouped wgrad kernels
- Megatron-Core MoE guide — https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html — where MoE backward sits in training
