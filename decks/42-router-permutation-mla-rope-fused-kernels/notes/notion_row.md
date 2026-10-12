# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/9d82ddbf31244093bc930a84ed679c2d
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: Router / permutation / MLA-RoPE fused kernels
- Category: Kernel Fusion & Runtime
- Core Mechanism: --moe-router-fusion fuses routing score/topk/bias, layered with permutation, MLA RoPE and FP8-cast fusion, plus cuDNN fused attention / FlashAttention integration.
- Impact / Measured Results: Cuts the launch and memory-traffic overhead of many small kernels; pairs with sync-free dropless MoE for higher compute efficiency.
- Reference: https://arxiv.org/abs/2603.07685
- Source: Megatron-LM / MCore, TransformerEngine

## Body
### Papers & slides (PDF)
- Scalable MoE training with Megatron-Core (PDF) — https://arxiv.org/pdf/2603.07685
- MegaBlocks: dropless MoE (PDF) — https://arxiv.org/pdf/2211.15841
### Public resources
- Megatron-Core MoE guide — https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html (router/permutation fusion flags)
- GTC session: Scale MoE training with Megatron Core — https://www.nvidia.com/gtc/session-catalog/sessions/gtc26-cwes81540/ (talk + slides)
- Scalable MoE training paper (MCore) — https://arxiv.org/html/2603.07685v2
