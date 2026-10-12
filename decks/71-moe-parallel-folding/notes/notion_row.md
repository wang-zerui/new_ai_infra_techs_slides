# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/ef4d75b8eee841a79ada63b79bff78d8
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: MoE Parallel Folding
- Category: Parallelism Strategy
- Core Mechanism: Decouples the parallel mappings of attention and MoE layers so EP can be "folded" inside TP x CP groups, resolving the dense-sparse parallelism mismatch.
- Impact / Measured Results: Breaks the EP <= DP limit: in a 256-GPU example EP rises from at most 8 to 64 (ETP=1, EDP=1).
- Reference: https://arxiv.org/abs/2603.07685
- Source: Megatron-LM / MCore

## Body
### Papers & slides (PDF)
- Scalable MoE training with Megatron-Core (PDF) — https://arxiv.org/pdf/2603.07685
- Efficient large-scale training on GPU clusters / PTD-P (PDF) — https://arxiv.org/pdf/2104.04473
### Public resources
- Megatron-Core MoE user guide — https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html (parallel folding configuration)
- MoE README (parallel folding advantages) — https://github.com/LAION-AI/Megatron-LM-Open-Sci/blob/main/megatron/core/transformer/moe/README.md
- Scalable MoE training with Megatron-Core (paper) — https://arxiv.org/html/2603.07685v2
