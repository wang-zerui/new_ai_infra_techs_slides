# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/3a91e54ff5c7400e9b2e65e0ba6479f5
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)
Page last edited: 2026-09-21T13:07:07Z

## Properties
- Technique: Memory-efficient token permutation
- Category: Memory & Offload Scheduling
- Core Mechanism: Algebraic reordering: moves the router-probability multiply out of unpermute and into the GroupedMLP activation function, avoiding an extra large buffer allocation.
- Impact / Measured Results: Removes one full activation copy - a classic low-cost fix for the MoE memory wall.
- Reference: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md
- Source: Megatron-LM / MCore

## Body
### Papers & slides (PDF)
- MegaBlocks: dropless MoE (PDF) — https://arxiv.org/pdf/2211.15841
- Scalable MoE training with Megatron-Core (PDF) — https://arxiv.org/pdf/2603.07685
### Public resources
- Megatron-Core MoE guide — https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html (permutation and GroupedMLP path)
- Scalable MoE training with Megatron-Core (paper) — https://arxiv.org/html/2603.07685v2 (memory-efficient permutation)
