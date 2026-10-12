# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/0e6015aa2a8449ff917574f6139df4c2
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: 1F1B / batch-level EP all-to-all overlap
- Category: Comm/Compute Overlap
- Core Mechanism: --overlap-moe-expert-parallel-comm with --delay-wgrad-compute hides EP all-to-all inside 1F1B pipeline slots and batch-level splits; DP/TP/PP communication overlap is supported as well.
- Impact / Measured Results: Buries EP communication in pipeline bubbles - the default optimization for large-EP training.
- Reference: https://arxiv.org/abs/2603.07685
- Source: Megatron-LM / MCore

## Body
### Papers & slides (PDF)
- Scalable MoE training with Megatron-Core (PDF) — https://arxiv.org/pdf/2603.07685
- Efficient large-scale training on GPU clusters / PTD-P (PDF) — https://arxiv.org/pdf/2104.04473
### Public resources
- Megatron-Core MoE user guide — https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html (overlap flags and recipes)
- GTC session: Scale MoE training with Megatron Core — https://www.nvidia.com/gtc/session-catalog/sessions/gtc26-cwes81540/ (talk + slides)
- Scalable MoE training paper (MCore) — https://arxiv.org/html/2603.07685v2
