# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/2e31e16f18ce45edba1e3d9058756c08
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)
Page last edited: 2026-10-12T19:13:55Z

## Properties
- Technique: Fine-grained selective recomputation
- Category: Memory & Offload Scheduling
- Core Mechanism: --recompute-granularity selective with --recompute-modules selects mla_up_proj / moe / moe_act / layernorm / norm per module, and works under FP8.
- Impact / Measured Results: Cuts activation memory for negligible extra compute, avoiding the 30%+ overhead of full-layer recomputation.
- Reference: https://arxiv.org/abs/2603.07685
- Source: Megatron-LM / MCore

## Body
### Papers & slides (PDF)
- Reducing activation recomputation in large transformers (PDF) — https://arxiv.org/pdf/2205.05198
- Scalable MoE training with Megatron-Core (PDF) — https://arxiv.org/pdf/2603.07685
### Public resources
- NeMo: activation recomputation — https://docs.nvidia.com/nemo-framework/user-guide/24.07/nemotoolkit/features/optimizations/activation_recomputation.html (selective vs full recompute)
- Megatron-Core MoE guide — https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html (module-level recompute flags)
