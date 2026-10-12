# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/893ccbb87b354e7e8fedbc0a18ca7117
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: Precision-aware optimizer (BF16/FP8 optimizer states)
- Category: Low-Precision & Quantization
- Core Mechanism: --use-precision-aware-optimizer lets TE optimizers keep first/second moments and master weights in BF16/FP8, mixing precision as needed.
- Impact / Measured Results: Large reduction in optimizer-state memory; standard for trillion-parameter MoEs under limited HBM.
- Reference: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md
- Source: Megatron-LM / MCore, TransformerEngine

## Body
### Papers & slides (PDF)
- ZeRO (PDF) — https://arxiv.org/pdf/1910.02054
- Scalable MoE training with Megatron-Core (PDF) — https://arxiv.org/pdf/2603.07685
### Public resources
- Megatron-Core MoE guide — https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html (precision-aware optimizer flags)
- Transformer Engine FP8 primer — https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html (low-precision state handling)
- Megatron-FSDP docs — https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/features/megatron_fsdp.html (how optimizer state is sharded)
