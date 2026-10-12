# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/bbdbb22301864a05bcbc8e4097a6b284
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: Megatron-FSDP / HSDP (with EP support)
- Category: Parallelism Strategy
- Core Mechanism: Fully-sharded data parallel and hybrid sharding (--num-distributed-optimizer-instances), composable with EP since v0.15; complements the distributed optimizer (ZeRO-1 equivalent).
- Impact / Measured Results: Model-state sharding and DP become interchangeable, simplifying memory planning for very large MoEs.
- Reference: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md
- Source: Megatron-LM / MCore

## Body
### Papers & slides (PDF)
- ZeRO (PDF) — https://arxiv.org/pdf/1910.02054
- PyTorch FSDP (PDF) — https://arxiv.org/pdf/2304.11277
### Public resources
- Megatron-FSDP docs — https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/features/megatron_fsdp.html (sharding options)
- MCore custom FSDP API guide — https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/custom_fsdp.html
- PyTorch FSDP introduction — https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/ (background)
