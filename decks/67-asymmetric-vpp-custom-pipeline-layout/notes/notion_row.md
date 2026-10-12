# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/0b667c3fda7240be99b87c3b35fdcfcd
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: Asymmetric VPP / custom pipeline layout
- Category: Parallelism Strategy
- Core Mechanism: --pipeline-model-parallel-layout allows non-uniform assignment of layers, embeddings and MTP heads across PP stages (flexible asymmetric virtual pipelining).
- Impact / Measured Results: Fixes the PP load imbalance caused by MTP/embedding layers and shrinks pipeline bubbles.
- Reference: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md
- Source: Megatron-LM / MCore

## Body
### Papers & slides (PDF)
- Efficient large-scale training on GPU clusters / PTD-P (PDF) — https://arxiv.org/pdf/2104.04473
- Megatron-LM (PDF) — https://arxiv.org/pdf/1909.08053
- GTC slides: Megatron-LM model parallelism (PDF) — https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21496-megatron-lm-training-multi-billion-parameter-language-models-using-model-parallelism.pdf
### Public resources
- Megatron-Core MoE guide — https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html (pipeline layout flags)
- GTC session: Scale MoE training with Megatron Core — https://www.nvidia.com/gtc/session-catalog/sessions/gtc26-cwes81540/ (talk + slides)
- GTC slides: Megatron-LM model parallelism (PDF) — (same as above) pipeline-schedule background
