# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/82d104d9bcc74fab931db687fed76e81
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: Context Parallelism / Dynamic CP
- Category: Parallelism Strategy
- Core Mechanism: Shards inputs and all activations along the sequence dimension (unlike SP, which only splits LayerNorm/Dropout), exchanging KV via ring/p2p; supports MHA/MQA/GQA, MLA and MTP, composes with TP/PP/DP, and now supports dynamic CP.
- Impact / Measured Results: Beats full recomputation beyond 8K sequences by removing most recompute cost while balancing compute and communication.
- Reference: https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/context_parallel.md
- Source: Megatron-LM / MCore, TransformerEngine

## Body
### Papers & slides (PDF)
- Ring Attention with blockwise transformers (PDF) — https://arxiv.org/pdf/2310.01889
- Scalable MoE training with Megatron-Core (PDF) — https://arxiv.org/pdf/2603.07685
### Public resources
- NVIDIA: Speeding up variable-length training with dynamic context parallelism — http://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core (the dynamic CP blog)
- Megatron Bridge parallelisms guide — https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/parallelisms.html (when to use CP)
- Megatron-Core MoE guide — https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html
