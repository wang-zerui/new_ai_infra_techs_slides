# Notion row (read-only copy, fetched 2026-10-13)

Page: https://app.notion.com/p/efa722c88bed4a4cb1baa6d8d0a0d6fa
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)
Last edited: 2026-10-12T19:13:40Z

## Properties

| Property | Value |
|---|---|
| Technique | Loss-free balance routing + sigmoid gating |
| Category | MoE & Routing |
| Core Mechanism | A learnable per-expert bias with fixed update rate regulates load instead of an auxiliary loss; scoring uses sigmoid. |
| Impact / Measured Results | Now the default routing setup for frontier open MoEs; MCore ships this router and its fused kernels. |
| Reference | https://arxiv.org/abs/2508.06471 |
| Source | GLM-4.5; DeepSeek V3.2; Megatron-LM / MCore |

## Body

### Papers & slides (PDF)
- Auxiliary-loss-free load balancing (PDF): https://arxiv.org/pdf/2408.15664
- DeepSeek-V3 technical report (PDF): https://arxiv.org/pdf/2412.19437

### Public resources
- HF blog: evolution of MoE load-balancing strategies — https://huggingface.co/blog/NormalUhr/moe-balance (aux-loss vs loss-free)
- Auxiliary-loss-free load balancing (paper) — https://arxiv.org/abs/2408.15664
- Megatron-Core MoE guide — https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html (router implementation)
