# Notion row (read-only copy, fetched 2026-10-13)

Page: https://app.notion.com/p/354b71e013554721a039936a8fff8b2a
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)
Last edited: 2026-10-12T19:13:38Z

## Properties

| Property | Value |
|---|---|
| Technique | MTP (multi-token prediction) layer |
| Category | Model Shape & Depth Scaling |
| Core Mechanism | One (MoE) MTP layer on top of the backbone provides multi-token supervision during training and serves as the speculative-decoding drafter at inference; MCore supports MTP + MLA + CP together. |
| Impact / Measured Results | Wins on both training efficiency and decode throughput; GLM-5.2's improved MTP raises accepted length by 20%; also raises per-step compute density in Ascend deployments. |
| Reference | https://arxiv.org/abs/2508.06471 |
| Source | DeepSeek V4; GLM-4.5; GLM-5 series; Megatron-LM / MCore |

## Body

### Papers & slides (PDF)
- DeepSeek-V4 technical report (PDF): https://arxiv.org/pdf/2606.19348
- Fast inference via speculative decoding (PDF): https://arxiv.org/pdf/2211.17192

### Public resources
- NVIDIA: introduction to speculative decoding — https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/ (MTP as a drafter)
- Sebastian Raschka: multi-token prediction — https://sebastianraschka.com/llm-architecture-gallery/mtp/ (illustrated explainer)
- vLLM MTP docs — https://docs.vllm.ai/en/stable/features/speculative_decoding/mtp/ (serving-side usage)
