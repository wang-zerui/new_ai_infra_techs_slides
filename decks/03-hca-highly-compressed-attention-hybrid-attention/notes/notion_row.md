# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/ceae2901ccd44561a14abacc7f6574bf
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties

| Property | Value |
|---|---|
| Technique | HCA highly-compressed attention + hybrid attention stacking |
| Category | Attention & Long Context |
| Core Mechanism | High-compression attention layers with m'=128 interleaved with CSA; every layer remains attention - no SSM or Gated-DeltaNet-style linear attention is introduced. |
| Impact / Measured Results | KV cache falls to 10% (Pro) / 7% (Flash) of V3.2, making 1M context routine. |
| Reference | https://arxiv.org/abs/2606.19348 |
| Source | DeepSeek V4 |

## Page body

### Papers & slides (PDF)
- DeepSeek-V4 technical report (PDF): https://arxiv.org/pdf/2606.19348
- DeepSeek-V2 / MLA (PDF): https://arxiv.org/pdf/2405.04434

### Public resources
- DeepSeek-V4 paper review: https://andlukyane.com/blog/paper-review-deepseekv4 — hybrid attention stack
- DeepSeek V4 compressed attention explainer: https://deepseek.ai/blog/deepseek-v4-compressed-attention — KV-cache math
- HF technical-report summary thread: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/discussions/129
