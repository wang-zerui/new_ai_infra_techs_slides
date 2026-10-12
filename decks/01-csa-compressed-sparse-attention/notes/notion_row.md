# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/e36ba1f7295647b897468dc2b4d30590
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties

| Property | Value |
|---|---|
| Technique | CSA compressed sparse attention |
| Category | Attention & Long Context |
| Core Mechanism | Every m=4 consecutive tokens are compressed into one KV entry by a learned overlapping-window weighted average (1M -> 250K), followed by top-k (512/1024) selection plus a 128-token sliding window for local detail; MQA at the core. |
| Impact / Measured Results | Per-token FLOPs at 1M context are only 27% (Pro) / 10% (Flash) of V3.2. |
| Reference | https://arxiv.org/abs/2606.19348 |
| Source | DeepSeek V4 |

## Page body

### Papers & slides (PDF)
- DeepSeek-V4 technical report (PDF): https://arxiv.org/pdf/2606.19348
- DeepSeek-V2 / MLA (PDF): https://arxiv.org/pdf/2405.04434

### Public resources
- DeepSeek-V4 paper review: https://andlukyane.com/blog/paper-review-deepseekv4 — why 1M context needs compression
- DeepSeek V4 compressed attention explainer: https://deepseek.ai/blog/deepseek-v4-compressed-attention — CSA in 60 seconds
- HF technical-report summary thread: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/discussions/129 — community notes
