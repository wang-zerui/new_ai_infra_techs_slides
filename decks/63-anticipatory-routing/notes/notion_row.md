# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/ecc924eb12db4b0fa39609cf42a53369
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: Anticipatory routing (asynchronous router parameters)
- Category: Optimizer & Stability
- Core Mechanism: Deliberately desynchronizes backbone and router updates: features use the newest parameters while routing decisions use cached older ones, breaking the "anomaly -> routing drift -> larger anomaly" feedback loop.
- Impact / Measured Results: With mHC and Muon, suppresses loss spikes and divergence in trillion-scale MoEs.
- Reference: https://arxiv.org/abs/2606.19348
- Source: DeepSeek V4

## Body
### Papers & slides (PDF)
- DeepSeek-V4 technical report (PDF) — https://arxiv.org/pdf/2606.19348
### Public resources
- DeepSeek-V4 paper review — https://andlukyane.com/blog/paper-review-deepseekv4 (stability stack of V4)
- HF technical-report summary thread — https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/discussions/129
