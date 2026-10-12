# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/9069ea0863a741658d052a19a186169b
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: Distributed engineering of Muon at scale
- Category: Optimizer & Stability
- Core Mechanism: Muon needs full gradient matrices, which conflicts with ZeRO element-wise sharding: V4 uses knapsack-based hybrid ZeRO bucketing (<10% padding memory), optimizes MoE params separately and syncs their gradients in BF16 with stochastic rounding; GLM-5 uses zero-redundancy communication so every DP rank need not all-gather full params.
- Impact / Measured Results: Halves MoE gradient-sync traffic; removes naive Muon's transient memory spikes and redundant communication.
- Reference: https://arxiv.org/abs/2606.19348
- Source: DeepSeek V4, GLM-5 series

## Body
### Papers & slides (PDF)
- Muon is scalable for LLM training (PDF) — https://arxiv.org/pdf/2502.16982
- DeepSeek-V4 technical report (PDF) — https://arxiv.org/pdf/2606.19348
### Public resources
- Muon is scalable for LLM training (PDF) — https://arxiv.org/pdf/2502.16982 (distributed Muon and ZeRO interaction)
- Keller Jordan: Muon optimizer — https://kellerjordan.github.io/posts/muon
- GLM-5 blog — https://z.ai/blog/glm-5 (zero-redundancy optimizer communication)
