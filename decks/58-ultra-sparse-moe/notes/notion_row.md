# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/b2283eba9cf14a8fb2fa3f7b2ba4127f
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: Ultra-sparse MoE (fine-grained routed experts + shared expert)
- Category: MoE & Routing
- Core Mechanism: DeepSeekMoE-style fine-grained expert pool plus one shared expert; total and activated parameters fully decoupled, with sparsity chosen by scaling laws.
- Impact / Measured Results: V4-Pro 1.6T/49B active, V4-Flash 284B/13B; K2 1.04T/32B; GLM-5 744B/40B.
- Reference: https://arxiv.org/abs/2606.19348
- Source: DeepSeek V4, Kimi K2, GLM-5 series

## Body
### Papers & slides (PDF)
- DeepSeek-V4 technical report (PDF) — https://arxiv.org/pdf/2606.19348
- DeepSeekMoE (PDF) — https://arxiv.org/pdf/2401.06066
### Public resources
- DeepSeek-V4 paper review — https://andlukyane.com/blog/paper-review-deepseekv4 (sparsity and scaling choices)
- Kimi K2 blog — https://moonshotai.github.io/Kimi-K2/ (1T/32B sparsity design)
- GLM-5 blog — https://z.ai/blog/glm-5
