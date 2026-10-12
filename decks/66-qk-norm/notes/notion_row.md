# Notion row (fetched 2026-10-13, read-only)

Page: https://app.notion.com/p/319387d19fc64c6d8079c56a5c3b7a69
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)
Last edited: 2026-09-21T13:07:12Z

## Properties

| Property | Value |
|---|---|
| Technique | QK-Norm |
| Category | Optimizer & Stability |
| Core Mechanism | Normalizes Q/K so the dynamic range of attention logits stays in a stable band (the same problem QK-Clip attacks from the other side). |
| Impact / Measured Results | Together with Muon, supports stable convergence at larger batch sizes. |
| Reference | https://arxiv.org/abs/2508.06471 |
| Source | GLM-4.5 |

## Body

### Papers & slides (PDF)
- GLM-4.5 technical report (PDF): https://arxiv.org/pdf/2508.06471
- Kimi K2 (QK-Clip) (PDF): https://arxiv.org/pdf/2507.20534

### Public resources
- GLM-4.5 blog: https://z.ai/blog/glm-4.5 (stability recipe)
- Kimi K2 blog: https://moonshotai.github.io/Kimi-K2/ (QK-Clip, the alternative fix)
