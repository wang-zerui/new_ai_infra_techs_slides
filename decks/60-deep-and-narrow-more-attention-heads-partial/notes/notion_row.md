# Notion row (fetched 2026-10-13, read-only)

Page: https://app.notion.com/p/ba32c8b828d748ea9f32a65e15598576
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)
Last edited: 2026-09-21T13:07:10Z

## Properties

| Property | Value |
|---|---|
| Technique | Deep-and-narrow + more attention heads + partial RoPE |
| Category | Model Shape & Depth Scaling |
| Core Mechanism | Narrower than DeepSeek-V3/K2 (hidden dim, number of routed experts) but deeper (89 MoE layers); GQA with partial RoPE and 2.5x more heads (96 heads at 5120 hidden). |
| Impact / Measured Results | Deeper models reason better; more heads do not lower training loss yet keep improving MMLU/BBH. |
| Reference | https://arxiv.org/abs/2508.06471 |
| Source | GLM-4.5 |

## Body

### Papers & slides (PDF)
- GLM-4.5 technical report (PDF): https://arxiv.org/pdf/2508.06471

### Public resources
- GLM-4.5 blog: https://z.ai/blog/glm-4.5 (depth/width and head-count choices)
- GLM-5 repo: https://github.com/zai-org/GLM-5 (configs for the successor models)
