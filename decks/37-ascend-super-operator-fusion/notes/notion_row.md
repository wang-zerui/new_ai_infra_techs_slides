# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/5926e0ea48474805988cb4d1419bac18
Database: 🏗️ Industrial Pretraining Architecture & Infra Tech Landscape (2026)
Page last edited: 2026-09-21

## Properties
- Technique: Ascend · super-operator fusion (MLAPO / Sparse FA / MoE Mega-Fusion)
- Category: Kernel Fusion & Runtime
- Core Mechanism: Coarse-grained fusion tuned to domestic NPU memory behavior: Lightning Indexer fuses
  score+ReLU+TopK into one op, Sparse Flash Attention fuses sparse indexing with attention, MLAPO collapses
  13 small ops into one super-operator, and MoE routing + permute + GroupedGEMM become a Mega-Fusion.
- Impact / Measured Results: Sharply reduces kernel-launch and intermediate memory traffic - the key to
  domestic hardware matching GPUs.
- Parent Library / Framework: Ascend adaptation stack (GLM-5) → deck 23
- Reference: https://arxiv.org/abs/2602.15763
- Source: GLM-5 series

## Body
### Papers & slides (PDF)
- GLM-5 technical report (PDF) — https://arxiv.org/pdf/2602.15763
- DeepSeek-V2 / MLA (PDF) — https://arxiv.org/pdf/2405.04434
### Public resources
- Huawei Pangu Ultra MoE recipe for Ascend NPUs (blog) —
  https://blog.gopenai.com/unlocking-trillion-parameter-ai-huaweis-pangu-ultra-moe-recipe-for-ascend-npus-23ecbf26b534
- GLM-5 blog — https://z.ai/blog/glm-5
- Podcast: DeepSeek V4 and Huawei Ascend NPU performance — https://www.youtube.com/watch?v=y2vz0REr5sQ

(For the blog item I read the underlying primary source instead: the Pangu Ultra MoE paper,
arXiv 2505.04519. The z.ai blog and the podcast were not used; nothing on the slides depends on them.)
