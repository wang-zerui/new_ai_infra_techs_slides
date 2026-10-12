# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/38d6d033f1b943b3b79615c26f4f0a2e
Database: 🏗️ Industrial Pretraining Architecture & Infra Tech Landscape (2026)
Page last edited: 2026-09-21

## Properties
- Technique: Ascend · W4A8/W8A8 hybrid quantization stack
- Category: Low-Precision & Quantization
- Core Mechanism: QuaRot rotation preprocessing removes activation outliers, Flex SmoothQuant searches
  per-layer migration strength, SSZ (scale-shift-zero) handles weight quantization; W4A8 for MoE experts,
  W8A8 kept for sensitive layers.
- Impact / Measured Results: Halves weight memory at near-lossless accuracy, with large throughput gains
  on domestic chips.
- Parent Library / Framework: Ascend adaptation stack (GLM-5) → deck 23
- Reference: https://arxiv.org/abs/2602.15763
- Source: GLM-5 series

## Body
### Papers & slides (PDF)
- GLM-5 technical report (PDF) — https://arxiv.org/pdf/2602.15763
- SmoothQuant (PDF) — https://arxiv.org/pdf/2211.10438
- QuaRot (PDF) — https://arxiv.org/pdf/2404.00456
### Public resources
- GLM-5 blog — https://z.ai/blog/glm-5 (deployment-side quantization claims)
- Huawei Pangu Ultra MoE on Ascend NPUs (blog) —
  https://blog.gopenai.com/unlocking-trillion-parameter-ai-huaweis-pangu-ultra-moe-recipe-for-ascend-npus-23ecbf26b534
- How GLM-5 runs on Huawei chips — https://letsdatascience.com/blog/china-trained-frontier-ai-model-glm-5-without-nvidia

(The two third-party blogs and the z.ai blog were not used as evidence. Instead I read the primary
sources: the GLM-5 report, the QuaRot and SmoothQuant papers, the AWQ paper, and the vLLM-Ascend
quantization code and GLM-5 deployment guide. Nothing on the slides depends on a blog post.)
