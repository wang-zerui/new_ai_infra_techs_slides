# Notion row — deck 23: Ascend adaptation stack (GLM-5)

Fetched 2026-10-13 from https://app.notion.com/p/3eda454335e04ef09f6a0b6991ee5608
(database "🏗️ Industrial Pretraining Architecture & Infra Tech Landscape (2026)"; page last edited
2026-09-21). Read-only copy; Notion was not modified.

## Properties

| Property | Value |
|---|---|
| Technique | Ascend adaptation stack (GLM-5) |
| Category | Frameworks & Platform Stacks |
| Core Mechanism | Container entry: GLM-5's day-one full-stack engineering for domestic GPU/NPU ecosystems, built on three pillars - aggressive quantization, high-performance fused operators, and inference-engine scheduling. Sub-techniques are the rows pointing here. |
| Impact / Measured Results | A single domestic GLM-5 node matches a dual-GPU cluster, with 50% lower long-sequence serving cost. |
| Reference | https://arxiv.org/abs/2602.15763 |
| Source | GLM-5 series, VeOmni |

## Page body

### Papers & slides (PDF)
- GLM-5 technical report (PDF) — https://arxiv.org/pdf/2602.15763

### Public resources
- GLM-5 blog — https://z.ai/blog/glm-5 — official model + infra claims
- "GLM-5.3-Flash runs on 100k domestic chips" — https://www.globaltimes.cn/page/202608/1369157.shtml
- "Huawei Pangu Ultra MoE recipe for Ascend NPUs" —
  https://blog.gopenai.com/unlocking-trillion-parameter-ai-huaweis-pangu-ultra-moe-recipe-for-ascend-npus-23ecbf26b534
  — closest public engineering deep dive

## Children (sub-technique rows pointing here)

- deck 12 — Ascend · comm/compute fusion and PD-disaggregated scheduling
- deck 37 — Ascend · super-operator fusion (MLAPO / Sparse FA / MoE Mega-Fusion)
- deck 44 — Ascend · W4A8/W8A8 hybrid quantization stack
