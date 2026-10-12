# Notion row — deck 12: Ascend · comm/compute fusion and PD-disaggregated scheduling

Fetched 2026-10-13 from https://app.notion.com/p/b1ba69567b7740fdbeead2013a32881c
(database "🏗️ Industrial Pretraining Architecture & Infra Tech Landscape (2026)"; page last edited
2026-09-21). Read-only copy; Notion was not modified.

## Properties

| Property | Value |
|---|---|
| Technique | Ascend · comm/compute fusion and PD-disaggregated scheduling |
| Category | Comm/Compute Overlap |
| Core Mechanism | Splits AllReduce into ReduceScatter + AllGather and pipelines it with tiled matrix computation to mask weaker interconnect bandwidth; layered with prefill/decode-disaggregated serving, prefix caching and MTP speculative decoding. |
| Impact / Measured Results | One domestic node matches a dual-GPU node; about 50% lower long-sequence serving cost. |
| Parent Library / Framework | Ascend adaptation stack (GLM-5) → deck 23 |
| Reference | https://arxiv.org/abs/2602.15763 |
| Source | GLM-5 series |

## Page body

### Papers & slides (PDF)
- GLM-5 technical report (PDF) — https://arxiv.org/pdf/2602.15763
- Flux: fine-grained comm/compute overlap (PDF) — https://arxiv.org/pdf/2406.06858

### Public resources
- GLM-5 blog — https://z.ai/blog/glm-5 — deployment claims
- "Huawei Pangu Ultra MoE on Ascend NPUs" —
  https://blog.gopenai.com/unlocking-trillion-parameter-ai-huaweis-pangu-ultra-moe-recipe-for-ascend-npus-23ecbf26b534
  — comm/compute recipe on the same hardware
- "GLM-5.3-Flash on 100k domestic chips" — https://www.globaltimes.cn/page/202608/1369157.shtml
