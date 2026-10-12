# Notion row (fetched 2026-10-13, read-only copy)

URL: https://app.notion.com/p/dfc27fd2e9d443c9a0a0efd215bdc3da
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)
Page last edited: 2026-10-12T19:14:19Z

## Properties

- **Technique:** MoonEP · dynamic redundant experts + online GPU planning + zero-copy static shapes
- **Category:** Expert-Parallel Communication
- **Core Mechanism:** Three co-designed pieces in one library: (1) replicates hot experts online according to the current batch's routing distribution so every EP rank receives exactly S x K / EP tokens, with the backward pass reducing redundant replicas' gradients back to the home rank (mathematically equivalent to no redundancy); (2) a GPU kernel solves redundant-expert placement + token assignment online for every micro-batch, with no host involvement and negligible cost relative to the all-to-all; (3) permute/unpermute is fused into dispatch/combine with a fixed static S x K buffer instead of a dynamic S x K x R shape, removing per-layer host-device syncs and intermediate copies.
- **Impact / Measured Results:** Removes the EP dispatch long tail and makes GroupedGEMM a fully static load, with no offline EPLB statistics; online planning overhead is about zero and routing drift is tracked batch by batch; static shapes give lower memory and copy overhead, are CUDA Graph friendly, and have no synchronization bubbles.
- **Parent Library / Framework:** https://app.notion.com/p/65b197b767bb4d5c9dc17bbce5c0ca13 (MoonEP, deck 27)
- **Reference:** https://github.com/MoonshotAI/MoonEP
- **Source:** MoonEP

## Body

### Papers & slides (PDF)
- Kimi K3 technical report (PDF): https://arxiv.org/pdf/2607.24653
- DeepSeek-V3 (redundant experts / EPLB) (PDF): https://arxiv.org/pdf/2412.19437
- MegaBlocks: dropless MoE (PDF): https://arxiv.org/pdf/2211.15841

### Public resources
- MoonEP repo: https://github.com/moonshotAI/moonep — algorithm, planner kernel, static-shape dispatch/combine, benchmarks
- MarkTechPost: Moonshot open-sources MoonEP: https://www.marktechpost.com/2026/07/29/moonshot-ai-open-sources-moonep-a-perfectly-balanced-expert-parallelism-library-for-moe-training/ — summary write-up
- DeepSeek EPLB: https://github.com/deepseek-ai/eplb — the offline redundant-expert baseline it replaces
- DeepSeek LPLB (LP-based balancer): https://github.com/deepseek-ai/LPLB — alternative planning formulation
- DeepEP repo: https://github.com/deepseek-ai/DeepEP — dynamic-shape baseline

## Slug note
`data/manifest.json` gives the slug `20-moonep-dynamic-redundant-experts-online-gpu` (used for this folder); the task
brief spelled it `20-moonep-dynamic-redundant-experts-online-gpu-plan` (the raw 48-character truncation).
