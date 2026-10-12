# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/99ccc6cd324244c0bf114f19b5440674
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: mHC: manifold-constrained hyper-connections
- Category: Model Shape & Depth Scaling
- Core Mechanism: Projects Hyper-Connections' residual mixing matrices onto the doubly-stochastic manifold (Birkhoff polytope) with Sinkhorn-Knopp, keeping spectral norm <= 1 and restoring identity-mapping behavior; companion kernel optimizations cut the extra memory traffic.
- Impact / Measured Results: Eliminates HC's gradient explosion at 27B (compound gain at depth 64 goes from 1e16 to about 1); a single Sinkhorn iteration suffices; also adopted by GLM-5.3.
- Reference: https://arxiv.org/abs/2512.24880
- Source: DeepSeek V4, GLM-5 series

## Body
### Papers & slides (PDF)
- mHC (PDF) — https://arxiv.org/pdf/2512.24880
- Hyper-Connections (PDF) — https://arxiv.org/pdf/2409.19606
### Public resources
- Hyper-Connections (paper) — https://arxiv.org/html/2409.19606v1 (the structure mHC constrains)
- ByteDance Seed: Hyper-Connections — https://seed.bytedance.com/zh/public_papers/hyper-connections
- DeepSeek-V4 paper review — https://andlukyane.com/blog/paper-review-deepseekv4 (mHC in V4)
