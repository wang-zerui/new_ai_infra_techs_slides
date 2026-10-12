# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/0bb7a1acbf924329aab698ccc8c5ffee
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)
Last edited (Notion metadata): 2026-10-12T19:13:38Z

## Properties
- Technique: Mega Gate / Mega mHC fused kernels
- Category: Kernel Fusion & Runtime
- Core Mechanism: Fuses router gating (Mega Gate) and mHC / HyperConnection residual mixing (Mega mHC) each into a single kernel, avoiding the swarm of small ops and intermediate tensors these new architectures introduce.
- Impact / Measured Results: Brings mHC's extra memory traffic into an acceptable range - a key enabler for shipping new residual structures to production.
- Parent Library / Framework: DeepGEMM (https://app.notion.com/p/c0133413563f4ee390f9e4fb55027780 = deck 25)
- Reference: https://github.com/deepseek-ai/DeepGEMM
- Source: DeepGEMM, DeepSeek V4

## Body
### Papers & slides (PDF)
- mHC: manifold-constrained hyper-connections (PDF) — https://arxiv.org/pdf/2512.24880
- Hyper-Connections (PDF) — https://arxiv.org/pdf/2409.19606
### Public resources
- DeepGEMM repo — https://github.com/deepseek-ai/DeepGEMM (Mega Gate / Mega mHC kernels)
- Hyper-Connections paper — https://arxiv.org/html/2409.19606v1 (the structure being fused)
- ByteDance Seed: Hyper-Connections — https://seed.bytedance.com/zh/public_papers/hyper-connections
