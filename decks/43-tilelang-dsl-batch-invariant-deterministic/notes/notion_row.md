# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/22c7e2d5f253481981afb6c6ee06e3f5
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: TileLang DSL + batch-invariant deterministic kernels
- Category: Kernel Fusion & Runtime
- Core Mechanism: Kernels written in the TileLang domain-specific language replace hundreds of fine-grained ATen ops; batch-invariant attention uses two kernels (full waves on a single SM, tail wave via distributed shared memory) to preserve accumulation order; the backward pass replaces atomicAdd with per-SM accumulation buckets and deterministic reduction.
- Impact / Measured Results: Outputs are independent of batch position and parallel strategy and consistent between training and inference, sharply reducing irreproducibility in RL and debugging.
- Reference: https://arxiv.org/abs/2606.19348
- Source: DeepSeek V4

## Body
### Papers & slides (PDF)
- TileLang (PDF) — https://arxiv.org/pdf/2504.17577
- DeepSeek-V4 technical report (PDF) — https://arxiv.org/pdf/2606.19348
### Public resources
- Thinking Machines: Defeating nondeterminism in LLM inference — https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ (batch-invariance argument)
- TileLang repo — https://github.com/tile-ai/tilelang (the DSL)
- TileLang overview docs — https://tilelang.com/get_started/overview.html
