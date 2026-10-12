# Notion row (fetched 2026-10-12T19:13:49Z, read-only)

URL: https://app.notion.com/p/50c0563d46924e00830d20a62dd6cbde
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: NCCL Gin backend (GPU-initiated RDMA)
- Category: Expert-Parallel Communication
- Core Mechanism: Moves from NVSHMEM to NCCL Gin: GPU threads issue one-sided put / put-with-signal directly, header-only, reusing existing NCCL communicators with symmetric memory window registration; the whole library is JIT-compiled.
- Impact / Measured Results: Lighter dependency chain and simpler deployment, and the basis for 0-SM communication paths.
- Parent Library / Framework: DeepEP v2 (https://app.notion.com/p/d471baa2bcdf452592855568ee99f726)
- Reference: https://github.com/deepseek-ai/DeepEP
- Source: DeepEP

## Body
### Papers & slides (PDF)
- DeepSeek-V3 technical report (PDF): https://arxiv.org/pdf/2412.19437
- Fire-Flyer AI-HPC (PDF): https://arxiv.org/pdf/2408.14158
### Public resources
- DeepEP repo: https://github.com/deepseek-ai/DeepEP — Gin backend notes
- PyTorch SymmetricMemory: NVLink programmability: https://dev-discuss.pytorch.org/t/pytorch-symmetricmemory-harnessing-nvlink-programmability-with-ease/2798 — GPU-initiated comm background
- Talk: a new path toward multi-GPU programming: https://www.youtube.com/watch?v=yZWHacgYhuk
