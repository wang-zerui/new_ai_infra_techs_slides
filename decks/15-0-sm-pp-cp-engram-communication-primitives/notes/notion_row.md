# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/c3f064d3adbd46a983472f8d7992f155
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)
Page last edited: 2026-10-12T19:13:49Z

## Properties
- Technique: 0-SM PP / CP / Engram communication primitives
- Category: Expert-Parallel Communication
- Core Mechanism: Beyond EP, experimental primitives: PP and Engram (remote memory access) go over RDMA with 0 SMs, CP uses the Copy Engine with 0 SMs; DP/TP all-gather and reduce-scatter implementations are included.
- Impact / Measured Results: Communication no longer competes with compute for SMs - the key direction toward "free" communication (currently experimental).
- Parent Library / Framework: DeepEP v2 (https://app.notion.com/p/d471baa2bcdf452592855568ee99f726)
- Reference: https://github.com/deepseek-ai/DeepEP
- Source: DeepEP

## Body
### Papers & slides (PDF)
- Fire-Flyer AI-HPC (PDF) — https://arxiv.org/pdf/2408.14158
- Flux: fine-grained comm/compute overlap (PDF) — https://arxiv.org/pdf/2406.06858
### Public resources
- DeepEP repo — https://github.com/deepseek-ai/DeepEP — experimental primitives
- PyTorch SymmetricMemory — https://dev-discuss.pytorch.org/t/pytorch-symmetricmemory-harnessing-nvlink-programmability-with-ease/2798 — copy-engine / 0-SM style transfers
- At Scale talk: PyTorch Symmetric Memory — https://atscaleconference.com/videos/pytorch-symmetric-memory-a-new-paradigm-for-programming-distributed-ai/
