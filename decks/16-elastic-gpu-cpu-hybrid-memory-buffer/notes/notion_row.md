# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/f8ca122ace3b41dba7d867f9c2731da3
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)
Page last edited: 2026-10-12T19:13:49Z

## Properties
- Technique: Elastic GPU/CPU hybrid memory buffer
- Category: Expert-Parallel Communication
- Core Mechanism: One contiguous virtual address range backed by a mix of GPU and CPU physical memory, transparently supporting Engram and imbalanced EP; EP replay compresses intermediate buffers.
- Impact / Measured Results: Mitigates V2's larger buffer footprint so worst-case load no longer dictates the memory budget.
- Parent Library / Framework: DeepEP v2 (https://app.notion.com/p/d471baa2bcdf452592855568ee99f726)
- Reference: https://github.com/deepseek-ai/DeepEP
- Source: DeepEP

## Body
### Papers & slides (PDF)
- DeepSeek-V3 technical report (PDF) — https://arxiv.org/pdf/2412.19437
### Public resources
- DeepEP repo — https://github.com/deepseek-ai/DeepEP — buffer design
- Azure: DeepEP performance guide — https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/achieving-optimal-performance-for-deepseek-expert-parallelism-deepep-on-azure/4414699 — memory/SM budgeting
