# Notion row (fetched 2026-10-13, read-only; page last edited 2026-10-12)

URL: https://app.notion.com/p/a065908961604600811834b0c1800744
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: Distributed checkpointing and elastic-resource training
- Category: Memory & Offload Scheduling
- Core Mechanism: VeOmni uses Torch Distributed Checkpoint for parallelism-agnostic save/load; K2 designs variable parallelism (16-way PP + 16-way EP + ZeRO-1 DP) for clusters whose resources change over time.
- Impact / Measured Results: Changing parallelism or hardware needs no checkpoint rewrite; training continues through node-count fluctuation.
- Parent Library / Framework: VeOmni (https://app.notion.com/p/9b9145984b6a433abf69f32d2c06db67)
- Reference: https://github.com/ByteDance-Seed/VeOmni
- Source: VeOmni, Kimi K2

## Body
### Papers & slides (PDF)
- VeOmni (PDF) — https://arxiv.org/pdf/2508.02317
- Kimi K2 technical report (PDF) — https://arxiv.org/pdf/2507.20534
### Public resources
- PyTorch: distributed checkpointing at scale — https://pytorch.org/blog/distributed-checkpoint-efficient-checkpointing-in-large-scale-jobs/ (DCP design)
- VeOmni repo — https://github.com/ByteDance-Seed/VeOmni (parallelism-agnostic save/load)
- Kimi K2 blog — https://moonshotai.github.io/Kimi-K2/ (variable-parallelism training)
