# Notion row (fetched 2026-10-13, read-only; page last edited 2026-09-21)

URL: https://app.notion.com/p/9b9145984b6a433abf69f32d2c06db67
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: VeOmni (model-centric omni-modal training framework)
- Category: Frameworks & Platform Stacks
- Core Mechanism: Container entry: ByteDance Seed's PyTorch-native omni-modal training framework; its core claim is decoupling parallelism/communication logic from model definitions, providing a composable distributed recipe zoo (FSDP1/2, Ulysses SP, EP) plus dynamic batching, DCP checkpointing and multiple hardware backends.
- Impact / Measured Results: More than 2800 tokens/s/GPU for a 30B omni-modal MoE on 128 GPUs with 160K context; adding a new modality takes very little code.
- Reference: https://arxiv.org/abs/2508.02317
- Source: VeOmni

## Body
### Papers & slides (PDF)
- VeOmni (PDF) — https://arxiv.org/pdf/2508.02317
- ZeRO (PDF) — https://arxiv.org/pdf/1910.02054
### Public resources
- VeOmni repo — https://github.com/ByteDance-Seed/VeOmni (recipe zoo and examples)
- ByteDance open-sources VeOmni — https://www.byteplus.com/blog/bytedance-veomni-open-source-ai-training (overview write-up)
