# Notion row (fetched 2026-10-13, read-only; page last edited 2026-09-21)

URL: https://app.notion.com/p/5cfd3adb77f74e0c8d09435a3af5d9b3
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: Dynamic batching / sequence packing
- Category: Memory & Offload Scheduling
- Core Mechanism: Buffers samples and packs them to a target sequence length instead of uniform padding; the MCore counterpart is the THD packed format (e.g. GLM-5.2's 128K THD packed CP training).
- Impact / Measured Results: Large gains in effective FLOP utilization for multimodal and variable-length data.
- Parent Library / Framework: VeOmni (https://app.notion.com/p/9b9145984b6a433abf69f32d2c06db67)
- Reference: https://arxiv.org/abs/2508.02317
- Source: VeOmni, Megatron-LM / MCore, GLM-5 series

## Body
### Papers & slides (PDF)
- VeOmni (PDF) — https://arxiv.org/pdf/2508.02317
- Scalable MoE training with Megatron-Core (PDF) — https://arxiv.org/pdf/2603.07685
### Public resources
- NVIDIA: variable-length training with dynamic CP and Megatron-Core — https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/ (packing + CP)
- What is sequence packing? — https://www.ai21.com/glossary/foundational-llm/what-is-sequence-packing/ (short primer)
- VeOmni repo — https://github.com/ByteDance-Seed/VeOmni (dynamic batching implementation)
