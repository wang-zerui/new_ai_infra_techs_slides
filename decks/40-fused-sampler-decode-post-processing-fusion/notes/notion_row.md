# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/3070daaed63740e3be901cba2641e152
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)
Page last edited: 2026-10-12T19:13:53Z

## Properties
- Technique: Fused Sampler (decode post-processing fusion)
- Category: Kernel Fusion & Runtime
- Core Mechanism: Folds repetition penalty, temperature scaling, Top-K, Top-P, softmax, random sampling and penalty-mask updates into two kernels (temperature-only cases take an even lighter path); small top-k uses local heap/merge selection, and top-k reduction is merged with softmax statistics to cut vocabulary passes.
- Impact / Measured Results: Up to 8.5x over vLLM-style PyTorch / FlashInfer at vocab 120832 and batch 1-512 (largest win in small-batch RL rollout).
- Parent Library / Framework: HPC-Ops (https://app.notion.com/p/85f1abe2180f4b86bf1e9ea6d23705ce)
- Reference: https://github.com/Tencent/hpc-ops
- Source: HPC-Ops

## Body
### Papers & slides (PDF)
- FlashInfer: sampling and attention kernels (PDF) — https://arxiv.org/pdf/2501.01005
### Public resources
- HPC-Ops repo — https://github.com/Tencent/hpc-ops (fused sampler kernels)
- LMSYS: HPC-Ops × SGLang — https://www.lmsys.org/blog/2026-08-07-hpc-ops-sglang/ (sampling benchmark results)
