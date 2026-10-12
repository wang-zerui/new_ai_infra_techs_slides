# Notion row (read-only, fetched 2026-10-13)

Page: DeepGEMM (unified tensor-core kernel library)
URL: https://app.notion.com/p/c0133413563f4ee390f9e4fb55027780
Database: 🏗️ Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties

- **Technique**: DeepGEMM (unified tensor-core kernel library)
- **Category**: Frameworks & Platform Stacks
- **Core Mechanism**: "Container entry: collects the core compute primitives of modern LLMs (GEMM,
  fused MoE, indexer scoring, HyperConnection) into a single CUDA codebase with very few kernels,
  all JIT-compiled at runtime. Sub-techniques are the rows pointing here."
- **Impact / Measured Results**: "Matches or beats expert-tuned libraries with a lightweight
  implementation; native support for variable per-expert token counts and fine-grained scaling."
- **Reference**: https://github.com/deepseek-ai/DeepGEMM
- **Source**: DeepGEMM

## Page body

Papers & slides (PDF)
- DeepSeek-V3 technical report (PDF) — https://arxiv.org/pdf/2412.19437
- DeepSeek-V4 technical report (PDF) — https://arxiv.org/pdf/2606.19348

Public resources
- DeepGEMM repo — https://github.com/deepseek-ai/DeepGEMM (official source)
- Exploring DeepGEMM FP8 kernels — https://kingsleykim.dev/blog/deepgemm/ (code-level walkthrough)
- DeepGEMM: unlocking the efficient FP8 GEMM library —
  https://medium.com/@ashlyajith619/deepgemm-unlocking-the-efficient-fp8-gemm-library-a5d7188c251b

## Children in the manifest (sibling decks)

10 (FP8 MQA logits / sparse indexer scoring), 30 (FP8/FP4/BF16 dense GEMM + UE8M0 layout),
32 (K-axis grouped GEMM), 33 (M-axis grouped GEMM, contiguous), 34 (masked grouped GEMM),
35 (Mega MoE fused megakernel), 39 (DeepJIT + PDL + determinism), 41 (Mega Gate / Mega mHC).
