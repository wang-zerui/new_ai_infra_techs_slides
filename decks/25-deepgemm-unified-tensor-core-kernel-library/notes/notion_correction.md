# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Container entry: collects the core compute primitives of modern LLMs (GEMM, fused MoE, indexer scoring, HyperConnection) into a single CUDA codebase with very few kernels, all JIT-compiled at runtime. Sub-techniques are the rows pointing here.
**After:** Container entry: collects the core compute primitives of modern LLMs (FP8/FP4/BF16 GEMM, fused MoE, indexer scoring, HyperConnection, fused routing gate) into a single CUDA codebase of compact kernels (a few hundred lines per kernel family), all JIT-compiled at runtime through DeepJIT. Sub-techniques are the rows pointing here.
**Why:**
- "Very few kernels" no longer holds: 19 kernel implementation files, 9,593 lines of device code (S6.2, derived); each family is a few hundred lines (S2.1 "~300 lines" in 2025; 452 / 579 lines today, S6.3).
- The 2026-09 release adds Mega Gate (fused routing GEMM) and moves JIT into the separate DeepJIT (S1.1, S1.4, S3.4); GEMM precisions from S1.1.

## Impact / Measured Results
**Before:** Matches or beats expert-tuned libraries with a lightweight implementation; native support for variable per-expert token counts and fine-grained scaling.
**After:** Matches or exceeds expert-tuned libraries per the README, up to 1550 TFLOPS on H800; its published speedups are against DeepSeek's own CUTLASS 3.6-based kernels; native support for variable per-expert token counts and fine-grained scaling.
**Why:**
- The claim is the README's qualitative sentence (S1.2); the only published speedup baseline is "our internally and carefully optimized implementation based on CUTLASS 3.6" (S2.3), not a vendor library, so the baseline is named.
- 1550 TFLOPS on H800 from README news 2025.04.18 (S1.3).

## Other properties
none
