# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** fp8_gemm_{nt,nn,tn,tt}; SM90 supports only the NT layout with FP32 scale factors, while SM100 supports all layouts with 4x UE8M0 packed into an int; LHS scale factors must be TMA-aligned and transposed.
**After:** unchanged
**Why:**
- Matches the README word for word (S1.1, S1.2); the `fp8_gemm_*` names are kept as aliases of `fp8_fp4_gemm_*` (S3.5). Omitting the two SM90 kernels (1d1d / 1d2d, S3.1) and SM100's gran_k 32 (S2.1) is harmless.

## Impact / Measured Results
**Before:** Peaked at 1550 TFLOPS on H800, matching or beating expert-tuned libraries with a minimal implementation.
**After:** Peaked at 1550 TFLOPS on H800 (April 2025 release note); per-shape H800 results are 1.1x to 2.7x over DeepSeek's internal tuned CUTLASS 3.6 implementation, with a core kernel of about 300 lines.
**Why:**
- 1550 TFLOPS is the README's peak across shapes (S1.4, news 2025-04-18); kept, dated.
- The only measured comparison is against DeepSeek's own CUTLASS 3.6-based implementation on H800 SXM5 (1.1x to 2.7x, S6); "expert-tuned libraries" in general is not what was measured, so the baseline is named.
- "Minimal implementation" made concrete as the README's "about 300 lines" core kernel (S6).

## Other properties
none (title keeps "BF16": DeepGEMM does expose `bf16_gemm_*`, though it usually dispatches to cuBLASLt, S3.4)
