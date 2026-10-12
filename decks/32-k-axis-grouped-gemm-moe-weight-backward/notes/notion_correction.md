# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** k_grouped_fp8_gemm_tn_contiguous keeps M/N fixed and groups along K, purpose-built for MoE weight gradients (wgrad), with a matching K-grouped UE8M0 packing kernel.
**After:** k_grouped_fp8_gemm_tn_contiguous (SM100) and k_grouped_fp8_gemm_nt_contiguous (SM90) keep M/N fixed and group along K (each expert's tokens, segments aligned to 128), purpose-built for MoE weight gradients (wgrad); SM100 packs the scales to UE8M0 with a K-grouped packing kernel, SM90 keeps FP32 scales.
**Why:**
- `k_grouped_fp8_gemm_tn_contiguous` is SM100-only; Hopper has used `k_grouped_fp8_gemm_nt_contiguous` since Nov 2025 ([API] l.320-427, [V3-SM90]).
- The UE8M0 packing kernel applies to SM100 only; SM90 takes FP32 scales with gran_k = k_alignment = 128 ([LAYOUT] l.89-116, [RD] l.162).
- K is the token dimension, each expert's segment aligned to BLOCK_K = 128 ([SCHED] l.40-43, l.63-82).

## Impact / Measured Results
**Before:** Puts the MoE backward pass on the FP8 fine-grained-scaling path as well - critical for training, unlike inference-only operator libraries.
**After:** Puts MoE weight-gradient GEMMs on the FP8 fine-grained-scaling path as well; published kernel numbers cover only the May 2025 per-expert loop (900 to 1044 TFLOPS on Hopper, 4 experts); the Sept 2026 release reports about 20% faster MoE backward GEMMs.
**Why:**
- The kernel covers weight gradients, not the whole backward pass (Dgrad uses the M-grouped kernels, [RD] l.78).
- "Unlike inference-only operator libraries" has no support in the primary sources; dropped. "Critical for training" dropped as unsourced framing.
- Added the only published numbers, with their scope: PR #95 grouped wgrad table, a host loop over experts, kernel time only, GPU unnamed but Hopper-only at the time ([PR95], [V1-API], [V1-TEST] l.294); the single-kernel versions have only "about 20%" from the Sept 2026 release ([PR432]).

## Other properties
none
