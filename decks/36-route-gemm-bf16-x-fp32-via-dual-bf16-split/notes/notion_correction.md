# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Splits FP32 weights into a high BF16 term plus a low residual BF16 term (fixed scale 1/256) and fuses two BF16 Tensor-Core GEMMs in one kernel with shared input gathering, register-resident accumulators and a single result write.
**After:** Splits FP32 weights into a high BF16 term plus a low residual BF16 term (fixed scale 1/256) and fuses two BF16 Tensor-Core GEMMs in one kernel that loads each input tile once for both, keeps both accumulators in registers and writes the result once.
**Why:**
- There is no gather: one TMA load of each X tile per K step feeds both WGMMAs (W_low and W_high), two register accumulators, one epilogue `acc_low * scale + acc_high` and one TMA store ([KER] lines 242-364).

## Impact / Measured Results
**Before:** Up to 3.22x over cuBLAS FP32/TF32 while keeping FP32 accuracy; suits precision-sensitive small matrices such as MoE router GEMMs and the state-compress GEMMs of linear/sparse attention.
**After:** Up to 3.22x over FP32 cuBLAS and 1.78x over TF32 cuBLAS (H20, K=4096, N=192) at close-to-FP32 accuracy (max abs error 0.00177 vs 0.06464 for TF32); suits precision-sensitive small matrices such as MoE router GEMMs and the state-compress GEMMs of linear/sparse attention.
**Why:**
- 3.22x is over FP32 cuBLAS only (M=512); over TF32 cuBLAS the maximum is 1.78x (M=96), range 1.25x to 1.78x ([LM] Table 2 and operator-benchmark paragraph; [RM] lists both baselines with one number).
- "Keeping FP32 accuracy" overstates: the split keeps about 16 of FP32's 24 significand bits, and the measured max abs error vs the FP32 cuBLAS reference is 0.00177, not zero; TF32 gives 0.06464 on the same sweep ([LM] Highlights; analytic bound in sources.md). The blog's own wording is "precision close to a full FP32 GEMM".
- Use cases kept: MoE router and sparse/linear-attention state-compress GEMMs are in the README Operator Catalog and Performance rows ([RM]).

## Other properties
none
